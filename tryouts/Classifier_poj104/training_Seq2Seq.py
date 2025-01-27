import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.fx
from torch import GradScaler
from torch import autocast
import os
torch._dynamo.config.verbose = True


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    dropout: float = 0.2


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # type: ignore
    freqs = torch.outer(t, freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # if torch.isnan(xq_out).any() or torch.isnan(xk_out).any():
    #     raise ValueError("NaN values found in apply_rotary_emb output")
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            dtype=torch.float32,
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            dtype=torch.float32,
        ).cuda()

        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.dropout = args.dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # if torch.isnan(x).any():
        #     raise ValueError("NaN values found in the input tensor x")
        x = self.attention_norm(x)

        # if torch.isnan(x).any():
        #     raise ValueError("NaN values found in the x ")

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # if torch.isnan(xq).any() or torch.isnan(xk).any() or torch.isnan(xv).any():
        #     raise ValueError("NaN values found in the xq or xk or xv")

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # if torch.isnan(xq).any() or torch.isnan(xk).any() or torch.isnan(xv).any():
        #     raise ValueError("NaN values found in the inputs to apply_rotary_emb")

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # if torch.isnan(xq).any() or torch.isnan(xk).any() or torch.isnan(xv).any():
        #     raise ValueError("NaN values found in the xq or xk or xv after the apply the rotary ")

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = repeat_kv(self.cache_k[:bsz, : start_pos + seqlen], self.n_rep)
        values = repeat_kv(self.cache_v[:bsz, : start_pos + seqlen], self.n_rep)

        # if torch.isnan(keys).any() or torch.isnan(values).any() :
        #     raise ValueError("Nan values in the keys and the values")

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                keys,
                values,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            # num_nan_scores = torch.isnan(scores).sum().item()
            # print(num_nan_scores)
            # if torch.isnan(scores).any() :
            #     raise ValueError("Nan values in the scores")

            # print(mask.shape)
            # print(scores.shape)

            if mask is not None:
                # if torch.isnan(mask).any():
                #     raise ValueError("NaN values detected in the mask in the slkdf sd")

                # Unsqueeze and repeat mask to match the shape of scores
                # Current mask shape: [32, 1, 1, 512]
                # Target shape: [32, 8, 512, 512]
                # mask = mask.unsqueeze(1).repeat(-1, scores.size(1), -1, -1)  # Repeat along head dimension
                # print("Mask after reshaping:", mask.shape)

                # Add the mask
                scores = scores + mask.detach()

            # if torch.isnan(scores).any() :
            #     raise ValueError("Nan values in the scores after mask")

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # num_nan_mask = torch.isnan(mask).sum().item()
            # num_nan_scores = torch.isnan(scores).sum().item()
            # print(num_nan_mask , num_nan_scores)
            # if torch.isnan(scores).any() :
            #     raise ValueError("Nan values in the scores after mask and softmax")
            output = self.attn_dropout(scores)
            output = torch.matmul(output, values)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # if torch.isnan(output).any() :
        #     raise ValueError("Nan value ecounter in the output")

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x):
        # if torch.isnan(x).any():
        #     raise ValueError("NaN values found in the input tensor x in the forward of the feedforward")
        x = self.ffn_norm(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # if torch.isnan(x).any():
        #     raise ValueError("NaN values found in the input tensor x in the trasnformer block")
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # if torch.isnan(h).any():
        #     raise ValueError("NaN values found in the input tensor h in the trasnformer block")
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, num_classes: int):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Linear(300, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, num_classes, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.tok_embeddings.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        masks: torch.Tensor,
        start_pos: int,
    ):
        # if torch.isnan(tokens).any():
        #     raise ValueError("NaN values found in the input tensor tokens in the forward of the transformer")
        _bsz, seqlen, embedding = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].detach()

        # Print unique values and their counts in the mask tensor
        # unique_values, counts = torch.unique(masks, return_counts=True)
        # print(f"Mask unique values: {unique_values}")
        # print(f"Mask counts: {counts}")
        # print(masks.shape)

        # if torch.isnan(masks).any():
        #     raise ValueError("NaN values detected in the masks tensor")

        masks = (masks > 0).float()
        masks = torch.nan_to_num(masks, nan=0.0)

        # Debug: Check unique values in the mask
        unique_values, counts = torch.unique(masks, return_counts=True)
        # print(f"Mask unique values: {unique_values}, counts: {counts}")

        # Expand the mask and apply -inf to masked positions
        safe_mask = masks.masked_fill(masks == 0, float("-inf"))

        # Expand the mask
        mask = safe_mask[:, None, None, :]

        # if torch.isnan(mask).any():
        #     raise ValueError("NaN values detected in the masks tensor after the inf added")
        # print(mask.shape)

        # if 1 == 1 :
        #     raise ValueError("Nothing")
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        h = h.mean(dim=1)  # Aggregate over the sequence length dimension
        output = self.output(h).float()
        return output


# Custom Dataset class that loads data in batches
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        self.X_data = self.data["data"].float()
        self.y_data = self.data["labels"].float()
        self.masks = self.data["masks"].float()
        self.lengths = [len(seq) for seq in self.X_data]

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        sequence = self.X_data[idx]
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        label = self.y_data[idx]
        mask = self.masks[idx]
        return sequence, length, label, mask


# Corrected collate_fn
def collate_fn(batch):
    sequences = [item[0] for item in batch]  # Extract sequences
    lengths = [item[1] for item in batch]  # Extract lengths
    labels = [item[2] for item in batch]  # Extract labels
    masks = [item[3] for item in batch]  # Extract masks

    # Pad sequences to the maximum length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

    # Convert lengths and labels into tensors
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)  # Stack one-hot encoded labels into a single tensor

    return padded_sequences, lengths, labels, padded_masks


if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ModelArgs(
        dim=512,
        n_layers=6,
        n_heads=8,
        vocab_size=300,
        max_batch_size=32,
        max_seq_len=512,
        dropout=0.2,
    )

    num_classes = 104
    model = Transformer(args, num_classes).to(device)
    start_pos = 0

    # Create dataset and dataloader
    train_dataset = CustomDataset(
        "/home/es21btech11028/IR2Vec/tryouts/Data_things/train_val_test_instrcution/train_data.pt"
    )
    val_dataset = CustomDataset(
        "/home/es21btech11028/IR2Vec/tryouts/Data_things/train_val_test_instrcution/val_data.pt"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.max_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.max_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Check if a saved model exists and load it
    best_model_path = "best_transformer_model.pth"
    start_epoch = 0
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if os.path.exists(best_model_path):
        logger.info(f"Loading model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        logger.info("Model loaded successfully")

    # Training loop
    num_epochs = 10
    model.train()
    scaler = GradScaler()

    # Initialize lists to store logging information
    training_losses = []
    validation_losses = []
    validation_accuracies = []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        )
        for tokens, lengths, labels, masks in progress_bar:
            optimizer.zero_grad()
            tokens, lengths, labels, masks = (
                tokens.to(device),
                lengths.to(device),
                labels.to(device),
                masks.to(device),
            )

            # Forward pass with autocast
            with autocast(device_type="cuda"):
                outputs = model(tokens, lengths, masks, start_pos)
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        training_losses.append(avg_loss)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for tokens, lengths, labels, masks in val_dataloader:
                tokens, lengths, labels, masks = (
                    tokens.to(device),
                    lengths.to(device),
                    labels.to(device),
                    masks.to(device),
                )
                with autocast(device_type="cuda"):
                    outputs = model(tokens, lengths, masks, start_pos)
                    loss = criterion(outputs, torch.argmax(labels, dim=1))
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(labels, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(all_labels, all_preds, normalize=True)
        validation_losses.append(avg_val_loss)
        validation_accuracies.append(val_accuracy)
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

        # Step the scheduler
        # scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), "transformer_model.pth")

    # Save the logging information
    log_data = {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "validation_accuracies": validation_accuracies,
    }
    torch.save(log_data, "training_log.pth")
