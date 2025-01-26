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

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
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
            ), dtype=torch.float32
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ), dtype=torch.float32
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = repeat_kv(self.cache_k[:bsz, : start_pos + seqlen], self.n_rep).detach()
        values = repeat_kv(self.cache_v[:bsz, : start_pos + seqlen], self.n_rep).detach()

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.detach()
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
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

    def forward(self, x):
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
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, num_classes: int):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.tok_embeddings = nn.Linear(300, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, num_classes, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, masks: torch.Tensor, start_pos: int):
        _bsz, seqlen , embedding = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].detach()

        mask = masks[:, None, None, :].float() * float('-inf')

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
        self.X_data = self.data['data'].float()
        self.y_data = self.data['labels'].float()
        self.masks = self.data['masks'].float()
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
    lengths = [item[1] for item in batch]    # Extract lengths
    labels = [item[2] for item in batch]     # Extract labels
    masks = [item[3] for item in batch]      # Extract masks

    # Pad sequences to the maximum length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

    # Convert lengths and labels into tensors
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)  # Stack one-hot encoded labels into a single tensor

    return padded_sequences, lengths, labels, padded_masks

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = ModelArgs(
        dim=512,
        n_layers=6,
        n_heads=8,
        vocab_size=300,
        max_batch_size=32,
        max_seq_len=512,
    )

    num_classes = 104
    model = Transformer(args, num_classes).to(device)
    start_pos = 0

    # Create dataset and dataloader
    train_dataset = CustomDataset('/home/es21btech11028/IR2Vec/tryouts/Data_things/train_val_test_instrcution/train_data.pt')
    val_dataset = CustomDataset('/home/es21btech11028/IR2Vec/tryouts/Data_things/train_val_test_instrcution/val_data.pt')
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.max_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.max_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    num_epochs = 10
    # model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for tokens, lengths, labels, masks in progress_bar:
            # Move to device
            optimizer.zero_grad()
            tokens, lengths, labels, masks = tokens.to(device), lengths.to(device), labels.to(device), masks.to(device)

            # Forward pass
            outputs = model(tokens, lengths, masks, start_pos)
            # print(outputs.dtype)
            # Compute loss
            loss = criterion(outputs,labels)
            # print(loss.dtype)
            # Backward pass and optimizationimport os
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for tokens, lengths, labels, masks in val_dataloader:
                tokens, lengths, labels, masks = tokens.to(device), lengths.to(device), labels.to(device), masks.to(device)
                outputs = model(tokens, lengths, masks, start_pos)
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(labels, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(all_labels, all_preds, normalize=True)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), 'transformer_model.pth')
