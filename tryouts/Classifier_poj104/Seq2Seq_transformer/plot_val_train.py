import re
import matplotlib.pyplot as plt

def extract_losses_from_log(file_path):
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    with open(file_path, 'r') as file:
        for line in file:
            # Extract training loss
            training_loss_match = re.search(r'Training Loss: ([0-9.]+)', line)
            if training_loss_match:
                training_losses.append(float(training_loss_match.group(1)))

            # Extract validation loss and accuracy
            validation_loss_match = re.search(r'Validation Loss: ([0-9.]+), Validation Accuracy: ([0-9.]+)', line)
            if validation_loss_match:
                validation_losses.append(float(validation_loss_match.group(1)))
                validation_accuracies.append(float(validation_loss_match.group(2)))

    return training_losses, validation_losses, validation_accuracies

# Extract data from both log files
training_losses_1, validation_losses_1, validation_accuracies_1 = extract_losses_from_log('/home/es21btech11028/IR2Vec/tryouts/Classifier_poj104/Seq2Seq_transformer/output1.log')
training_losses_2, validation_losses_2, validation_accuracies_2 = extract_losses_from_log('/home/es21btech11028/IR2Vec/tryouts/Classifier_poj104/Seq2Seq_transformer/output2.log')

# Combine the data
training_losses = training_losses_1 + training_losses_2
validation_losses = validation_losses_1 + validation_losses_2
validation_accuracies = validation_accuracies_1 + validation_accuracies_2
# print("zjdnsdf" , training_losses)
# Plot the data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_plot.png')