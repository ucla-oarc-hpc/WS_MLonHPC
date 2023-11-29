import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import time
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f"Using device: {device}")

# Setting a seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # For CUDA

# Function to generate random DNA sequences
def generate_random_dna(length):
    return ''.join(random.choice('ATCG') for _ in range(length))

# Generating larger synthetic dataset
sequence_length = 200  # Increased length
num_sequences = 2000   # Increased number of sequences

gene_sequences = [generate_random_dna(sequence_length) for _ in range(num_sequences)]
non_gene_sequences = [generate_random_dna(sequence_length) for _ in range(num_sequences)]
sequences = gene_sequences + non_gene_sequences
labels = [1]*num_sequences + [0]*num_sequences

# Encoding DNA sequences into numerical format
label_encoder = LabelEncoder()
encoded_sequences = [label_encoder.fit_transform(list(seq)) for seq in sequences]
encoded_sequences = torch.tensor(encoded_sequences)
labels = torch.tensor(labels)

# Creating DataLoader for batch processing
dataset = TensorDataset(encoded_sequences, labels)
train_loader, test_loader = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_loader, batch_size=32, shuffle=True)  # Using a batch size of 32
test_loader = DataLoader(test_loader, batch_size=32)  # Same for test set

# Neural Network Model
class DNANet(nn.Module):
    def __init__(self):
        super(DNANet, self).__init__()
        self.fc1 = nn.Linear(sequence_length, 100)  # More complex model
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model, Loss Function, and Optimizer
model = DNANet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
start_time = time.time()
for epoch in range(100):  # Reduced number of epochs for time saving
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(seq.float())
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/100, Loss: {loss.item()}')

end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for seq, label in test_loader:
        seq, label = seq.to(device), label.to(device)
        outputs = model(seq.float())
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        actuals.extend(label.cpu().numpy())

accuracy = np.mean(np.array(predictions) == np.array(actuals))
precision = precision_score(actuals, predictions)
recall = recall_score(actuals, predictions)
f1 = f1_score(actuals, predictions)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

