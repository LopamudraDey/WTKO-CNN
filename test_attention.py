import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################
# Dataset
############################################
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

############################################
# One-hot encoding
############################################
nucleotides = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1], 'N':[0,0,0,0]}

def one_hot_encode(sequences, seq_length=None):
    encoded_sequences = []
    for seq in sequences:
        seq = seq.upper()
        encoded = np.array([nucleotides.get(b,[0,0,0,0]) for b in seq])
        if seq_length:
            encoded = pad_sequences([encoded], maxlen=seq_length, padding='post', dtype='float32')[0]
        encoded_sequences.append(encoded)
    return np.array(encoded_sequences)

############################################
# Attention Layer
############################################
class Attention1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = self.attn(x)                  # [batch, seq_len, 1]
        weights = torch.softmax(weights, dim=1)
        out = torch.sum(x * weights, dim=1)     # [batch, features]
        return out, weights

############################################
# CNN + Attention Model
############################################
class SequenceModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=10)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=10)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.attn = Attention1D(128)
        self.fc1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0,2,1)                    # [batch, channels, seq_len]

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = x.permute(0, 2, 1)                  # [batch, seq_len, features]
        x, attn_weights = self.attn(x)

        x = F.relu(self.fc1(x))
        x = self.output(x)

        return x, attn_weights

############################################
# Load Data
############################################
wt_df = pd.read_csv('Eb3_WT_spe.csv', delimiter=";")
ko_df = pd.read_csv('Eb3_KO_spe.csv', delimiter=";")
wt_df['label'] = 1
ko_df['label'] = 0
n = min(len(wt_df), len(ko_df))
df = pd.concat([wt_df.sample(n, random_state=42),
                ko_df.sample(n, random_state=42)]).reset_index(drop=True)
df = df.dropna(subset=['sequence'])

train_df = df[~df['chrom'].isin(['chr6','chr7'])]
test_df = df[df['chrom'].isin(['chr6','chr7'])]

X_train = train_df['sequence'].values
y_train = train_df['label'].values
X_test = test_df['sequence'].values
y_test = test_df['label'].values

train_encoded = one_hot_encode(X_train, seq_length=500)
test_encoded = one_hot_encode(X_test, seq_length=500)

train_dataset = SequenceDataset(train_encoded, y_train)
test_dataset = SequenceDataset(test_encoded, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = SequenceModelWithAttention().to(device)
model.load_state_dict(torch.load("Eb3_model.pth"))
model.eval()

# ------------------------------
# Function: compute saliency for a sequence
# ------------------------------
def compute_saliency(model, sequence, seq_length=500):
    """
    sequence: string DNA sequence
    Returns:
        saliency: numpy array of shape [seq_length], importance of each position
    """
    # One-hot encode
    encoded_seq = one_hot_encode([sequence], seq_length=seq_length)
    seq_tensor = torch.tensor(encoded_seq, dtype=torch.float32, device=device)
    seq_tensor.requires_grad = True

    # Forward pass
    output, _ = model(seq_tensor)
    pred_class = torch.argmax(output, dim=1)
    score = output[0, pred_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Compute saliency
    saliency = seq_tensor.grad.data.abs()          # [1, seq_length, 4]
    saliency = saliency.max(dim=2)[0].squeeze()    # max over A,C,G,T -> [seq_length]
    return saliency.cpu().numpy()

# ------------------------------
# Extract top k-mers from saliency
# ------------------------------
def extract_top_kmers(sequence, saliency, k=12, top_n=5):
    """
    sequence: string
    saliency: numpy array [seq_length]
    k: length of k-mer
    top_n: how many top positions to extract
    Returns: list of top k-mers
    """
    top_positions = np.argsort(saliency)[-top_n:]  # top positions
    kmers = []
    for pos in top_positions:
        start = max(0, pos - k//2)
        end = min(len(sequence), pos + k//2)
        kmer = sequence[start:end]
        # pad if shorter
        if len(kmer) < k:
            kmer = kmer.ljust(k, "N")
        kmers.append(kmer)
    return kmers

# ------------------------------
# Run on all test sequences
# ------------------------------
all_kmers = []
for seq in X_test:
    sal = compute_saliency(model, seq, seq_length=500)
    kmers = extract_top_kmers(seq, sal, k=12, top_n=3)  # top 3 kmers per seq
    all_kmers.extend(kmers)

print(f"Extracted {len(all_kmers)} k-mers from test sequences.")
print("Example top k-mers:", all_kmers[:10])

import matplotlib.pyplot as plt
import pandas as pd
import os

# Create folder to save plots
os.makedirs("Eb3Saliency_Plots", exist_ok=True)

# Parameters
k = 12
top_n = 3  # top 3 positions per sequence
save_csv_path = "LSDSaliency_TopPositions.csv"

results = []

# Loop over test sequences
for idx, seq in enumerate(X_test):
    saliency = compute_saliency(model, seq, seq_length=500)
    top_positions = np.argsort(saliency)[-top_n:]
    
    # Extract k-mers at top positions
    kmers = []
    for pos in top_positions:
        start = max(0, pos - k//2)
        end = min(len(seq), pos + k//2)
        kmer = seq[start:end].ljust(k, "N")
        kmers.append(kmer)
    
    # Save for CSV
    results.append({
        "sequence_index": idx,
        "sequence": seq,
        "top_positions": top_positions.tolist(),
        "top_kmers": kmers
    })
    
    # Plot saliency
    plt.figure(figsize=(12,3))
    plt.plot(saliency, label="Saliency")
    plt.scatter(top_positions, saliency[top_positions], color="red", label="Top positions")
    plt.title(f"Sequence {idx} - Saliency Map")
    plt.xlabel("Position")
    plt.ylabel("Importance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Eb3Saliency_Plots/Sequence_{idx}_saliency.png")
    plt.close()

# Save all results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(save_csv_path, index=False)

print(f"Saliency plots saved to 'Eb3Saliency_Plots/' folder")
print(f"Top positions and k-mers saved to {save_csv_path}")