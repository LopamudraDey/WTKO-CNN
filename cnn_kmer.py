import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences

############################################
# DEVICE
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# DATASET
############################################

class SequenceDataset(Dataset):
    
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        return seq

############################################
# ONE HOT ENCODING
############################################

nucleotides = {
    'A':[1,0,0,0],
    'C':[0,1,0,0],
    'G':[0,0,1,0],
    'T':[0,0,0,1],
    'N':[0,0,0,0]
}

def one_hot_encode(sequences, seq_length=500):

    encoded_sequences = []

    for seq in sequences:

        seq = seq.upper()

        encoded = np.array([nucleotides.get(b,[0,0,0,0]) for b in seq])

        encoded = pad_sequences(
            [encoded],
            maxlen=seq_length,
            padding='post',
            dtype='float32'
        )[0]

        encoded_sequences.append(encoded)

    return np.array(encoded_sequences)

############################################
# ATTENTION LAYER
############################################

class Attention1D(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim,1)

    def forward(self,x):

        weights = self.attn(x)

        weights = torch.softmax(weights,dim=1)

        out = torch.sum(x*weights,dim=1)

        return out,weights

############################################
# CNN MODEL
############################################

class SequenceModelWithAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(4,64,kernel_size=10)

        self.pool1 = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64,128,kernel_size=10)

        self.pool2 = nn.MaxPool1d(2)

        self.dropout2 = nn.Dropout(0.3)

        self.attn = Attention1D(128)

        self.fc1 = nn.Linear(128,64)

        self.output = nn.Linear(64,2)

    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.dropout1(x)

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.dropout2(x)

        x = x.permute(0,2,1)

        x,_ = self.attn(x)

        x = F.relu(self.fc1(x))

        x = self.output(x)

        return x

############################################
# LOAD DATA
############################################

wt_df = pd.read_csv('LSD_WT.csv', delimiter=";")
ko_df = pd.read_csv('LSD_WT.csv', delimiter=";")

df = pd.concat([wt_df, ko_df]).reset_index(drop=True)

df = df.dropna(subset=['sequence'])

sequences = df["sequence"].values

encoded = one_hot_encode(sequences,500)

dataset = SequenceDataset(encoded)

loader = DataLoader(dataset,batch_size=32,shuffle=False)

############################################
# LOAD MODEL
############################################

model = SequenceModelWithAttention().to(device)

model.load_state_dict(torch.load("LSD1a600final_model.pth",map_location=device))

model.eval()

############################################
# GET CONV1 ACTIVATIONS
############################################

all_activations = []

all_sequences = []

with torch.no_grad():

    for seq_batch in loader:

        seq_batch = seq_batch.to(device)

        x = seq_batch.permute(0,2,1)

        conv_out = F.relu(model.conv1(x))

        all_activations.append(conv_out.cpu().numpy())

        all_sequences.append(seq_batch.cpu().numpy())

activations = np.concatenate(all_activations)

sequences = np.concatenate(all_sequences)

############################################
# ONE HOT TO DNA
############################################

def onehot_to_seq(onehot):

    nucs = ['A','C','G','T']

    seq = ""

    for pos in onehot:

        seq += nucs[np.argmax(pos)]

    return seq

############################################
# EXTRACT FILTER MOTIFS
############################################

KERNEL_SIZE = 10
TOP_K = 60

results = []

num_filters = activations.shape[1]

for f in range(num_filters):

    filter_act = activations[:,f,:]

    flat_scores = filter_act.flatten()

    top_idx = np.argsort(flat_scores)[-TOP_K:]

    kmers = []
    scores = []

    for idx in top_idx:

        seq_idx = idx // filter_act.shape[1]
        pos = idx % filter_act.shape[1]

        seq = onehot_to_seq(sequences[seq_idx])

        kmer = seq[pos:pos+KERNEL_SIZE]

        if len(kmer)==KERNEL_SIZE:

            kmers.append(kmer)

            scores.append(filter_act[seq_idx,pos])

    if len(kmers)==0:
        continue

    matrix = np.zeros((KERNEL_SIZE,4))

    nuc_map = {'A':0,'C':1,'G':2,'T':3}

    for kmer in kmers:

        for i,b in enumerate(kmer):

            matrix[i,nuc_map[b]] += 1

    nucs = ['A','C','G','T']

    consensus = ""

    for i in range(KERNEL_SIZE):

        consensus += nucs[np.argmax(matrix[i])]

    results.append({
        "filter":f,
        "consensus_kmer":consensus,
        "mean_activation":np.mean(scores)
    })

############################################
# SAVE RESULTS
############################################

df_results = pd.DataFrame(results)

df_results.to_csv("cnn_filter_Kmers_LSD.csv",index=False)

print(df_results.head())

print("\nSaved motifs to cnn_filter_kmers_Rela.csv")