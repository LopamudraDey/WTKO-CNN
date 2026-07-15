#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


############################################
# Device
############################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


############################################
# Dataset
############################################

class SequenceDataset(Dataset):

    def __init__(
        self,
        sequences,
        labels
    ):
        self.sequences = sequences
        self.labels = labels


    def __len__(self):

        return len(self.sequences)


    def __getitem__(
        self,
        idx
    ):

        seq = torch.tensor(
            self.sequences[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return seq, label



############################################
# One-hot encoding
############################################

nucleotides = {

    'A':[1,0,0,0],
    'C':[0,1,0,0],
    'G':[0,0,1,0],
    'T':[0,0,0,1],
    'N':[0,0,0,0]

}



def one_hot_encode(
    sequences,
    seq_length=None
):

    encoded_sequences = []


    for seq in sequences:

        seq = seq.upper()


        encoded = np.array(
            [
                nucleotides.get(
                    b,
                    [0,0,0,0]
                )
                for b in seq
            ],
            dtype=np.float32
        )


        if seq_length:

            encoded = pad_sequences(
                [
                    encoded
                ],
                maxlen=seq_length,
                padding="post",
                dtype="float32"
            )[0]


        encoded_sequences.append(
            encoded
        )


    return np.array(
        encoded_sequences
    )



############################################
# Attention
############################################

class Attention1D(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.attn = nn.Linear(
            input_dim,
            1
        )


    def forward(
        self,
        x
    ):

        weights = self.attn(x)

        weights = torch.softmax(
            weights,
            dim=1
        )


        out = torch.sum(
            x * weights,
            dim=1
        )


        return out, weights



############################################
# CNN Attention Model
############################################

class SequenceModelWithAttention(nn.Module):


    def __init__(self):

        super().__init__()


        self.conv1 = nn.Conv1d(
            4,
            64,
            kernel_size=10
        )


        self.pool1 = nn.MaxPool1d(2)


        self.dropout1 = nn.Dropout(
            0.3
        )


        self.conv2 = nn.Conv1d(
            64,
            128,
            kernel_size=10
        )


        self.pool2 = nn.MaxPool1d(2)


        self.dropout2 = nn.Dropout(
            0.3
        )


        self.attn = Attention1D(
            128
        )


        self.fc1 = nn.Linear(
            128,
            64
        )


        self.output = nn.Linear(
            64,
            2
        )



    def forward(
        self,
        x
    ):


        x = x.permute(
            0,
            2,
            1
        )


        x = self.pool1(
            F.relu(
                self.conv1(x)
            )
        )


        x = self.dropout1(
            x
        )


        x = self.pool2(
            F.relu(
                self.conv2(x)
            )
        )


        x = self.dropout2(
            x
        )


        x = x.permute(
            0,
            2,
            1
        )


        x, attn_weights = self.attn(
            x
        )


        x = F.relu(
            self.fc1(x)
        )


        x = self.output(
            x
        )


        return x, attn_weights



############################################
# Saliency calculation
############################################

def compute_saliency(
    model,
    sequence,
    seq_length=600
):


    encoded_seq = one_hot_encode(
        [
            sequence
        ],
        seq_length=seq_length
    )


    seq_tensor = torch.tensor(
        encoded_seq,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )


    output, _ = model(
        seq_tensor
    )


    pred_class = torch.argmax(
        output,
        dim=1
    )


    score = output[
        0,
        pred_class
    ]


    model.zero_grad()


    score.backward()


    saliency = seq_tensor.grad.abs()


    saliency = saliency.max(
        dim=2
    )[0].squeeze()


    return saliency.cpu().numpy()
############################################
# Extract top k-mers
############################################

def extract_top_kmers(
    sequence,
    saliency,
    k=20,
    top_n=3
):

    top_positions = np.argsort(
        saliency
    )[-top_n:]


    kmers = []


    for pos in top_positions:

        start = max(
            0,
            pos - k//2
        )

        end = min(
            len(sequence),
            pos + k//2
        )


        kmer = sequence[
            start:end
        ]


        if len(kmer) < k:

            kmer = kmer.ljust(
                k,
                "N"
            )


        kmers.append(
            kmer
        )


    return top_positions, kmers



############################################
# Main saliency analysis function
############################################

def compute_saliency_analysis(
    wt_file,
    ko_file,
    model_path,
    output_dir,
    test_chromosomes=["chr6", "chr7"],
    seq_length=600,
    k=20,
    top_n=3
):


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    plots_dir = os.path.join(
        output_dir,
        "saliency_plots"
    )


    os.makedirs(
        plots_dir,
        exist_ok=True
    )


    ########################################
    # Load WT/KO data
    ########################################

    wt_df = pd.read_csv(
        wt_file
    )


    ko_df = pd.read_csv(
        ko_file
    )


    wt_df = wt_df.rename(
        columns={
            "chr":"chrom"
        }
    )


    ko_df = ko_df.rename(
        columns={
            "chr":"chrom"
        }
    )


    wt_df["label"] = 1

    ko_df["label"] = 0



    ########################################
    # Balance classes
    ########################################

    n = min(
        len(wt_df),
        len(ko_df)
    )


    df = pd.concat(
        [
            wt_df.sample(
                n,
                random_state=42
            ),

            ko_df.sample(
                n,
                random_state=42
            )

        ]
    ).reset_index(
        drop=True
    )


    df = df.dropna(
        subset=[
            "sequence"
        ]
    )



    ########################################
    # Chromosome test split
    ########################################

    test_df = df[
        df.chrom.isin(
            test_chromosomes
        )
    ]


    X_test = test_df[
        "sequence"
    ].values



    ########################################
    # Encode test sequences
    ########################################

    test_encoded = one_hot_encode(
        X_test,
        seq_length=seq_length
    )


    test_dataset = SequenceDataset(
        test_encoded,
        test_df.label.values
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )



    ########################################
    # Load trained CNN model
    ########################################

    model = SequenceModelWithAttention().to(
        device
    )


    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )


    model.eval()



    ########################################
    # Compute saliency
    ########################################

    results = []

    all_kmers = []



    for idx, seq in enumerate(X_test):


        saliency = compute_saliency(
            model,
            seq,
            seq_length
        )


        top_positions, kmers = extract_top_kmers(
            seq,
            saliency,
            k=k,
            top_n=top_n
        )


        all_kmers.extend(
            kmers
        )


        results.append(
            {
                "sequence_index": idx,
                "sequence": seq,
                "top_positions": top_positions.tolist(),
                "top_kmers": kmers
            }
        )



        ####################################
        # Plot saliency
        ####################################

        plt.figure(
            figsize=(12,3)
        )


        plt.plot(
            saliency,
            label="Saliency"
        )


        plt.scatter(
            top_positions,
            saliency[top_positions],
            color="red",
            label="Top positions"
        )


        plt.title(
            f"Sequence {idx} Saliency"
        )


        plt.xlabel(
            "Position"
        )


        plt.ylabel(
            "Importance"
        )


        plt.legend()

        plt.grid()


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                plots_dir,
                f"Sequence_{idx}_saliency.png"
            )
        )


        plt.close()



    ########################################
    # Save CSV
    ########################################

    csv_path = os.path.join(
        output_dir,
        "top_saliency_positions.csv"
    )


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        csv_path,
        index=False
    )



    print(
        f"Extracted {len(all_kmers)} k-mers"
    )


    print(
        "Example:",
        all_kmers[:10]
    )


    print(
        "Saliency plots:",
        plots_dir
    )


    print(
        "CSV:",
        csv_path
    )



    return {

        "plots_dir": plots_dir,

        "csv_path": csv_path,

        "num_sequences": len(X_test),

        "num_kmers": len(all_kmers)

    }