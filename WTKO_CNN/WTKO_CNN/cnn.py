import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

torch.manual_seed(30)
np.random.seed(30)


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
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}


def one_hot_encode(sequences, seq_length=None):

    encoded_sequences = []

    for seq in sequences:

        seq = seq.upper()

        encoded = np.array(
            [
                nucleotides.get(
                    b,
                    [0, 0, 0, 0]
                )
                for b in seq
            ],
            dtype=np.float32
        )

        if seq_length:

            if len(encoded) < seq_length:

                pad = np.zeros(
                    (
                        seq_length - len(encoded),
                        4
                    ),
                    dtype=np.float32
                )

                encoded = np.vstack(
                    [
                        encoded,
                        pad
                    ]
                )

            else:

                encoded = encoded[:seq_length]

        encoded_sequences.append(encoded)

    return np.array(
        encoded_sequences,
        dtype=np.float32
    )


############################################
# Attention Layer
############################################

class Attention1D(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.attn = nn.Linear(
            input_dim,
            1
        )

    def forward(self, x):

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
# CNN + Attention Model
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

        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(
            64,
            128,
            kernel_size=10
        )

        self.pool2 = nn.MaxPool1d(2)

        self.dropout2 = nn.Dropout(0.3)

        self.attn = Attention1D(128)

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.output = nn.Linear(
            64,
            2
        )

    def forward(self, x):

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

        x = self.dropout1(x)

        x = self.pool2(
            F.relu(
                self.conv2(x)
            )
        )

        x = self.dropout2(x)

        x = x.permute(
            0,
            2,
            1
        )

        x, attn_weights = self.attn(x)

        x = F.relu(
            self.fc1(x)
        )

        x = self.output(x)

        return x, attn_weights


############################################
# Load Data
############################################

def train_cnn_attention(
        wt_file,
        ko_file,
        output_dir,
        test_chromosomes=["chr6", "chr7"],
        epochs=100,
        batch_size=16,
        seed=42,
        class_names=["KO", "WT"]
):

    import os

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    ##################################
    # Random seed
    ##################################

    torch.manual_seed(seed)
    np.random.seed(seed)

    ##################################
    # Load WT/KO sequences
    ##################################

    wt_df = pd.read_csv(
        wt_file
    )

    ko_df = pd.read_csv(
        ko_file
    )

    wt_df = wt_df.rename(
        columns={
            "chr": "chrom"
        }
    )

    ko_df = ko_df.rename(
        columns={
            "chr": "chrom"
        }
    )

    wt_df["label"] = 1
    ko_df["label"] = 0

    ##################################
    # Balance classes
    ##################################

    n = min(
        len(wt_df),
        len(ko_df)
    )

    df = pd.concat(
        [
            wt_df.sample(
                n,
                random_state=seed
            ),

            ko_df.sample(
                n,
                random_state=seed
            )

        ]
    ).reset_index(drop=True)

    df = df.dropna(
        subset=["sequence"]
    )

    ##################################
    # Chromosome split
    ##################################

    train_df = df[
        ~df.chrom.isin(test_chromosomes)
    ]

    test_df = df[
        df.chrom.isin(test_chromosomes)
    ]

    ##################################
    # Encoding
    ##################################

    X_train = one_hot_encode(
        train_df.sequence.values,
        seq_length=600
    )

    X_test = one_hot_encode(
        test_df.sequence.values,
        seq_length=600
    )

    y_train = train_df.label.values
    y_test = test_df.label.values

    train_loader = DataLoader(
        SequenceDataset(
            X_train,
            y_train
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        SequenceDataset(
            X_test,
            y_test
        ),
        batch_size=batch_size,
        shuffle=False
    )

    ##################################
    # Model
    ##################################

    model = SequenceModelWithAttention().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    scaler = GradScaler(
        enabled=torch.cuda.is_available()
    )

    best_acc = 0

    model_path = os.path.join(
        output_dir,
        "best_model.pth"
    )

    train_acc_history = []
    test_acc_history = []

    ##################################
    # Training
    ##################################

    for epoch in range(epochs):

        model.train()

        correct = 0
        total = 0

        optimizer.zero_grad()

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast(
                enabled=torch.cuda.is_available()
            ):

                outputs, _ = model(inputs)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad()

            _, pred = torch.max(
                outputs,
                1
            )

            correct += (
                pred == labels
            ).sum().item()

            total += labels.size(0)

        train_acc = correct / total

        ##################################
        # Validation
        ##################################

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for inputs, labels in test_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs, _ = model(inputs)

                _, pred = torch.max(
                    outputs,
                    1
                )

                correct += (
                    pred == labels
                ).sum().item()

                total += labels.size(0)

        test_acc = correct / total

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Train={train_acc:.3f} "
            f"Test={test_acc:.3f}"
        )

        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
                model_path
            )

    print(
        "Best model saved:",
        model_path
    )

    ##################################
    # Accuracy plot
    ##################################

    best_epoch = np.argmax(test_acc_history) + 1
    best_acc = test_acc_history[best_epoch - 1]

    accuracy_plot_path = os.path.join(
        output_dir,
        "accuracy.png"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, epochs + 1),
        train_acc_history,
        label="Train Accuracy"
    )

    plt.plot(
        range(1, epochs + 1),
        test_acc_history,
        label="Test Accuracy"
    )

    plt.scatter(
        best_epoch,
        best_acc,
        color="red",
        label=f"Best {best_acc*100:.2f}%"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy")
    plt.legend()
    plt.grid()

    plt.savefig(accuracy_plot_path)
    plt.close()

    print("Accuracy plot saved:", accuracy_plot_path)

    ##################################
    # Reload best model for evaluation
    ##################################

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            outputs, _ = model(inputs)

            _, pred = torch.max(
                outputs,
                1
            )

            all_preds.extend(
                pred.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    ##################################
    # Confusion matrix
    ##################################

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    confusion_matrix_path = os.path.join(
        output_dir,
        "confusion_matrix.png"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix - Best Model")

    plt.savefig(confusion_matrix_path)
    plt.close()

    print("Confusion matrix saved:", confusion_matrix_path)

    ##################################
    # Classification report
    # (precision, recall, f1-score)
    ##################################

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    classification_report_path = os.path.join(
        output_dir,
        "classification_report.txt"
    )

    with open(classification_report_path, "w") as f:
        f.write("Classification Report (Best Model):\n\n")
        f.write(report_text)

    print("\nClassification Report (Best Model):\n")
    print(report_text)

    print("Classification report saved:", classification_report_path)

    return {
        "model_path": model_path,
        "best_acc": best_acc,
        "accuracy_plot_path": accuracy_plot_path,
        "confusion_matrix_path": confusion_matrix_path,
        "classification_report_path": classification_report_path,
        "classification_report": report_dict,
        "train_acc_history": train_acc_history,
        "test_acc_history": test_acc_history
    }
