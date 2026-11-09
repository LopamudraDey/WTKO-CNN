
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)


############################################
# Reproducibility and device
############################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


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
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0]
}


def one_hot_encode(sequences, seq_length=600):

    encoded_sequences = []

    for seq in sequences:

        seq = str(seq).upper()

        encoded = np.array(
            [
                nucleotides.get(
                    base,
                    [0, 0, 0, 0]
                )
                for base in seq
            ],
            dtype=np.float32
        )

        # Pad shorter sequences with zeros
        if len(encoded) < seq_length:

            padding = np.zeros(
                (seq_length - len(encoded), 4),
                dtype=np.float32
            )

            encoded = np.vstack(
                [encoded, padding]
            )

        # Truncate longer sequences
        else:

            encoded = encoded[:seq_length]

        encoded_sequences.append(encoded)

    return np.array(
        encoded_sequences,
        dtype=np.float32
    )


############################################
# Attention layer
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
# CNN + Attention model
############################################

class SequenceModelWithAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=10
        )

        self.pool1 = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
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

        # [batch, sequence length, channels]
        # to [batch, channels, sequence length]

        x = x.permute(0, 2, 1)

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

        # [batch, channels, sequence length]
        # to [batch, sequence length, channels]

        x = x.permute(0, 2, 1)

        x, attn_weights = self.attn(x)

        x = F.relu(
            self.fc1(x)
        )

        x = self.output(x)

        return x, attn_weights


############################################
# Main training function
############################################

def train_cnn_attention(

    wt_file,
    ko_file,
    output_dir,

    validation_chromosome="chr5",
    test_chromosomes=("chr6", "chr7"),

    epochs=100,
    batch_size=16,
    learning_rate=0.001,

    sequence_length=600,
    seed=42,

    class_names=("KO", "WT")
):

    ########################################
    # Create output directory
    ########################################

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    set_seed(seed)

    ########################################
    # Load WT and KO data
    ########################################

    wt_df = pd.read_csv(wt_file)
    ko_df = pd.read_csv(ko_file)

    # Standardize chromosome column name

    wt_df = wt_df.rename(
        columns={"chr": "chrom"}
    )

    ko_df = ko_df.rename(
        columns={"chr": "chrom"}
    )

    # Check required columns

    required_columns = ["chrom", "sequence"]

    for column in required_columns:

        if column not in wt_df.columns:
            raise ValueError(
                f"WT file is missing column: {column}"
            )

        if column not in ko_df.columns:
            raise ValueError(
                f"KO file is missing column: {column}"
            )

    # Assign labels
    # KO = 0
    # WT = 1

    wt_df["label"] = 1
    ko_df["label"] = 0

    # Remove missing sequences and chromosomes

    wt_df = wt_df.dropna(
        subset=["sequence", "chrom"]
    ).copy()

    ko_df = ko_df.dropna(
        subset=["sequence", "chrom"]
    ).copy()

    ########################################
    # Balance WT and KO classes
    ########################################

    n = min(
        len(wt_df),
        len(ko_df)
    )

    if n == 0:
        raise ValueError(
            "No WT or KO sequences available."
        )

    wt_df = wt_df.sample(
        n=n,
        random_state=seed
    )

    ko_df = ko_df.sample(
        n=n,
        random_state=seed
    )

    df = pd.concat(
        [wt_df, ko_df],
        ignore_index=True
    )

    ########################################
    # Chromosome-level data partitioning
    ########################################

    validation_chromosomes = [
        validation_chromosome
    ]

    test_chromosomes = list(
        test_chromosomes
    )

    # Ensure validation and test chromosomes
    # are not overlapping

    if set(validation_chromosomes).intersection(
        set(test_chromosomes)
    ):
        raise ValueError(
            "Validation and test chromosomes overlap."
        )

    # Training set:
    # All chromosomes except validation and test

    excluded_chromosomes = (
        validation_chromosomes + test_chromosomes
    )

    train_df = df[
        ~df["chrom"].isin(excluded_chromosomes)
    ].copy()

    # Validation set

    val_df = df[
        df["chrom"].isin(validation_chromosomes)
    ].copy()

    # Independent test set

    test_df = df[
        df["chrom"].isin(test_chromosomes)
    ].copy()

    ########################################
    # Check data partitions
    ########################################

    if len(train_df) == 0:
        raise ValueError("Training set is empty.")

    if len(val_df) == 0:
        raise ValueError("Validation set is empty.")

    if len(test_df) == 0:
        raise ValueError("Test set is empty.")

    # Check that both classes are represented
    # in each partition

    for name, subset in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:

        class_counts = subset["label"].value_counts()

        print(
            f"\n{name} set:"
        )

        print(
            "Number of sequences:",
            len(subset)
        )

        print(
            "KO sequences:",
            int(class_counts.get(0, 0))
        )

        print(
            "WT sequences:",
            int(class_counts.get(1, 0))
        )

        if set(subset["label"].unique()) != {0, 1}:
            raise ValueError(
                f"{name} set does not contain both WT and KO classes. "
                "Choose a different chromosome partition."
            )

    ########################################
    # Extract sequences and labels
    ########################################

    X_train = train_df["sequence"].values
    y_train = train_df["label"].values

    X_val = val_df["sequence"].values
    y_val = val_df["label"].values

    X_test = test_df["sequence"].values
    y_test = test_df["label"].values

    ########################################
    # One-hot encoding
    ########################################

    X_train = one_hot_encode(
        X_train,
        seq_length=sequence_length
    )

    X_val = one_hot_encode(
        X_val,
        seq_length=sequence_length
    )

    X_test = one_hot_encode(
        X_test,
        seq_length=sequence_length
    )

    ########################################
    # Create datasets
    ########################################

    train_dataset = SequenceDataset(
        X_train,
        y_train
    )

    val_dataset = SequenceDataset(
        X_val,
        y_val
    )

    test_dataset = SequenceDataset(
        X_test,
        y_test
    )

    ########################################
    # Create data loaders
    ########################################

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    ########################################
    # Initialize model
    ########################################

    model = SequenceModelWithAttention().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scaler = GradScaler(
        enabled=torch.cuda.is_available()
    )

    ########################################
    # Model saving
    ########################################

    model_path = os.path.join(
        output_dir,
        "best_model.pth"
    )

    best_val_acc = -1.0
    best_epoch = 0

    ########################################
    # Accuracy histories
    ########################################

    train_acc_history = []
    val_acc_history = []

    train_loss_history = []
    val_loss_history = []

    ########################################
    # Training loop
    ########################################

    for epoch in range(epochs):

        ####################################
        # Training
        ####################################

        model.train()

        correct_train = 0
        total_train = 0
        running_train_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(
                enabled=torch.cuda.is_available()
            ):

                outputs, _ = model(inputs)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_train_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            correct_train += (
                predicted == labels
            ).sum().item()

            total_train += labels.size(0)

        train_acc = (
            correct_train / total_train
        )

        train_loss = (
            running_train_loss / len(train_loader)
        )

        ####################################
        # Validation
        ####################################

        model.eval()

        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():

            for inputs, labels in val_loader:

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

                running_val_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    dim=1
                )

                correct_val += (
                    predicted == labels
                ).sum().item()

                total_val += labels.size(0)

        val_acc = (
            correct_val / total_val
        )

        val_loss = (
            running_val_loss / len(val_loader)
        )

        ####################################
        # Store accuracy and loss
        ####################################

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        ####################################
        # Print epoch results
        ####################################

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Acc: {val_acc:.4f}"
        )

        ####################################
        # Save best model based ONLY
        # on validation accuracy
        ####################################

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                model_path
            )

    ########################################
    # End of training
    ########################################

    print("\nTraining completed.")

    print(
        "Best validation accuracy:",
        round(best_val_acc, 4)
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best model saved:",
        model_path
    )

    ########################################
    # Plot training and validation accuracy
    ########################################

    accuracy_plot_path = os.path.join(
        output_dir,
        "accuracy.png"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, epochs + 1),
        train_acc_history,
        label="Training Accuracy"
    )

    plt.plot(
        range(1, epochs + 1),
        val_acc_history,
        label="Validation Accuracy"
    )

    plt.scatter(
        best_epoch,
        best_val_acc,
        color="red",
        label=f"Best Validation: {best_val_acc * 100:.2f}%"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig(
        accuracy_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ########################################
    # Plot training and validation loss
    ########################################

    loss_plot_path = os.path.join(
        output_dir,
        "loss.png"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, epochs + 1),
        train_loss_history,
        label="Training Loss"
    )

    plt.plot(
        range(1, epochs + 1),
        val_loss_history,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.savefig(
        loss_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ########################################
    # Load best model selected using
    # validation accuracy
    ########################################

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    ########################################
    # Independent test evaluation
    #
    # The test set is evaluated only here,
    # after model selection is complete.
    ########################################

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            outputs, _ = model(inputs)

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    ########################################
    # Final test accuracy
    ########################################

    test_acc = accuracy_score(
        all_labels,
        all_preds
    )

    print("\nIndependent test results")

    print(
        "Test accuracy:",
        round(test_acc, 4)
    )

    ########################################
    # Confusion matrix
    ########################################

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[0, 1]
    )

    confusion_matrix_path = os.path.join(
        output_dir,
        "confusion_matrix.png"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        cmap=plt.cm.Blues
    )

    plt.title(
        "Confusion Matrix - Independent Test Set"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    ########################################
    # Classification report
    ########################################

    report_text = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    classification_report_path = os.path.join(
        output_dir,
        "classification_report.txt"
    )

    with open(
        classification_report_path,
        "w"
    ) as f:

        f.write(
            "Classification Report - Independent Test Set\n\n"
        )

        f.write(report_text)

        f.write(
            f"\nIndependent test accuracy: {test_acc:.4f}\n"
        )

        f.write(
            f"Best validation accuracy: {best_val_acc:.4f}\n"
        )

        f.write(
            f"Best epoch: {best_epoch}\n"
        )

    print(
        "\nClassification Report - Independent Test Set:\n"
    )

    print(report_text)

    ########################################
    # Save training history
    ########################################

    history_df = pd.DataFrame({

        "epoch": range(1, epochs + 1),

        "training_accuracy": train_acc_history,

        "validation_accuracy": val_acc_history,

        "training_loss": train_loss_history,

        "validation_loss": val_loss_history

    })

    history_path = os.path.join(
        output_dir,
        "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    ########################################
    # Return results
    ########################################

    return {

        "model_path": model_path,

        "best_epoch": best_epoch,

        "best_validation_accuracy": best_val_acc,

        "independent_test_accuracy": test_acc,

        "accuracy_plot_path": accuracy_plot_path,

        "loss_plot_path": loss_plot_path,

        "confusion_matrix_path": confusion_matrix_path,

        "classification_report_path": classification_report_path,

        "classification_report": report_dict,

        "train_acc_history": train_acc_history,

        "val_acc_history": val_acc_history,

        "train_loss_history": train_loss_history,

        "val_loss_history": val_loss_history,

        "history_path": history_path
    }


############################################
# Run the model
############################################

if __name__ == "__main__":

    results = train_cnn_attention(

        wt_file="RelaWT600.csv",

        ko_file="RelaKO600.csv",

        output_dir="WTKO_CNN_RelA_results",

        validation_chromosome="chr5",

        test_chromosomes=("chr6", "chr7"),

        epochs=100,

        batch_size=16,

        learning_rate=0.001,

        sequence_length=600,

        seed=42
    )

    print("\nFinal results:")

    print(
        "Best validation accuracy:",
        results["best_validation_accuracy"]
    )

    print(
        "Independent test accuracy:",
        results["independent_test_accuracy"]
    )
