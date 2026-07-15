#!/usr/bin/env python3

import argparse

from cnn import train_cnn_attention



if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Train WT/KO CNN attention model"
    )


    parser.add_argument(
        "--wt",
        required=True,
        help="WT.csv"
    )


    parser.add_argument(
        "--ko",
        required=True,
        help="KO.csv"
    )


    parser.add_argument(
        "--outdir",
        required=True,
        help="model output directory"
    )


    parser.add_argument(
        "--epochs",
        default=100,
        type=int
    )


    parser.add_argument(
        "--batch_size",
        default=16,
        type=int
    )


    args = parser.parse_args()



    result = train_cnn_attention(
        wt_file=args.wt,
        ko_file=args.ko,
        output_dir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )



    model_path = result["model_path"]

    best_acc = result["best_acc"]

    accuracy_plot = result["accuracy_plot_path"]

    confusion_matrix = result["confusion_matrix_path"]

    classification_report = result["classification_report_path"]



    print("\nFinished")

    print(
        "Model:",
        model_path
    )

    print(
        "Best Accuracy:",
        best_acc
    )

    print(
        "Accuracy plot:",
        accuracy_plot
    )

    print(
        "Confusion matrix:",
        confusion_matrix
    )

    print(
        "Classification report:",
        classification_report
    )