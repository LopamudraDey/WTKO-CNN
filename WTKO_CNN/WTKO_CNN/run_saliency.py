#!/usr/bin/env python3

import argparse

from test_attention import compute_saliency_analysis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CNN attention saliency analysis"
    )


    parser.add_argument(
        "--wt",
        required=True
    )


    parser.add_argument(
        "--ko",
        required=True
    )


    parser.add_argument(
        "--model",
        required=True
    )


    parser.add_argument(
        "--outdir",
        required=True
    )


    args = parser.parse_args()



    result = compute_saliency_analysis(
        wt_file=args.wt,
        ko_file=args.ko,
        model_path=args.model,
        output_dir=args.outdir
    )



    print("\nSaliency completed")

    print(
        "Plots:",
        result["plots_dir"]
    )

    print(
        "CSV:",
        result["csv_path"]
    )

    print(
        "Sequences:",
        result["num_sequences"]
    )

    print(
        "K-mers:",
        result["num_kmers"]
    )