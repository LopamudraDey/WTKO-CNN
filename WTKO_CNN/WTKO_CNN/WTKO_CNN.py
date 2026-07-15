#!/usr/bin/env python3

import os
import subprocess
import argparse


def run_command(cmd):

    print("\nRunning:")
    print(cmd)

    subprocess.run(
        cmd,
        shell=True,
        check=True
    )


def main(args):

    # -----------------------------
    # Step 1: WT/KO specific peaks
    # -----------------------------

    print(
        "Step 1: Identifying WT-specific and KO-specific peaks"
    )

    run_command(
        f"bedtools intersect -v "
        f"-a {args.wt_bed} "
        f"-b {args.ko_bed} "
        f"> {args.outdir}/wtspecific.bed"
    )

    run_command(
        f"bedtools intersect -v "
        f"-a {args.ko_bed} "
        f"-b {args.wt_bed} "
        f"> {args.outdir}/kospecific.bed"
    )

    # -----------------------------
    # Step 2: Generate sequence CSVs
    # -----------------------------

    print(
        "Step 2: Generating WT.csv and KO.csv"
    )

    run_command(
        f"python bed_to_csv.py "
        f"--bed {args.outdir}/wtspecific.bed "
        f"--genome {args.genome} "
        f"--out {args.outdir}/WT.csv "
        f"--label WT "
        f"--width 600"
    )

    run_command(
        f"python bed_to_csv.py "
        f"--bed {args.outdir}/kospecific.bed "
        f"--genome {args.genome} "
        f"--out {args.outdir}/KO.csv "
        f"--label KO "
        f"--width 600"
    )

    # -----------------------------
    # Step 3: CNN training
    # -----------------------------

    print(
        "Step 3: Training CNN attention model"
    )

    run_command(
        f"python run_cnn.py "
        f"--wt {args.outdir}/WT.csv "
        f"--ko {args.outdir}/KO.csv "
        f"--outdir {args.outdir}/model"
    )

    # -----------------------------
    # Step 4: Saliency analysis
    # -----------------------------

    print(
        "Step 4: Running saliency analysis"
    )

    run_command(
        f"python run_saliency.py "
        f"--wt {args.outdir}/WT.csv "
        f"--ko {args.outdir}/KO.csv "
        f"--model {args.outdir}/model/best_model.pth "
        f"--outdir {args.outdir}/saliency"
    )

    print(
        "\nWTKO-CNN pipeline completed successfully!"
    )

    print(
    "Step 5: Motif clustering and sequence logos"
    )


    run_command(
      f"python run_motif.py "
      f"--saliency {args.outdir}/saliency/top_saliency_positions.csv "
      f"--outdir {args.outdir}/motifs "
    )
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wt_bed",
        required=True
    )

    parser.add_argument(
        "--ko_bed",
        required=True
    )

    parser.add_argument(
        "--genome",
        required=True
    )

    parser.add_argument(
        "--outdir",
        default="Results"
    )

    args = parser.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    main(args)