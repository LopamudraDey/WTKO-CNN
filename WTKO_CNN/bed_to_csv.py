import pandas as pd
from pyfaidx import Fasta
import argparse


def center_and_expand(df, width=600):
    """
    Convert peaks to centered fixed length regions.
    """

    df = df.copy()

    mid = (df["start"] + df["end"]) // 2

    df["start"] = mid - width // 2
    df["end"] = mid + width // 2

    return df



def bed_to_csv(
        bed_file,
        genome_file,
        output_file,
        label,
        width=600
):

    print(f"Processing {bed_file}")


    # Read BED
    bed = pd.read_csv(
        bed_file,
        sep="\t",
        header=None
    )


    # Keep first three BED columns
    bed = bed.iloc[:,0:3]

    bed.columns = [
        "chr",
        "start",
        "end"
    ]


    # Generate 600 bp regions
    bed = center_and_expand(
        bed,
        width
    )


    # Load genome
    genome = Fasta(genome_file)


    sequences = []


    for _, row in bed.iterrows():

        try:

            seq = genome[
                row["chr"]
            ][
                row["start"]:row["end"]
            ].seq.upper()


        except KeyError:

            seq = "N" * width


        sequences.append(seq)


    # Add sequence column
    bed["sequence"] = sequences


    # Add label
    bed["label"] = label


    # Add ID
    bed.insert(
        0,
        "id",
        [
            f"{label}_{i+1}"
            for i in range(len(bed))
        ]
    )


    # Save CSV
    bed.to_csv(
        output_file,
        index=False
    )


    print(
        f"Saved {output_file}"
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--bed",
        required=True
    )

    parser.add_argument(
        "--genome",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    parser.add_argument(
        "--label",
        required=True
    )

    parser.add_argument(
        "--width",
        default=600,
        type=int
    )


    args = parser.parse_args()


    bed_to_csv(
        args.bed,
        args.genome,
        args.out,
        args.label,
        args.width
    )