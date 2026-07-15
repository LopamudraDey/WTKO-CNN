#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np

from collections import Counter

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

import logomaker



############################################
# Flatten saliency output
############################################

def flatten_kmers(
        input_csv,
        output_csv
):

    df = pd.read_csv(
        input_csv
    )


    rows = []


    for _, row in df.iterrows():

        positions = eval(
            row["top_positions"]
        )

        kmers = eval(
            row["top_kmers"]
        )


        for pos,kmer in zip(
            positions,
            kmers
        ):

            rows.append(
                {
                    "sequence_index":
                        row["sequence_index"],

                    "sequence":
                        row["sequence"],

                    "top_position":
                        pos,

                    "top_kmer":
                        kmer
                }
            )


    flat = pd.DataFrame(
        rows
    )


    flat.to_csv(
        output_csv,
        index=False
    )


    return flat



############################################
# Encode kmer
############################################

def encode_kmer(kmer):

    nucleotides = {
        "A":0,
        "C":1,
        "G":2,
        "T":3,
        "N":4
    }


    encoded = np.zeros(
        (
            len(kmer),
            5
        )
    )


    for i,c in enumerate(
        kmer.upper()
    ):

        encoded[
            i,
            nucleotides.get(c,4)
        ] = 1


    return encoded.flatten()



############################################
# Consensus motif
############################################

def consensus_motif(kmers):

    motif=""


    for i in range(
        len(kmers[0])
    ):

        counts = Counter(
            [
                k[i]
                for k in kmers
            ]
        )

        motif += counts.most_common(
            1
        )[0][0]


    return motif



############################################
# Main motif analysis
############################################

def motif_analysis(
        saliency_csv,
        output_dir,
        n_clusters=10
):


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    flat_csv=os.path.join(
        output_dir,
        "flattened_kmers.csv"
    )


    df_kmers = flatten_kmers(
        saliency_csv,
        flat_csv
    )


    ########################################
    # Encode
    ########################################

    X=np.array(
        [
            encode_kmer(k)
            for k in df_kmers.top_kmer
        ]
    )


    print(
        "Encoded:",
        X.shape
    )


    ########################################
    # Cluster
    ########################################

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average"
    )


    labels = clustering.fit_predict(
        X
    )


    df_kmers["cluster"]=labels



    df_kmers.to_csv(
        os.path.join(
            output_dir,
            "clustered_kmers.csv"
        ),
        index=False
    )


    ########################################
    # Consensus motifs
    ########################################

    motifs=[]


    for c in sorted(
        df_kmers.cluster.unique()
    ):


        kmers=df_kmers[
            df_kmers.cluster==c
        ].top_kmer.tolist()


        motifs.append(
            [
                c,
                consensus_motif(kmers),
                len(kmers)
            ]
        )



    motif_df=pd.DataFrame(
        motifs,
        columns=[
            "cluster",
            "consensus",
            "count"
        ]
    )


    motif_df.to_csv(
        os.path.join(
            output_dir,
            "consensus_motifs.csv"
        ),
        index=False
    )


    ########################################
    # FASTA clusters
    ########################################

    fasta_dir=os.path.join(
        output_dir,
        "cluster_FASTA"
    )


    os.makedirs(
        fasta_dir,
        exist_ok=True
    )


    for c in sorted(
        df_kmers.cluster.unique()
    ):


        fasta=os.path.join(
            fasta_dir,
            f"cluster_{c}.fasta"
        )


        kmers=df_kmers[
            df_kmers.cluster==c
        ].top_kmer.tolist()


        with open(
            fasta,
            "w"
        ) as f:


            for i,kmer in enumerate(kmers):

                f.write(
                    f">cluster{c}_{i}\n{kmer}\n"
                )



    ########################################
    # Sequence logos
    ########################################

    logo_dir=os.path.join(
        output_dir,
        "logos"
    )


    os.makedirs(
        logo_dir,
        exist_ok=True
    )


    for c in sorted(
        df_kmers.cluster.unique()
    ):


        kmers=df_kmers[
            df_kmers.cluster==c
        ].top_kmer.tolist()


        pwm=pd.DataFrame(
            0,
            index=list("ACGT"),
            columns=range(
                len(kmers[0])
            )
        )


        for kmer in kmers:

            for i,nt in enumerate(kmer):

                if nt in "ACGT":

                    pwm.loc[
                        nt,
                        i
                    ] += 1



        pwm=pwm.div(
            pwm.sum(axis=0),
            axis=1
        )


        plt.figure(
            figsize=(10,2)
        )


        logomaker.Logo(
            pwm.T
        )


        plt.title(
            f"Cluster {c}"
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                logo_dir,
                f"cluster_{c}_logo.png"
            )
        )


        plt.close()



    print(
        "Motif analysis completed"
    )


    return motif_df



############################################
# Command line
############################################

if __name__=="__main__":


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--saliency",
        required=True
    )


    parser.add_argument(
        "--outdir",
        required=True
    )


    parser.add_argument(
        "--clusters",
        type=int,
        default=10
    )


    args=parser.parse_args()



    motif_analysis(
        args.saliency,
        args.outdir,
        args.clusters
    )