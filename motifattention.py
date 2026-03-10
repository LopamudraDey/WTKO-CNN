import pandas as pd

# Read your existing results
df = pd.read_csv("LSDSaliency_TopPositions.csv")

# Flatten top_kmers and top_positions so each sequence has multiple rows per k-mer
rows = []
for idx, row in df.iterrows():
    seq_idx = row['sequence_index']
    seq = row['sequence']
    positions = eval(row['top_positions'])   # convert string list to Python list
    kmers = eval(row['top_kmers'])
    for pos, kmer in zip(positions, kmers):
        rows.append({
            "sequence_index": seq_idx,
            "sequence": seq,
            "top_position": pos,
            "top_kmer": kmer
        })

df_flat = pd.DataFrame(rows)
df_flat.to_csv("LSDSaliency_TopPositions_Flat.csv", index=False)
print("Saved flattened saliency CSV as 'Saliency_TopPositions_Flat.csv'")

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import matplotlib.pyplot as plt
import logomaker
import os

# ------------------------------
# Parameters
# ------------------------------
saliency_csv = "LSDSaliency_TopPositions_Flat.csv"  # your flattened CSV
n_clusters = 10       # number of motif clusters
kmer_length = 20      # k-mer length

# Folder to save logos
os.makedirs("Motif_Logos", exist_ok=True)

# ------------------------------
# 1. Load flattened k-mers CSV
# ------------------------------
df_kmers = pd.read_csv(saliency_csv)
print(f"Loaded {len(df_kmers)} k-mers for motif analysis")

# ------------------------------
# 2. Encode k-mers for clustering
# ------------------------------
nucleotides = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}

def encode_kmer(kmer):
    encoded = np.zeros((len(kmer), 5))
    for i, c in enumerate(kmer.upper()):
        idx = nucleotides.get(c, 4)
        encoded[i, idx] = 1
    return encoded.flatten()

X_encoded = np.array([encode_kmer(k) for k in df_kmers['top_kmer']])
print("Encoded k-mers shape:", X_encoded.shape)

# ------------------------------
# 3. Cluster k-mers
# ------------------------------
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='cosine',      # use metric instead of affinity
    linkage='average'
)
labels = clustering.fit_predict(X_encoded)
df_kmers['cluster'] = labels
print("Cluster counts:\n", df_kmers.groupby('cluster').size())

# ------------------------------
# 4. Generate consensus motifs per cluster
# ------------------------------
def consensus_motif(kmers):
    if not kmers: return ""
    k = len(kmers[0])
    motif = ""
    for i in range(k):
        counts = Counter([kmer[i] for kmer in kmers])
        motif += counts.most_common(1)[0][0]
    return motif

consensus_list = []
for cluster_id in sorted(df_kmers['cluster'].unique()):
    kmers_in_cluster = df_kmers[df_kmers['cluster']==cluster_id]['top_kmer'].tolist()
    motif = consensus_motif(kmers_in_cluster)
    consensus_list.append((cluster_id, motif, len(kmers_in_cluster)))

df_motifs = pd.DataFrame(consensus_list, columns=['cluster','consensus','count'])
df_motifs.to_csv("LSDMotif_Consensus.csv", index=False)
print("Consensus motifs saved to 'Motif_Consensus.csv'")
print(df_motifs)
# Save all kmers cluster-wise to FASTA for MEME
with open("Clustered_Kmers.fasta", "w") as f:
    for cluster_id in df_kmers['cluster'].unique():
        kmers_in_cluster = df_kmers[df_kmers['cluster']==cluster_id]['top_kmer'].tolist()
        for i, kmer in enumerate(kmers_in_cluster):
            f.write(f">cluster{cluster_id}_{i+1}\n{kmer}\n")
print("All cluster kmers saved to 'Clustered_Kmers.fasta' for MEME input")# ------------------------------
# 5. Visualize sequence logos per cluster
# ------------------------------
for cluster_id in df_kmers['cluster'].unique():
    kmers_in_cluster = df_kmers[df_kmers['cluster']==cluster_id]['top_kmer'].tolist()
    if len(kmers_in_cluster) == 0: continue

    # Create PWM
    pwm = pd.DataFrame(0, index=list("ACGT"), columns=range(len(kmers_in_cluster[0])))
    for kmer in kmers_in_cluster:
        for i, nt in enumerate(kmer):
            if nt in "ACGT":
                pwm.loc[nt,i] += 1
    pwm = pwm / pwm.sum(axis=0)

    plt.figure(figsize=(10,2))
    logo = logomaker.Logo(pwm.T)
    logo.ax.set_title(f"Cluster {cluster_id} - {len(kmers_in_cluster)} k-mers")
    plt.tight_layout()
    plt.savefig(f"LSDMotif_Logos/Cluster_{cluster_id}_logo.png")
    plt.close()

print("Sequence logos saved in 'Motif_Logos/' folder")