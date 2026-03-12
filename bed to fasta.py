import pandas as pd

# Load your newly created BED files
ko_bed = pd.read_csv('D:/GSE149836/RelaKO_specific_peaks.bed', sep='\t', names=['chrom', 'start', 'stop','s','sr','st'])
wt_bed = pd.read_csv('D:/GSE149836/RelaWT_specific_peaks.bed', sep='\t', names=['chrom', 'start', 'stop','s','sr','st'])

def center_and_expand(df, width=600):
    mid = (df['start'] + df['stop']) // 2
    df['start'] = mid - (width // 2)
    df['stop'] = mid + (width // 2)
    return df

# Center them so the CNN has a consistent view
ko_bed_fixed = center_and_expand(ko_bed)
wt_bed_fixed = center_and_expand(wt_bed)

from pyfaidx import Fasta
bed=ko_bed_fixed
fasta_file = "D:/X-inactivation/mm10.fa"
output_file = "D:/GSE149836/RelaKO600.csv"
# --- Load genome ---
genome = Fasta(fasta_file)

# --- Read BED file ---
#bed = pd.read_csv(bed_file, sep="\t", header=None, comment='#')
#bed = bed.sample(n=66000, random_state=42)
bed.columns = ["chr", "start", "end"] + [f"col{i}" for i in range(3, len(bed.columns))]

# --- Extract sequences ---
sequences = []
for i, row in bed.iterrows():
    try:
        seq = genome[row["chr"]][row["start"]:row["end"]].seq
    except KeyError:
        seq = "N" * (row["end"] - row["start"])  # handle missing chromosomes
    sequences.append(seq)

# --- Add sequence column ---
bed["sequence"] = sequences

# --- Save output ---
bed.to_csv(output_file, sep="\t", header=False, index=False)

print(f"✅ Done! Saved sequences to {output_file}")