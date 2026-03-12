import pandas as pd
import numpy as np

# Load the data
filepath = 'GSE107075_Relamatrix.bgcorrected.deseqnorm_anno.txt.gz'
df = pd.read_csv(filepath, sep='\t', compression='gzip')


df['log2FC'] = np.log2((df['Rela-KO'] + 1) / (df['WT_3T3'] + 1))


# KO Peaks
ko_peaks = df[df['log2FC'] > 1.0]

# WT Peaks
wt_peaks = df[(df['log2FC'].abs() < 0.2) & (df['WT_3T3'] > 10)]

# BED file generation
def to_bed(data, filename):
    bed = data[['Peak chromosome', 'Peak start', 'Peak stop', 'id', 'log2FC']]
    bed.to_csv(filename, sep='\t', header=False, index=False)
    print(f"Created {filename} with {len(bed)} peaks.")

to_bed(ko_peaks, 'Rela_KO.bed')
to_bed(wt_peaks, 'Rela_WT.bed')