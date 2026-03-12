import pandas as pd


df = pd.read_csv("GSE107075_Rela.matrix.bgcorrected.deseqnorm_anno.txt.gz", sep="\t", compression='gzip')

df['PeakID'] = df.index + 1

# Threshold for presence
threshold = 5

# Separate peaks
wt_specific = df[(df['WT_3T3'] > threshold) & (df['Rela-KO'] <= threshold)]
ko_specific = df[(df['Rela-KO'] > threshold) & (df['WT_3T3'] <= threshold)]
shared = df[(df['WT_3T3'] > threshold) & (df['Rela-KO'] > threshold)]

# Save as BED-like files
wt_specific[['Peak chromosome','Peak start','Peak stop','PeakID','WT_3T3','Rela-KO']].to_csv("RelaWT_specific_peaks.bed", sep="\t", index=False, header=False)
ko_specific[['Peak chromosome','Peak start','Peak stop','PeakID','WT_3T3','Rela-KO']].to_csv("RelaKO_specific_peaks.bed", sep="\t", index=False, header=False)
shared[['Peak chromosome','Peak start','Peak stop','PeakID','WT_3T3','Rela-KO']].to_csv("RelaShared_peaks.bed", sep="\t", index=False, header=False)
