import pandas as pd

# Peek at logCPM (tab-separated)
lc = pd.read_csv('counts_and_metadata/logCPM_matrix_filtered_samples.csv', sep='\t', nrows=5)
print('=== logCPM ===')
print(f'Shape (5 rows): {lc.shape}')
print(f'Columns (first 10): {list(lc.columns[:10])}')
print(f'Columns (last 5): {list(lc.columns[-5:])}')
print(f'First col name: {lc.columns[0]}')
print(lc.iloc[:3, :5])
print()

# Full shape
lc_full = pd.read_csv('counts_and_metadata/logCPM_matrix_filtered_samples.csv', sep='\t', index_col=0)
print(f'Full logCPM: {lc_full.shape}')
print(f'Index name: {lc_full.index.name}, first 5 index: {list(lc_full.index[:5])}')
print()

# Peek at metadata (tab-separated)
md_full = pd.read_csv('counts_and_metadata/metadata_with_sample_annotations.csv', sep='\t')
print('=== Metadata ===')
print(f'Shape: {md_full.shape}')
print(f'Columns: {list(md_full.columns)}')
print(md_full.head(3).to_string())
print()

print(f'Experiment unique: {md_full["Experiment"].nunique()}')
print(f'Run unique: {md_full["Run"].nunique()}')
print(f'BioProject unique: {md_full["BioProject"].nunique()}')
print(f'BioProject values:\n{md_full["BioProject"].value_counts().head(10)}')
print()

# Check overlap: logCPM columns are SRR IDs, match with Run column
sample_cols = set(lc_full.columns)
meta_runs = set(md_full['Run'].astype(str))
overlap = sample_cols & meta_runs
print(f'logCPM sample columns: {len(sample_cols)}')
print(f'Metadata Run IDs: {len(meta_runs)}')
print(f'Overlap (Run): {len(overlap)}')
print(f'In logCPM but not metadata: {len(sample_cols - meta_runs)}')
print(f'In metadata but not logCPM: {len(meta_runs - sample_cols)}')

# Check other metadata columns that might be useful
print(f'\nMetadata column summaries (low cardinality):')
for col in md_full.columns:
    nu = md_full[col].nunique()
    if nu < 30:
        print(f'  {col}: {nu} unique -> {md_full[col].value_counts().head(5).to_dict()}')
