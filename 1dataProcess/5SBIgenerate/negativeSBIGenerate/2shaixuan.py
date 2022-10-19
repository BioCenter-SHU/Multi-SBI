import pandas as pd
dbi = pd.read_table('SBI_all.txt', header=None, names=['drugA', 'drugB'])

dbi.drop_duplicates(inplace=True, keep = False)

dbi.to_csv('SBI_negative_287268.txt', sep='\t', header=None, index=False)