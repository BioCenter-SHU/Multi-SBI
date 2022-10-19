import pandas as pd

drug_smile = pd.read_table('bio_seq_148.txt',names = ['d','s'])
proteinall = pd.read_table('biotech_proteinall.txt',names = ['a','b'])


drug_smile_a = set(drug_smile.d)
proteinall_a = set(proteinall.a)


j = set(drug_smile_a & proteinall_a)

proteinall2 = proteinall[proteinall.a.isin(j)].reset_index(drop=True)

proteinall2.to_csv('biotech_proteinall_2.txt', sep='\t', header=None, index=False)

#############################

drug_smile = pd.read_table('durg_smile_1941.txt',names = ['d','s'])
proteinall = pd.read_table('small_proteinall.txt',names = ['a','b'])


drug_smile_a = set(drug_smile.d)
proteinall_a = set(proteinall.a)


j = set(drug_smile_a & proteinall_a)

proteinall2 = proteinall[proteinall.a.isin(j)].reset_index(drop=True)

proteinall2.to_csv('small_proteinall_2.txt', sep='\t', header=None, index=False)
