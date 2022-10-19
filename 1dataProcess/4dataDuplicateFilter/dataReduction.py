import pandas as pd
SBI = pd.read_table('BSI_41857.txt',header=None)
SBI = SBI.rename(columns={0:'s1',1:'s1_name',2:'b1',3:'b1_name',4:'sbi'})

durg_smile = pd.read_table('durg_smile_6907.txt',header=None)
durg_smile = durg_smile.rename(columns={0:'drug1',1:'s1'})

bio_seq = pd.read_table('bio_seq_194.txt',header=None)
bio_seq = bio_seq.rename(columns={0:'bio1',1:'s1'})

bio_bio = pd.read_table('BBI.txt',header=None)
bio_bio = bio_bio.rename(columns={0:'bio1',1:'bio2'})

drug_drug = pd.read_table('SSI_1348313.txt',header=None)
drug_drug = drug_drug.rename(columns={0:'drug1',1:'drug2'})

SBI_b1_set = set(SBI.b1)
SBI_s1_set = set(SBI.s1)
durg_smile_drug1_set = set(durg_smile.drug1)
bio_seq_bio1_set = set(bio_seq.bio1)

bio_bio_bio1_set = set(bio_bio.bio1)
drug_drug_drug1_set = set(drug_drug.drug1)
bio_set = SBI_b1_set & bio_seq_bio1_set & bio_bio_bio1_set
drug_set = SBI_s1_set & durg_smile_drug1_set & drug_drug_drug1_set

SBI2 = SBI[SBI.s1.isin(drug_set)].reset_index(drop=True)
SBI3 = SBI2[SBI2.b1.isin(bio_set)].reset_index(drop=True)

SBI3_b1_set = set(SBI3.b1)
SBI3_s1_set = set(SBI3.s1)

drug_drug2 = drug_drug[drug_drug.drug1.isin(SBI3_s1_set)].reset_index(drop=True)
durg_smile2 = durg_smile[durg_smile.drug1.isin(SBI3_s1_set)].reset_index(drop=True)
bio_seq2 = bio_seq[bio_seq.bio1.isin(bio_set)].reset_index(drop=True)
bio_bio2 = bio_bio[bio_bio.bio1.isin(bio_set)].reset_index(drop=True)

SBI3.to_csv('SBI_40959.txt', sep='\t', header=None, index=False)
drug_drug2.to_csv('SSI_1348313.txt', sep='\t', header=None, index=False)
durg_smile2.to_csv('durg_smile_1941.txt', sep='\t', header=None, index=False)
bio_seq2.to_csv('bio_seq_148.txt', sep='\t', header=None, index=False)
bio_bio2.to_csv('BBI_7418.txt', sep='\t', header=None, index=False)
