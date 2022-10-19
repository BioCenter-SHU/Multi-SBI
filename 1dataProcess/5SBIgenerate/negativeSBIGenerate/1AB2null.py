import pandas as pd

def WriteData(path, data):
    data1 = pd.DataFrame(data)
    data1.to_csv(path, sep='\t', header=None, index=False)

# dbi = pd.read_table('SBI_41598_new.txt', header=None, names=['drugA', 'drugAname', 'drugB', 'drugBname', 'mechanism', 'nums'])

# ddi=[]
# for i in range(len(dbi)):
#     ddi.append(dbi['drugA'] + dbi['drugB'])

df_drug = pd.read_table('durg_smile_1941.txt', header=None, names=['name','smile'])
df_bio = pd.read_table('bio_seq_148.txt', header=None, names=['name','seq'])

drugA = df_drug['name']
drugB = df_bio['name']

Write_data = []
for i in range(len(drugA)):
    for j in range(len(drugB)):
        Write_data.append([drugA[i], drugB[j]])

WriteData("SBI_all.txt", Write_data)

