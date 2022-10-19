import pandas as pd
SBI = pd.read_table('SBI_40959.txt',header=None)
SBI = SBI.rename(columns={0:'s1',1:'s1_name',2:'b1',3:'b1_name',4:'sbi'})

for index,row in SBI.iterrows():
    print(row['sbi'])
    row['sbi'] = row['sbi'].replace(row['s1_name'],'name')
    row['sbi'] = row['sbi'].replace(row['b1_name'],'name')
    print(row['sbi'])


SBI.to_csv('SBI2.txt', sep='\t', header=None, index=False)

