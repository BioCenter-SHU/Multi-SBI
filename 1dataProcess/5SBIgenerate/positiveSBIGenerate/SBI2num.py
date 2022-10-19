import pandas as pd

SBI = pd.read_table('SBI2.txt', header=None)
SBI = SBI.rename(columns={0: 's1', 1: 's1_name', 2: 'b1', 3: 'b1_name', 4: 'sbi', 5: 'num'})
SBI


drug_ddi = SBI.sbi
ddi_dict = {}
for ddi in drug_ddi:
    ddi_dict.setdefault(ddi, 0)  # 代码简洁、且性能高
    ddi_dict[ddi] += 1

res = 0
for k, v in ddi_dict.items():
    if (v < 10):
        res = res + v


ddi_dict_shanjian = {}
for k, v in ddi_dict.items():
    if (v > 9):
        # print(k, '=', v)
        ddi_dict_shanjian[k] = v
    else:
        ddi_dict_shanjian[k] = -91


import numpy as np

SBI = np.array(SBI)


for i in range(40959):
    print(SBI[i][4])
    if SBI[i][4] in ddi_dict_shanjian:
        SBI[i][5] = ddi_dict_shanjian[SBI[i][4]]

SBI2 = pd.DataFrame(SBI)

SBI2.to_csv('SBI_40959_name.txt', sep='\t', header=None, index=False)




