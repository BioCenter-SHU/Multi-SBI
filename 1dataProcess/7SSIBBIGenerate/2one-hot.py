import pandas as pd
import numpy as np
target = pd.read_table('BBI_7418_2.txt',names = ['a','b'])
enzyme = pd.read_table('SSI_2.txt',names = ['c','d'])

#target
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
p = list(set(target.b.values))
le.fit(p)
target['b'] = le.transform(target['b'])
#enzyme
le = preprocessing.LabelEncoder()
p = list(set(enzyme.d.values))
le.fit(p)
enzyme['d'] = le.transform(enzyme['d'])

target.to_csv('BBI_7418_3.txt', sep='\t', header=None, index=False)
enzyme.to_csv('SSI_3.txt', sep='\t', header=None, index=False)

target['a1'] = target['a']
le = preprocessing.LabelEncoder()
p = list(set(target.a1.values))
le.fit(p)
target['a1'] = le.transform(target['a1'])
biotech_proteinall_feature_matrix = np.zeros((target['a1'].max()+1, target['b'].max()+1), dtype=int)
for index, row in target.iterrows():
    print(row['a'])
    biotech_proteinall_feature_matrix[row['a1']][row['b']] = 1

enzyme['c1'] = enzyme['c']
le = preprocessing.LabelEncoder()
p = list(set(enzyme.c1.values))
le.fit(p)
enzyme['c1'] = le.transform(enzyme['c1'])
small_proteinall_feature_matrix = np.zeros((enzyme['c1'].max()+1, enzyme['d'].max()+1), dtype=int)
for index, row in enzyme.iterrows():
    print(row['c'])
    small_proteinall_feature_matrix[row['c1']][row['d']] = 1

np.savetxt('BBI_feature_matrix.txt',biotech_proteinall_feature_matrix,fmt='%d')
np.savetxt('SSI_feature_matrix.txt',small_proteinall_feature_matrix,fmt='%d')
