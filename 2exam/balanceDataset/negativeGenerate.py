import numpy as np
import pandas as pd

DecisionTreePro = pd.read_table('..\data\prop_DecisionTreeClassifier1024.csv', header=None, names=['pro'])
dbi_null2 = pd.read_table('..\data\dbi_null_246309.txt', header=None, names=['drugA', 'drugB'])
dbi_null2 = np.hstack((dbi_null2, DecisionTreePro))
dbi_null2 = pd.DataFrame(dbi_null2)
new_col = ['drugA', 'drugB', 'pro']
dbi_null2.columns = new_col
dbi_null2.drop(index=(dbi_null2.loc[(dbi_null2['pro'] >= 1)].index), inplace=True)
dbi_null2.index = range(len(dbi_null2))
# #dbi_null2_1 = dbi_null2.head(len(dbi))
dbi_null2_2 = dbi_null2.sample(n=40959, replace=False, weights=None, random_state=None, axis=0)
dbi_null2_2.index = range(len(dbi_null2_2))
dbi_null2_2_print = pd.DataFrame(dbi_null2_2)

# dbi_null2 = dbi_null.sample(n=len(dbi), replace=False, weights=None, random_state=None, axis=0)
# dbi_null2.index = range(len(dbi_null2))
#
# dbi_null2_print = pd.DataFrame(dbi_null2)
dbi_null2_2_print.to_csv('SBI_negative_40959.txt', sep='\t', header=None, index=False)