import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator

# df_feature = pd.read_table('BBI_feature_matrix.txt', header=None, )
# sim_matrix = Jaccard(np.array(df_feature))
# sim_matrix = np.array(df_feature)
# pca = PCA(n_components=148)  # PCA dimension
# pca.fit(sim_matrix)
# sim_matrix = pca.transform(sim_matrix)
# sim_matrix_print = pd.DataFrame(sim_matrix)
# sim_matrix_print.to_csv('BBI_feature_matrix148.txt', sep='\t', header=None, index=False)

df_feature = pd.read_table('SSI_feature_matrix.txt', header=None, )
sim_matrix = Jaccard(np.array(df_feature))
sim_matrix = np.array(df_feature)
pca = PCA(n_components=512)  # PCA dimension
pca.fit(sim_matrix)
sim_matrix = pca.transform(sim_matrix)
sim_matrix_print = pd.DataFrame(sim_matrix)
sim_matrix_print.to_csv('SSI_feature_matrix512.txt', sep='\t', header=None, index=False)
