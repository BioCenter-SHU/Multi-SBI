import sys

sys.path.append('/home/20721511')

import numpy as np
import pandas as pd

import dataProcess.PULearning as PU


def WriteData(path, data):
    data1 = pd.DataFrame(data)
    data1.to_csv(path, index=False, header=False)


def feature_vector(feature_name):
    # df are the 572 kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    if (feature_name == "daylight"):
        df_feature = pd.read_table('daylight_512.txt', header=None, )
        # sim_matrix = Jaccard(np.array(df_feature))
        # sim_matrix = np.array(df_feature)
        # pca = PCA(n_components=512)  # PCA dimension
        # pca.fit(sim_matrix)
        # sim_matrix = pca.transform(sim_matrix)
        return df_feature
    if (feature_name == "sequence"):
        df_feature = pd.read_table('bio_mse_148.txt', header=None, )
        # pca = PCA(n_components=172)  # PCA dimension
        # pca.fit(df_feature)
        # sim_matrix = pca.transform(df_feature)
        return df_feature
    if (feature_name == "pubchem"):
        df_feature = pd.read_table('pubchem_881.txt', header=None, )


index_all_class_r = pd.read_table('index_all_class.txt', header=None, )
index_all_class2 = index_all_class_r.values.reshape(-1)
index_all_class = index_all_class2[0:int(len(index_all_class2) / 2)]

df_drug = pd.read_table('durg_smile_1941.txt', header=None, names=['name', 'smile'])
df_bio = pd.read_table('bio_seq_148.txt', header=None, names=['name', 'seq'])

dbi = pd.read_table('SBI_40959_name.txt', header=None,
                    names=['drugA', 'drugAname', 'drugB', 'drugBname', 'mechanism', 'nums'])
dbi_array = dbi.values
# train_index = np.where(index_all_class != 0)
# dbi_array = dbi_array[train_index]
# dbi = pd.DataFrame(dbi_array)
# new_col = ['drugA', 'drugAname', 'drugB', 'drugBname', 'mechanism', 'nums']
# dbi.columns = new_col

dbi_null = pd.read_table('SBI_negative_287268.txt', header=None, names=['drugA', 'drugB'])

mechanism = dbi['nums']
drugA = dbi['drugA']
drugB = dbi['drugB']

drugA2 = dbi_null['drugA']
drugB2 = dbi_null['drugB']

d_label = {}
d_feature = {}
b_feature = {}

d_event = []
# 添加ddi
for i in range(len(mechanism)):
    d_event.append(1)
for i in range(len(drugA2)):
    d_event.append(0)
count = {}
for i in d_event:
    if i in count:
        count[i] += 1
    else:
        count[i] = 1
list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 事件数量排序

for i in range(len(list1)):
    d_label[list1[i][0]] = i
vector_drug = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)
vector_bio = np.zeros((len(np.array(df_bio['name']).tolist()), 0), dtype=float)

vector_drug = np.hstack((vector_drug, feature_vector('daylight')))
vector_bio = np.hstack((vector_bio, feature_vector("sequence")))
dsa = len(np.array(df_drug['name']).tolist())
for i in range(len(np.array(df_drug['name']).tolist())):
    d_feature[np.array(df_drug['name']).tolist()[i]] = vector_drug[i]

for i in range(len(np.array(df_bio['name']).tolist())):
    b_feature[np.array(df_bio['name']).tolist()[i]] = vector_bio[i]
new_feature_pos = []
new_feature_nes = []
new_label = []
# print(len(mechanism))
# N = 10
for i in range(len(mechanism)):  # 生成阳性药物对
    # for i in range(N):  # 生成阳性药物对
    new_feature_pos.append(np.hstack((d_feature[drugA[i]], b_feature[drugB[i]])))
    new_label.append(d_label[d_event[i]])

print("load pos drug ok")

# print(len(drugA2))
for i in range(len(drugA2)):  # 生成阴性药物对
    # for i in range(N):  # 生成阴性药物对
    new_feature_nes.append(np.hstack((d_feature[drugA2[i]], b_feature[drugB2[i]])))
    new_label.append(d_label[d_event[i + len(mechanism)]])

print("load nes drug ok")

# print(new_feature_nes[0].shape)

new_feature_nes = np.array(new_feature_nes)
new_feature_pos = np.array(new_feature_pos)

print("over")

propData = pd.read_csv("prop_DecisionTreeClassifier1024.csv", header=None)
propData = propData.values
print(propData.shape)
print(new_feature_nes.shape)
# min_rate = 1
# P = PU.PULearning(new_feature_pos, new_feature_nes)
PU.DrawEembeddingPicture(new_feature_pos, new_feature_nes, new_feature_pos.shape[0], propData)
# WriteData("prop.csv", P.tolist())
