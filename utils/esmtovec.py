import pandas as pd
from xmlread import write_data
import torch
import esm
# import numpy as np
# import CommonHHJ.CsvTools as csv

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

seq_path = "../data/raw_data/bio_seq_148.txt"
raw_data = pd.read_table(seq_path, header=None)
# 转化成 ndarray 数据格式
raw_data = raw_data.values
embedding_data = []
for line in raw_data:
    # 显示 ID
    print(line[0])
    # print(i[1])
    # i[1] = str.replace(i[1], "\n", "")
    # d = (i[0],i)
    seq_data = []
    # 对 sequence 进行阶段取前1024位来进行embedding
    seq_data.append((line[0], str.replace(line[1], "\n", "")[0:1022]))
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    for i, (u, seq) in enumerate(seq_data):
        repre = token_representations[i, 1: len(seq) + 1].mean(0)
        print(token_representations[i, 1: len(seq) + 1].mean(0).shape)
        # Write_data.append([u, seq] + repre.numpy().tolist())
        embedding_data.append(repre.numpy().tolist())
#
# np.zeros().tolist()
# a = [1,2,3]
# b = [4,5,6]
# print(a+b)
# csv.WriteData("sss.csv", Write_data)
write_data("../data/processed_data/bio_mse.txt", embedding_data)
# print(seq_data[0])
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
# ]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

# csv.WriteData()
