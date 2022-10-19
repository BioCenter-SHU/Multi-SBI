import pandas as pd
import numpy as np
import torch
import esm
# import CommonHHJ.CsvTools as csv


def write_data(path, data):
    """
    功能：将data以CSV格式输出
    参数：文件路径 path 和 文件本身 data
    输入：data文件
    输出：在 path 路径下的一个 CSV 文件
    """
    data1 = pd.DataFrame(data)
    data1.to_csv(path, sep='\t', header=False, index=False)


# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

seq_path = "../../../data/raw_data/bio_seq_148.txt"
data = pd.read_table(seq_path, header=None)
data = data.values
Write_data = []
for i in data:
    print(i[0])
    # print(i[1])
    # i[1] = str.replace(i[1], "\n", "")
    # d = (i[0],i)
    seq_data = []
    seq_data.append((i[0], str.replace(i[1], "\n", "")[0:1022]))
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    for i, (u, seq) in enumerate(seq_data):
        repre = token_representations[i, 1: len(seq) + 1].mean(0)
        print(token_representations[i, 1: len(seq) + 1].mean(0).shape)
        # Write_data.append([u, seq] + repre.numpy().tolist())
        Write_data.append(repre.numpy().tolist())
#
# np.zeros().tolist()
# a = [1,2,3]
# b = [4,5,6]
# print(a+b)
# csv.WriteData("sss.csv", Write_data)
write_data("bio_mse.txt", Write_data)
# print(seq_data[0])
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
# ]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

# csv.WriteData()
