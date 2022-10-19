import requests
import re
"""
功能：提取出拥有氨基酸序列的生物药物的相对序列信息，生成最后的194组DB的ID与氨基酸序列对应关系
输入：生物药物序列：biotech_433.txt 就是属于生物药物的所有药物，但是可能没有氨基酸序列，所以实际有用的只有194个有序列的biotech
输出：194种生物药物的氨基酸序列：bio_seq.txt
疑问：biotech_433.txt 文件怎么提取的，是不是和第一步 XML2SBI.py 一样从整体的xml文件提取
"""


# 使用正则表达式 re 模块的 compile 函数
str_info = re.compile('>.*?\n')
# d = str_info.sub('\t', c).replace('\n','')
headers = {
    'Referer': 'https://go.drugbank.com/releases/latest/'
}
# requests.adapters.DEFAULT_RETRIES = 10
target_file = open('bio_seq.txt', 'w')
with open('biotech_433.txt', 'r') as source_file:
    for line in source_file:
        drug_index, drug_id = line.split()
        drug_sequence_session = requests.session()
        drug_sequence_session.keep_alive = False
        response = drug_sequence_session.get(f"https://go.drugbank.com/drugs/{drug_id}/polypeptide_sequences.fasta",
                                             headers=headers)
        drug_seq = str(response.content, encoding="utf-8")
        drug_seq = str_info.sub('\t', drug_seq).replace('\n', '')
        if drug_seq != "":
            target_file.writelines([f"{drug_id}", f"{drug_seq}\n"])
        print([f"{drug_id}", f"{drug_seq}\n"])
target_file.close()
