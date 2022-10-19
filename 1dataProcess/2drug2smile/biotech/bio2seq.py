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
requests.adapters.DEFAULT_RETRIES = 10
f1 = open('bio_seq.txt', 'w')

with open('biotech_433.txt', 'r') as f:
    for s in f:
        ss = requests.session()
        ss.keep_alive = False
        ss.proxies = {"https": "57.10.114.47:8000", "http": "32.218.1.7:9999", }
        a, b = s.split()
        response = ss.get(f"https://go.drugbank.com/drugs/{b}/polypeptide_sequences.fasta", headers=headers)
        c = str(response.content, encoding="utf-8")
        c = str_info.sub('\t', c).replace('\n', '')
        if c != "":
            f1.writelines([f"{b}", f"{c}\n"])
        print([f"{b}", f"{c}\n"])
f1.close()
