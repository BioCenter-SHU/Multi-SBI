import requests
import time
import urllib3
urllib3.disable_warnings()
"""
功能：从网络上去下载对应药物的SMILES字符串
核心API：requests 模块，Python 内置了 requests 模块，该模块主要用来发 送 HTTP 请求，requests 模块比 urllib 模块更简洁。
        每次调用 requests 请求之后，会返回一个 response 对象，该对象包含了具体的响应信息。
输入：
输出：
疑问：
    1.为什么不直接从XML获取呢？是不是为了获得最新的数据库
    2.为什么分成7个文件进行？
    3.原始的small名单从何而来？
    4.不需要VPN貌似也可以访问？
"""
headers = {
    'Referer': 'https://go.drugbank.com/releases/latest/', 'Connection': 'close',
}
# requests.adapters.DEFAULT_RETRIES = 100
target_file = open('../../data/processed_data/drug_smile1.txt', 'w', encoding='utf-8')
with open('../../data/raw_data/small_1.txt', 'r') as source_file:
    for line in source_file:
        # 由于原始txt文件有index，类似1	DB00104，故后者为 id
        index, id = line.split()
        ss = requests.Session()
        ss.keep_alive = False
        # 下面链接能直接查询药物的SMILES序列
        response = ss.get(f"https://go.drugbank.com/structures/small_molecule_drugs/{id}.smiles",
                          headers=headers, verify=False)
        smiles = str(response.content, encoding="utf-8")
        target_file.writelines([f"{id}\t", f"{smiles}\n"])
        print(id)
        print(smiles)
target_file.close()
