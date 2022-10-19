import requests
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
理论上使用一个API就可以了，不用写两个，将drug2smile和biodrug2seq公用一个函数
"""
headers = {
    'Referer': 'https://go.drugbank.com/releases/latest/', 'Connection': 'close',
}
# requests.adapters.DEFAULT_RETRIES = 100
target_file = open('../data/processed_data/SMD_SMILES.txt', 'w', encoding='utf-8')
with open('../data/raw_data/SMD_List.txt', 'r') as source_file:
    for line in source_file:
        # 由于原始txt文件有index，类似1	DB00104，故后者为 id
        drug_index, drug_id = line.split()
        drug_smile_session = requests.Session()
        drug_smile_session.keep_alive = False
        # 下面链接能直接查询药物的SMILES序列
        response = drug_smile_session.get(f"https://go.drugbank.com/structures/small_molecule_drugs/{drug_id}.smiles",
                                          headers=headers, verify=False)
        drug_smiles = str(response.content, encoding="utf-8")
        target_file.writelines([f"{drug_id}\t", f"{drug_smiles}\n"])
        # 每隔100个输出打印一次
        if int(drug_index) % 100 == 0:
            print(drug_index + "\t" + drug_id)
target_file.close()
