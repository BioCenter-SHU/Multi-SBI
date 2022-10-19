import requests
import time
# import urllib3
# urllib3.disable_warnings()

headers={
    'Referer':'https://go.drugbank.com/releases/latest/', 'Connection': 'close',
}
requests.adapters.DEFAULT_RETRIES = 100

f1 = open('durg_smile4.txt', 'w', encoding='utf-8')
with open('small_4.txt', 'r') as f:
    for s in f:
        ss = requests.session()
        ss.keep_alive = False
        ss.proxies = {"https": "57.10.114.47:8000", "http": "32.218.1.7:9999", }
        a,b = s.split()
        response = ss.get(f"https://go.drugbank.com/structures/small_molecule_drugs/{b}.smiles",headers= headers, verify=False)
        c = str(response.content, encoding = "utf-8")
        f1.writelines([f"{b}\t", f"{c}\n"])
        print(b)
        print(c)
f1.close()