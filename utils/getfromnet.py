import requests
import urllib3
urllib3.disable_warnings()
"""
功能：提取出拥有氨基酸序列的生物药物的相对序列信息，生成最后的194组DB的ID与氨基酸序列对应关系
输入：生物药物序列：biotech_433.txt 就是属于生物药物的所有药物，但是可能没有氨基酸序列，所以实际有用的只有194个有序列的biotech
输出：194种生物药物的氨基酸序列：bio_seq.txt
"""


def get_feature_from_network(headers, target_path, source_path, url_head, url_tail, regular=None):
    target_file = open(target_path, 'w')
    with open(source_path, 'r') as source_file:
        for line in source_file:
            drug_index, drug_id = line.split()
            drug_session = requests.session()
            drug_session.keep_alive = False
            response = drug_session.get(f"{url_head}/{drug_id}/{url_tail}", headers=headers)
            drug_feature = str(response.content, encoding="utf-8")
            if regular is not None:
                drug_feature = regular.sub('\t', drug_feature).replace('\n', '')
            if drug_feature != "":
                target_file.writelines([f"{drug_id}", f"{drug_feature}\n"])
            # 每隔100个输出打印一次
            if int(drug_index) % 100 == 0:
                print([f"{drug_id}", f"{drug_feature}\n"])
    target_file.close()
