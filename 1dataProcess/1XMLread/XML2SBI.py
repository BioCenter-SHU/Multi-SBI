from lxml import etree
import io
import pandas as pd
def WriteData(path, data):
    data1 = pd.DataFrame(data)
    data1.to_csv(path, sep='\t', header=None, index=False)

def getChildNode(node, contains_str):
    n = None
    for x in node.getchildren():
        if (str(x.tag).__contains__(contains_str)):
            n = x
            break
    return n


def getxml_content():
    tree = etree.parse("full database.xml")  # 获取树结构
    root = tree.getroot()  # 获取根节点
    i = 0
    data = []
    for elments in root:  # 遍历根节点获取子节点
        # 0 drugbank-id;3 姓名
        if (elments.getchildren()[0].text in BiotechDrug_set):
            print(1)
            # n = elments.xpath("//{http://www.drugbank.ca}drug-interactions")
            # print(len(n))
            drug_A_id = elments.getchildren()[0].text
            drug_A_name = getChildNode(elments, "name").text
            n = getChildNode(elments, "drug-interactions")
            if n != None:
                for x in n.getchildren():
                    drug_B_id = getChildNode(x, "drugbank-id").text
                    if (drug_B_id in BiotechDrug_set):
                        drug_B_name = getChildNode(x, "name").text
                        description = getChildNode(x, "description").text
                        #d = [drug_A_id, drug_A_name, drug_B_id, drug_B_name, description]
                        d = [drug_A_id, drug_B_id]
                        data.append(d)


            i = i + 1
            # print("= =")
        # for node in elments.getchildren():
        #     print(node.tag)
        # break
    return data

BiotechDrug = open("bio_seq_194.txt", encoding='gb18030', errors='ignore')
#file = open(path, encoding='gb18030'）
#file = open(path, encoding='gb18030', errors='ignore')
BiotechDrug_set = set()
for drug in BiotechDrug:
    BiotechDrug_set.add(drug.split("\t")[0].strip())

SmallMoleculeDrug = open("durg_smile_6907.txt", encoding='gb18030', errors='ignore')
SmallMoleculeDrug_set = set()
for drug in SmallMoleculeDrug:
    SmallMoleculeDrug_set.add(drug.split("\t")[0].strip())

WriteData("BBI.txt", getxml_content())


