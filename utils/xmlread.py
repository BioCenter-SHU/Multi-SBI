from lxml import etree
import pandas as pd
"""
功能：XML->CSV，即将 DrugBank 网站上下载的药物（包括生物医药）XML文件（树状结构）筛选得出药物相关数据BBI（生物医药的相互作用）
构成：定义了3个函数，来实现这个任务
    -write_data(path, data)：写入文件的函数
    -get_child_node(node, contains_str)：
    -get_xml_content()：
输入：
    -生物药物的氨基酸序列：bio_seq_194.txt
    -所有6907种药物的SMILES序列：drug_smile_6907.txt
输出：194种生物药物之间存在的相互作用：BBI.txt
注意：
"""


def write_data(path, data):
    """
    功能：将data以CSV格式输出
    参数：文件路径 path 和 文件本身 data
    输入：data文件
    输出：在 path 路径下的一个 CSV 文件
    """
    data1 = pd.DataFrame(data)
    data1.to_csv(path, sep='\t', header=False, index=False)


def get_child_node(node, contains_str):
    """
    将结点node的所有子结点遍历筛选对应属性结点
    API：
        -Module: lxml.etree._Element.getchildren() 是Python的内置方法
        -Function: Returns all direct children. The elements are returned in document order.
        -Attention: 该方法已在2.0版本被舍弃，应该使用list(element)进行简单遍历
    """
    n = None
    # 将所有子节点中匹配 contains_str 字符的结点依次返回结点
    # 其作用就是把所有包含关键字的子结点筛选出来
    for x in node.getchildren():
        if str(x.tag).__contains__(contains_str):
            n = x
            break
    return n


def get_xml_content():
    # 获取XML文件，返回一个 ElementTree 对象
    tree = etree.parse("../data/raw_data/full database.xml")
    # 获取 tree 对象的根节点，可以把 root 看作是一个 list，因此下面 for 循环才可以迭代访问
    root = tree.getroot()
    # 新建计数 i 统计所有的 BBI 数量
    i = 0
    # 新建 BBI 数据 list 来存放每一对 BBI 数据
    data = []
    # 遍历根节点获取子节点
    for elements in root:
        # 0 DrugBank-id;3 姓名
        if elements.getchildren()[0].text in biotech_drug_set:
            # n = elements.xpath("//{http://www.drugbank.ca}drug-interactions")
            # print(len(n))
            drug_a_id = elements.getchildren()[0].text
            # drug_a_name = get_child_node(elements, "name").text
            # 调用自定义函数来遍历 elements 对象 tag 为"drug-interactions"的结点
            n = get_child_node(elements, "drug-interactions")
            # 当 n 也就是结点非空的时候进行
            if n is not None:
                # 对于结点 n 的所有 DDI 进行筛选，得到 BBI
                for x in n.getchildren():
                    drug_b_id = get_child_node(x, "drugbank-id").text
                    if drug_b_id in biotech_drug_set:
                        # drug_b_name = get_child_node(x, "name").text
                        # description = get_child_node(x, "description").text
                        # d = [drug_a_id, drug_a_name, drug_b_id, drug_b_name, description]
                        # 新建 list 来存储 DDI（BBI） 对
                        d = [drug_a_id, drug_b_id]
                        # 并将 BBI list 存到一开始新建的 data list中
                        data.append(d)
            i = i + 1
            # print("= =")
        # for node in elements.getchildren():
        #     print(node.tag)
        # break
    # 输出所有 BBI 的 list
    return data


biotech_drug = open("../data/raw_data/bio_seq_194.txt", encoding='gb18030', errors='ignore')
# file = open(path, encoding='gb18030'）
# file = open(path, encoding='gb18030', errors='ignore')
biotech_drug_set = set()
# 将所有生物药物的 DB ID 收集至 biotech_drug_set 集合里
for drug in biotech_drug:
    biotech_drug_set.add(drug.split("\t")[0].strip())

small_molecule_drug = open("../data/raw_data/drug_smile_6907.txt", encoding='gb18030', errors='ignore')
small_molecule_drug_set = set()
# 将所有小分子药物的 DB ID 收集至 small_molecule_drug_set 集合里
for drug in small_molecule_drug:
    small_molecule_drug_set.add(drug.split("\t")[0].strip())

write_data("../data/processed_data/BBI.txt", get_xml_content())
