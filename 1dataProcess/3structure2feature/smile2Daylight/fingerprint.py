import pandas as pd
import numpy as np
from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
import PyFingerprint


if not isJVMStarted():
    cdk_path = PyFingerprint.__path__[0] + '\\CDK\\cdk-2.2.jar'
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
    cdk = JPackage('org').openscience.cdk


def cdk_parser_smiles(smi):
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    try:
        mol = sp.parseSmiles(smi)
    except:
        raise IOError('invalid smiles input')
    return mol


def cdk_fingerprint(smi, fp_type="pubchem", size=1024, depth=6, output='bit'):
    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860
    else:
        nbit = size

    _fingerprinters = {"daylight": cdk.fingerprint.Fingerprinter(size, depth),  # >1015,
                        "extended": cdk.fingerprint.ExtendedFingerprinter(size, depth),  # >1013
                        "graph": cdk.fingerprint.GraphOnlyFingerprinter(size, depth),  # >1002
                        "maccs": cdk.fingerprint.MACCSFingerprinter(),  # 166+1
                        "pubchem": cdk.fingerprint.PubchemFingerprinter(
                            cdk.silent.SilentChemObjectBuilder.getInstance()),  # 881
                        "estate": cdk.fingerprint.EStateFingerprinter(),  # 79
                        "hybridization": cdk.fingerprint.HybridizationFingerprinter(size, depth),  # >1015
                        "lingo": cdk.fingerprint.LingoFingerprinter(depth),  # >968
                        "klekota-roth": cdk.fingerprint.KlekotaRothFingerprinter(),  # 4860
                        "shortestpath": cdk.fingerprint.ShortestPathFingerprinter(size),  # 报错
                        "signature": cdk.fingerprint.SignatureFingerprinter(depth),  # 报错
                        "circular": cdk.fingerprint.CircularFingerprinter(),  # >973
                       }

    mol = cdk_parser_smiles(smi)
    if fp_type in _fingerprinters:
        fingerprinter = _fingerprinters[fp_type]
    else:
        raise IOError('invalid fingerprint type')

    fp = fingerprinter.getBitFingerprint(mol).asBitSet()
    bits = []
    idx = fp.nextSetBit(0)
    while idx >= 0:
        bits.append(idx)
        idx = fp.nextSetBit(idx + 1)
    if output == 'bit':
        return bits
    else:
        vec = np.zeros(nbit)
        vec[bits] = 1
        vec = vec.astype(int)
        return vec


def WriteData(path, data):
    data1 = pd.DataFrame(data)
    #data1.to_csv(path, index=False, header=False)
    data1.to_csv(path, sep='\t', header=None, index=False)


smile = "C[C@@H](O[C@H]1OCCN(CC2=NNC(=O)N2)[C@H]1C1=CC=C(F)C=C1)C1=CC(=CC(=C1)C(F)(F)F)C(F)(F)F"

fr = open("durg_smile_1941.txt", "r")
# fw = open("test_fingerprint.txt", "w")
data = []


def main(args):
    fp_type = args['fp_type'][0] # 分类器

    durg_smile = pd.read_table('durg_smile_1941.txt', header=None)
    durg_smile = durg_smile.rename(columns={0: 'drug1', 1: 's1'})
    durg_smile_set = set(durg_smile.drug1)

    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860
    else:
        nbit = 1024

    for s in fr:
        n = np.zeros(nbit)  # 换
        drug = s.split('\t')
        fingerprint = cdk_fingerprint(drug[1], fp_type)  # 换
        fingerprint2 = cdk_fingerprint(drug[1], 'klekota-roth')  # 换
        fingerprint3 = cdk_fingerprint(drug[1], 'pubchem')  # 换
        # if (len(fingerprint) <= 10 | len(fingerprint2) <= 10 | len(fingerprint3) <= 10):
        if len(fingerprint) <= 10:
            print(s)
            print('daylight')
            durg_smile_set.remove(drug[0])
        elif len(fingerprint2) <= 10:
            print(s)
            print('klekota-roth')
            durg_smile_set.remove(drug[0])
        elif len(fingerprint3) <= 10:
            print(s)
            print('pubchem')
            durg_smile_set.remove(drug[0])
        else:
            n[fingerprint] = 1
            data.append(n)
        # print(s)
        # print(fingerprint)

    durg_smile_shanjian = durg_smile[durg_smile.drug1.isin(durg_smile_set)].reset_index(drop=True)
    durg_smile_shanjian.to_csv('durg_smile_feature.txt', sep='\t', header=None, index=False)

    WriteData(str(fp_type)+"_"+str(nbit)+".txt", data)  # 换


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--fp_type", choices=["daylight", "maccs", "estate", "pubchem", "klekota-roth"],
                        default=["daylight"], help="特征提取方式")
    args = vars(parser.parse_args())
    print(args)
    main(args)


