# Multi-SBI
## Description
A multi-modal representation learning method for interaction prediction between SMDs and BioDs. 
## Data
The data used in our paper comes from the DrugBank：https://go.drugbank.com/releases/latest
## Enviroment
Anaconda (https://www.anaconda.com)  
PyTorch(python 3.7) (https://pytorch.org)
## DataProcess
### 1.Raw Drugbank XML file parse
File position: paper_code/Multi-SBI_code/1dataProcess/1XMLread  

Input: full database.xml（DrugBank download file）+ XML2SBI.py  
Ouput: bio_seq_194.txt（all BioD sequence）+ durg_smile_6907.txt（all SMD smiles）+ SBI.txt + BBI.txt + SSI.txt

### 2.SMILES generate
File position: paper_code/Multi-SBI_code/1dataProcess/2drug2smile  

Input 1: biotech_433.txt + bio2seq.py  
Ouput 1: bio_seq.txt  
  
Input 2: small_1.txt + drug2smile.py  
Ouput 2： durg_smile1.txt

### 3.Daylight/ESM structure feature encoding
File position: paper_code/Multi-SBI_code/1dataProcess/3structure2feature  
Input 1: bio_seq_148.txt + SqeToVec.py  
Ouput 1: bio_mse.txt  

Input 2: durg_smile_1941.txt + fingerprint.py  
Ouput 2: daylight_1024.txt

### 4.Delete the feature missing drug
File position: paper_code/Multi-SBI_code/1dataProcess/4dataDuplicateFilter  

Input: BSI_41857.txt + durg_smile_6907.txt + bio_seq_194.txt + BBI.txt + SSI.txt + dataReduction.py  
Ouput: SBI_40959.txt + SSI_1348313.txt（SMD relation）+ BBI_7418.txt（BioD relation）+ durg_smile_1941.txt（SMD）+ bio_seq_148.txt（BioD）

### 5.SBI generate
File position: paper_code/Multi-SBI_code/1dataProcess/5SBIgenerate  

Input 1: durg_smile_1941.txt（SMD）+ bio_seq_148.txt（BioD）+ 1AB2null.py+ 2shaixuan.py
Ouput 1: SBI_null_246309.txt（unlabeled sample）  

Input 2: SBI_40959.txt + 1nameReplace.py + SBI2num.py
Ouput 2: SBI_40959_name.txt
### 6.SPI BPI generate
File position: paper_code/Multi-SBI_code/1dataProcess/6SPIBPIGenerate  

Input: bio_seq_148.txt（BioD）+ biotech_proteinall.txt（BioD relation）+ durg_smile_1941.txt（SMD）+ small_proteinall.txt（SPI）+ 1intersectionExtraction.py + 2one-hot.py + 3pca.py   
Ouput: small_protein_matrix512.txt（SPI matrix）+ biotech_protein_matrix148.txt（BPI matrix）

### 7.SSI BBI generate
File position: paper_code/Multi-SBI_code/1dataProcess/7SSIBBIGenerate  
Input: bio_seq_148.txt（BioD）+durg_smile_1941.txt（SMD）+BBI_7418.txt（BioD relation）+SSI_1348313.txt（D relation）+1intersectionExtraction.py+2one-hot.py+3pca.py  
Ouput: SSI_feature_matrix512.txt（SSI 特征矩阵）+BBI_feature_matrix148.txt（BBI 特征矩阵）
