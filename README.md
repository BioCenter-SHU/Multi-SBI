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

Input: full database.xml(DrugBank download file) + XML2SBI.py  
Ouput: bio_seq_194.txt(all BioD sequence) + durg_smile_6907.txt(all SMD smiles) + SBI.txt + BBI.txt + SSI.txt

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
Ouput: SBI_40959.txt + SSI_1348313.txt(SMD relation) + BBI_7418.txt(BioD relation) + durg_smile_1941.txt(SMD)+ bio_seq_148.txt(BioD)

### 5.SBI generate
File position: paper_code/Multi-SBI_code/1dataProcess/5SBIgenerate  

Input 1: durg_smile_1941.txt(SMD)+ bio_seq_148.txt(BioD)+ 1AB2null.py+ 2shaixuan.py  
Ouput 1: SBI_null_246309.txt(unlabeled samples)  

Input 2: SBI_40959.txt + 1nameReplace.py + SBI2num.py  
Ouput 2: SBI_40959_name.txt

### 6.SPI BPI generate
File position: paper_code/Multi-SBI_code/1dataProcess/6SPIBPIGenerate  

Input: bio_seq_148.txt(BioD)+ biotech_proteinall.txt(BioD relation) + durg_smile_1941.txt(SMD)+ small_proteinall.txt(SPI)+ 1intersectionExtraction.py + 2one-hot.py + 3pca.py   
Ouput: small_protein_matrix512.txt(SPI matrix) + biotech_protein_matrix148.txt(BPI matrix)

### 7.SSI BBI generate
File position: paper_code/Multi-SBI_code/1dataProcess/7SSIBBIGenerate  

Input: bio_seq_148.txt(BioD) + durg_smile_1941.txt(SMD) + BBI_7418.txt(BioD relation) + SSI_1348313.txt(SMD relation) + 1intersectionExtraction.py + 2one-hot.py + 3pca.py  
Ouput: SSI_feature_matrix512.txt(SSI matrix) + BBI_feature_matrix148.txt(BBI matrix)

## Experiment
### 1.UnbalanceDataset
File position: paper_code/Multi-SBI_code/2exam/balanceDataset  

Input 1: daylight_512.txt(SMD feature) + bio_mse_148.txt(BioD feature) + index_all_class.txt(5-fold index) + durg_smile_1941.txt(SMD) + bio_seq_148.txt(BioD) + SBI_40959_name.txt(positive samples) + dbi_null_input.txt(unlabeled samples) + input.py + PULearning.py  
Ouput 1: prop_DecisionTreeClassifier1024.csv(positive sample posibility) 

Input 2: prop_DecisionTreeClassifier1024.csv(positive sample posibility) + SBI_null_246309.txt(unlabeled samples)  
Ouput 2: SBI_negative_40959.txt(negative samples)

### 2.Prediction
File position: paper_code/Multi-SBI_code/2exam/  

#### 2.1 CNN
File position: paper_code/Multi-SBI_code/2exam/cnn  

Input: train_drugs4.txt(SMD train feature) + train_prots4.txt(BioD train feature) + test_drug4.txt(SMD test feature) + test_prots4.txt(BioD test feature) + train_Y4.txt(drug train label) + test_Y4.txt(drug test label) + all_drugs4.txt(SMD) + all_prots4.txt(BioD) + cnn_structure.py  
Ouput: cnn_pred_score.txt(cnn prediction)

#### 2.2 other
File position: paper_code/Multi-SBI_code/2exam/other  

Input: daylight_1024.txt(SMD Daylight embedding) + mse_1280.txt(BioD MSE embedding) + SSI_feature_matrix512.txt(SSI) + BBI_feature_matrix148.txt(BBI) + small_proteinall_feature_matrix.txt(SPI) + biotech_proteinall_feature_matrix.txt(BPI) + index_all_class.txt(5-fold index) + durg_smile_1941.txt + bio_seq_148.txt + SBI_40959_name.txt(positive samples) + SBI_negative_40959.txt(negative samples) + dnn_all.py  
Ouput: new_label_all.txt(durg to label) + other_pred_score.txt + y_test.txt

#### 2.3 Fusion
File position: paper_code/Multi-SBI_code/2exam/evaluation  

Input: y_test.txt + cnn_pred_score.txt + other_pred_score.txt + evaluation.py  
Ouput: final_pred_score.csv+ y_score_multiSBI.txt
