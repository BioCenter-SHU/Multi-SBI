import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping

def save_result(feature_name, result_type, result):
    with open(feature_name + result_type + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    res = _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
    return res

def evaluate(pred_score, y_test, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_type = np.argmax(np.array(pred_score), axis=1).reshape(-1, 1)
    y_test = np.array(y_test)
    pred_score = np.array(pred_score)

    a = accuracy_score(y_test, pred_type)
    result_all[0] = round(a, 4)
    b = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[1] = round(b, 4)
    c = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = round(c, 4)
    d = f1_score(y_test, pred_type, average='macro')
    result_all[3] = round(d, 4)
    e = precision_score(y_test, pred_type, average='macro')
    result_all[4] = round(e, 4)
    f = recall_score(y_test, pred_type, average='macro')
    result_all[5] = round(f, 4)
    return result_all

y_true = pd.read_table('y_test.txt', header=None)
y_score_dnn = pd.read_table('cnn_pred_score.txt', header=None)
y_score_ssi = pd.read_table('other_pred_score.txt', header=None)
y_score0 = y_score_dnn + y_score_ssi
y_score = y_score0 / 2
y_score.to_csv('y_score_multiSBI.txt', sep='\t', header=None, index=False)
result_all = evaluate(y_score, y_true, 49)
save_result( 'final_' , "pred_score", result_all)