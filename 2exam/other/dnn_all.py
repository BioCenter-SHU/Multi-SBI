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
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def DNN(event_num, vector_size, droprate):
    train_input = Input(shape=(vector_size,), name='Inputlayer')
    train_in = Dense(512, activation='relu')(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num)(train_in)
    out = Activation('sigmoid')(train_in)
    model = Model(input=train_input, output=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# def dpi_prepare(df_drug, df_bio, drug_feature, bio_feature,vector_size,mechanism,drugA,drugB):
def dpi_prepare(df_drug, df_bio, drug_feature_list, bio_feature_list, vector_size, mechanism, drugA, drugB, drugA2, drugB2,
                categoryType):
    d_label = {}
    d_feature = {}
    b_feature = {}

    d_event = []
    # 添加DDI
    if categoryType == "two":
        for i in range(len(mechanism)):
            d_event.append(1)
        for i in range(len(mechanism)):
            d_event.append(0)
    elif categoryType == "all":
        for i in range(len(mechanism)):
            d_event.append(mechanism[i])
        for i in range(len(mechanism)):
            d_event.append(-1)
    elif categoryType == "multi":
        for i in range(len(mechanism)):
            d_event.append(mechanism[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 事件数量排序
    # list1_1 = pd.DataFrame(list1)
    # list1_1.to_csv('ddiSort_all.txt', sep='\t', header=None, index=False)
    # list1 = pd.read_table('ddiSort_all.txt', header=None, )
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    vector_drug = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)
    vector_bio = np.zeros((len(np.array(df_bio['name']).tolist()), 0), dtype=float)

    for i in drug_feature_list:
        vector_drug = np.hstack((vector_drug, feature_vector(i)))  # np.hstack():在水平方向上平铺拼接数组
    for i in bio_feature_list:
        vector_bio = np.hstack((vector_bio, feature_vector(i)))  # np.hstack():在水平方向上平铺拼接数组
    #vector_drug = np.hstack((vector_drug, feature_vector2(drug_feature, vector_size)))  # np.hstack():在水平方向上平铺拼接数组
    #vector_bio = np.hstack((vector_bio, feature_vector2(bio_feature, vector_size)))  # np.hstack():在水平方向上平铺拼接数组

    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector_drug[i]

    for i in range(len(np.array(df_bio['name']).tolist())):
        b_feature[np.array(df_bio['name']).tolist()[i]] = vector_bio[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []
    for i in range(len(mechanism)):  # 生成药物对
        new_feature.append(np.hstack((d_feature[drugA[i]], b_feature[drugB[i]])))
        new_label.append(d_label[d_event[i]])
    if (categoryType == 'two') | (categoryType == 'all'):
        for i in range(len(mechanism)):  # 生成药物对
            new_feature.append(np.hstack((d_feature[drugA2[i]], b_feature[drugB2[i]])))
            new_label.append(d_label[d_event[i + len(mechanism)]])

    new_feature = np.array(new_feature)  # 功能： 将数据转化为矩阵
    new_label = np.array(new_label)
    new_label2 = pd.DataFrame(new_label)
    new_label2.to_csv('new_label_all.txt', sep='\t', header=None, index=False)
    return (new_feature, new_label)


def feature_vector(feature_name):
    # df are the 572 kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    print(str(feature_name))
    if (feature_name == "daylight"):
        df_feature = pd.read_table('D:\pyFile\DPI8\\1data\daylight_1024.txt', header=None, )
        return df_feature
    elif (feature_name == "sequence"):
        df_feature = pd.read_table('D:\pyFile\DPI8\\1data\mse_1280.txt', header=None, )
        return df_feature
    elif (feature_name == "ssi"):
        df_feature = pd.read_table('D:\pyFile\DPI8\\1data\SSI_feature_matrix512.txt', header=None, )
        return df_feature
    elif (feature_name == "bbi"):
        df_feature = pd.read_table('D:\pyFile\DPI8\\1data\BBI_feature_matrix148.txt', header=None, )
        return df_feature
    elif (feature_name == "sprotein"):
        df_feature = pd.read_table('D:\pyFile\DPI8\\1data\small_proteinall_feature_matrix.txt', header=None, )
        return df_feature
    elif (feature_name == "bprotein"):
        df_feature = pd.read_table('D:\pyFile\DPI8\\1data\\biotech_proteinall_feature_matrix.txt', header=None, )
        return df_feature

    return sim_matrix


def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1

    return index_all_class


def cross_validation(feature_matrix, label_matrix, event_num, vector_size, droprate, epoch, batch_num, seed, CV,
                     categoryType,drug_feature_listName,bio_feature_listeName):
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    # index_all_class = get_index(label_matrix, event_num, seed, CV)
    # index_all_class2 = pd.DataFrame(index_all_class)
    # index_all_class2.to_csv('index_all_class.txt', sep='\t', header=None, index=False)
    index_all_class_r = pd.read_table('D:\pyFile\DPI8\\1data\index_all_class.txt', header=None, )
    index_all_class2 = index_all_class_r.values.reshape(-1)
    if categoryType == "multi":
        index_all_class = index_all_class2[0:int(len(index_all_class2) / 2)]
    else:
        index_all_class = index_all_class2[0:int(len(index_all_class2))]

    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        feature_matrix = matrix
    for k in range(0, 1):
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        for i in range(len(feature_matrix)):
            x_train = feature_matrix[i][train_index]
            x_test = feature_matrix[i][test_index]
            y_train = label_matrix[train_index]
            # one-hot encoding
            y_train_one_hot = np.array(y_train)
            y_train_one_hot = (np.arange(y_train_one_hot.max() + 1) == y_train[:, None]).astype(dtype='float32')
            y_test = label_matrix[test_index]
            # one-hot encoding
            y_test_one_hot = np.array(y_test)
            y_test_one_hot = (np.arange(y_test_one_hot.max() + 1) == y_test[:, None]).astype(dtype='float32')
            a = np.shape(x_train)[1]
            dnn = DNN(event_num, a, droprate)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            dnn.fit(x_train, y_train_one_hot, batch_size=batch_num, epochs=epoch,
                    validation_data=(x_test, y_test_one_hot), callbacks=[early_stopping])
            pred += dnn.predict(x_test)
    pred_score = pred / len(feature_matrix)
    pred_type = np.argmax(pred_score, axis=1)  # 获取array的某一个维度中数值最大的那个元素的索引
    y_true = np.hstack((y_true, y_test))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score))  # numpy 数组增加列，增加行的函数：column_stack,row_stack

    y_test_print = pd.DataFrame(y_true)
    y_test_print.to_csv('y_test.txt', sep='\t', header=None, index=False)

    #pred_type_print = pd.DataFrame(y_pred)
    #pred_type_print.to_csv(str(categoryType) + '_' + 'y_pred.txt', sep='\t', header=None, index=False)

    pred_score_print = pd.DataFrame(y_score)
    pred_score_print.to_csv(str(categoryType) +drug_feature_listName+bio_feature_listeName+ '_' + 'pred_score_test.txt', sep='\t', header=None, index=False)

    result_all = evaluate(y_pred, y_score, y_true, event_num, categoryType)

    return result_all


def evaluate(pred_type, pred_score, y_test, event_num, categoryType):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))

    # precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)

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


def self_metric_calculate(y_true, pred_type):
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


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


# 保存结果
def save_result(feature_name, result_type, result):
    with open(feature_name + result_type + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


def main(args):
    # start = time.clock()
    categoryType = args['categoryType'][0]
    seed = 0
    CV = 5
    vector_size = 572
    event_num = 2  # 48
    if categoryType == "multi":
        event_num = 48
    elif categoryType == "all":
        event_num = 49
    epoch = int(args['epoch'][0])
    numb = int(args['numb'][0])
    batch_num = int(args['batch_num'][0])
    droprate = float(args['droprate'][0])
    print('epoch:' + str(epoch) + 'batch_num:' + str(batch_num) + 'droprate:' + str(droprate) + '18')
    drug_feature_list = args['dFeatureList']
    bio_feature_list = args['bFeatureList']
    drug_feature_listName = "+".join(drug_feature_list)
    bio_feature_listeName = "+".join(bio_feature_list)
    print(str(categoryType) + '_' + drug_feature_listName + '_' + bio_feature_listeName, '_e' + str(epoch) + '_b' + str(batch_num) + '_d' + str(droprate))
    # drug_feature_num = len(drug_feature)
    # drug_featureName = "+".join(drug_feature)
    #
    # bio_feature_num = len(bio_feature)
    # bio_featureName = "+".join(drug_feature)

    df_drug = pd.read_table('D:\pyFile\DPI8\\1data\durg_smile_1941.txt', header=None, names=['name', 'smile'])
    df_bio = pd.read_table('D:\pyFile\DPI8\\1data\\bio_seq_148.txt', header=None, names=['name', 'seq'])
    dbi = pd.read_table('D:\pyFile\DPI8\\1data\SBI_40959_name.txt', header=None,
                        names=['drugA', 'drugAname', 'drugB', 'drugBname', 'mechanism', 'nums'])
    dbi_null2 = pd.read_table('D:\pyFile\DPI8\\1data\dbi_null_40959_no1.txt', header=None, names=['drugA', 'drugB', 'prop'])
    # DecisionTreePro = pd.read_table('D:\pyFile\DPI6\\1data\prop_DecisionTreeClassifier1024.csv', header=None, names=['pro'])
    # dbi_null2 = pd.read_table('D:\pyFile\DPI6\\1data\dbi_null_246309.txt', header=None, names=['drugA', 'drugB'])
    # dbi_null2 = np.hstack((dbi_null2, DecisionTreePro))
    # dbi_null2 = pd.DataFrame(dbi_null2)
    # new_col = ['drugA', 'drugB', 'pro']
    # dbi_null2.columns = new_col
    # dbi_null2.drop(index=(dbi_null2.loc[(dbi_null2['pro'] >= 1)].index), inplace=True)
    # dbi_null2.index = range(len(dbi_null2))
    # # #dbi_null2_1 = dbi_null2.head(len(dbi))
    # dbi_null2_2 = dbi_null2.sample(n=len(dbi), replace=False, weights=None, random_state=None, axis=0)
    # dbi_null2_2.index = range(len(dbi_null2_2))
    # dbi_null2_2_print = pd.DataFrame(dbi_null2_2)

    # dbi_null2 = dbi_null.sample(n=len(dbi), replace=False, weights=None, random_state=None, axis=0)
    # dbi_null2.index = range(len(dbi_null2))
    #
    # dbi_null2_print = pd.DataFrame(dbi_null2)
    # dbi_null2_print.to_csv('dbi_null_suiji.txt', sep='\t', header=None, index=False)

    mechanism = dbi['nums']
    drugA = dbi['drugA']
    drugB = dbi['drugB']
    drugA2 = dbi_null2['drugA']
    drugB2 = dbi_null2['drugB']

    new_feature, new_label = dpi_prepare(df_drug, df_bio, drug_feature_list, bio_feature_list, vector_size, mechanism, drugA,
                                         drugB, drugA2, drugB2, categoryType)
    # new_feature, new_label = dpi_prepare(df_drug, df_bio, drug_feature, bio_feature, vector_size, mechanism, drugA,
    # drugB)

    all_matrix_first = []
    all_matrix_first.append(new_feature)
    # for feature in drug_feature_list:
    #     print(feature)
    #     new_feature, new_label = prepare(df_drug, [feature], vector_size, mechanism,drugA,drugB)
    #     all_matrix_first.append(new_feature)
    #
    # for feature in bio_feature_list:
    #     print(feature)
    #     new_feature, new_label = prepare(df_drug, [feature], vector_size, mechanism,drugA,drugB)
    #     all_matrix_first.append(new_feature)

    all_result = cross_validation(all_matrix_first, new_label, event_num, vector_size, droprate, epoch, batch_num, seed,
                                  CV, categoryType,drug_feature_listName,bio_feature_listeName)
    save_result(str(categoryType) + '_' + drug_feature_listName + '_' + bio_feature_listeName,
                '_e' + str(epoch) + '_b' + str(batch_num) + '_d' + str(droprate), all_result)
    # print("time used:", time.clock() - start)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--categoryType", choices=["two", "multi", "all"], default=["all"], help=" type",
                        nargs="+")
    parser.add_argument("-b", "--batch_num", choices=["256", "128", "64"], default=["128"], help=" batch_size",
                        nargs="+")
    parser.add_argument("-e", "--epoch", choices=["100", "50", "1"], default=["22"], help="epoch", nargs="+")
    parser.add_argument("-n", "--numb", choices=["1", "2", "3", "4", "5"], default=["1"], help="numb", nargs="+")
    parser.add_argument("-d", "--droprate", choices=["0.2", "0.3", "0.1"], default=["0.3"], help="droprate", nargs="+")
    parser.add_argument("-fd", "--dFeatureList", default=["daylight"],
                        help="ssi-daylight-sprotein", nargs="+")
    parser.add_argument("-fb", "--bFeatureList", default=["sequence"], help="bbi-sequence-bprotein", nargs="+")
    args = vars(parser.parse_args())
    print(args)
    main(args)
