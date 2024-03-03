import numpy as np
import sklearn.metrics
from sklearn import metrics

def compute_auc_eer_phonematchPaper(label, pred):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr
    auc = metrics.auc(fpr, tpr)
    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2

    print("AUC:", auc)
    print("EER:", eer)
    return eer

def compute_auc_eer(label, pred, f, type_name):
    # 计算TPR和FPR
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    print("thresholds:", thresholds)

    # 计算AUC
    auc = metrics.auc(fpr, tpr)

    # 计算EER
    # Calculate EER (Equal Error Rate)
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    f.write(type_name+'\n')
    f.write("AUC:" + str(auc) + '\n')
    f.write("EER:" + str(eer) + '\n')

    print("AUC:", auc)
    print("EER:", eer)


def load_result_file(path_file):
    label_easy = []
    pred_easy = []
    label_hard = []
    pred_hard = []
    with open(path_file, "r") as f:
        content = f.readlines()
        for line in content:
            line = line.split()
            type_name = line[-4]
            if 'positive' in type_name:
                label_easy.append(int(line[-3]))
                pred_easy.append(int(line[-1]))
                label_hard.append(int(line[-3]))
                pred_hard.append(int(line[-1]))
            elif 'easy' in type_name:
                label_easy.append(int(line[-3]))
                pred_easy.append(int(line[-1]))
            elif 'hard' in type_name:
                label_hard.append(int(line[-3]))
                pred_hard.append(int(line[-1]))
    return np.array(label_easy), np.array(pred_easy), np.array(label_hard), np.array(pred_hard)


if __name__ == "__main__":
    pathFile = "/s3prl/result/downstream/sp_debug/test_all.txt"
    resultFile = "/s3prl/result/downstream/sp_debug/result_auc_eer.txt"
    with open(resultFile, "w") as f:
        label_easy, pred_easy, label_hard, pred_hard = load_result_file(pathFile)
        compute_auc_eer(label_easy, pred_easy, f, "libriphrase_easy:")
        compute_auc_eer(label_hard, pred_hard, f, "libriphrase_hard:")
