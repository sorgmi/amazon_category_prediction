import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import glob
import numpy as np


def plotCM(labels, predictions,savepath=None, figsize=(30,20), show=True):
    cm = confusion_matrix(labels,predictions)
    #df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],columns = [i for i in "ABCDEFGHIJK"])
    df_cm = pd.DataFrame(cm)
    plt.clf()
    plt.figure(figsize=figsize)
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    if savepath is not None:
        plt.savefig(savepath+"cm.png")

    if show == True:
        plt.show()
    else:
        plt.clf()


def plotfromExperiment(path, epoch):

    files = glob.glob(path+"val_label_pred_hist*.npy")
    if len(files) != 1:
        print(files)
        raise Exception

    for f in files:
        x = np.load(f)[epoch]
        plotCM(x[0], x[1], path, show=False)

def getClassificationReport(path, epoch):
    files = glob.glob(path+"val_label_pred_hist*.npy")
    for f in files:
        x = np.load(f)[epoch]
        r = classification_report(x[0],x[1])
        print(r)



if __name__== "__main__":
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    #plotCM(y_true, y_pred)

    p = "../serverblobs/2019-06-24_balanced_full_baseline/"
    plotfromExperiment(p, -1)