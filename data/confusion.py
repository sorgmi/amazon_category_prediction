import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def plotCM(labels, predictions,savepath=None, figsize=(10, 7)):
    cm = confusion_matrix(labels,predictions)
    #df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],columns = [i for i in "ABCDEFGHIJK"])
    df_cm = pd.DataFrame(cm)
    plt.clf()
    plt.figure(figsize=figsize)
    sn.heatmap(df_cm, annot=True, fmt='d')

    if savepath is not None:
        plt.savefig(savepath+"cm.png")

    plt.show()




if __name__== "__main__":
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    plotCM(y_true, y_pred)