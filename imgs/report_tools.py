import matplotlib.pyplot as plt
import os, glob, pickle
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report



def plotCM(labels, predictions, save=None, figsize=(30, 20), show=True):
    cm = confusion_matrix(labels, predictions)
    # df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],columns = [i for i in "ABCDEFGHIJK"])
    df_cm = pd.DataFrame(cm)
    plt.clf()
    plt.figure(figsize=figsize)
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    if save is not None:
        plt.savefig(save + "cm.png")

    if show == True:
        plt.show()
    else:
        plt.clf()


def plotExperimentCM(path, epoch, figsize=(30, 20)):
    files = glob.glob(path + "val_label_pred_hist*.npy")
    if len(files) != 1:
        print(files)
        raise Exception

    for f in files:
        x = np.load(f)[epoch]
        plotCM(x[0], x[1], path, show=True, figsize=figsize)


def getClassificationReport(path, epoch):
    files = glob.glob(path + "val_label_pred_hist*.npy")
    for f in files:
        x = np.load(f)[epoch]
        r = classification_report(x[0], x[1])
        print(r)

def plotExperimentLoss(p1, p2, title1, title2, figsize=(10, 5), save=None):

        f1 = glob.glob(p1 + "*loss.npy")
        f2 = glob.glob(p2 + "*loss.npy")
        x1 = np.load(f1[0])
        x2 = np.load(f1[1])
        x3 = np.load(f2[0])
        x4 = np.load(f2[1])
        m = np.max([x1, x2, x3, x4]) + 0.5

        l1 = f1[0][len(p1):-4]
        l2 = f1[1][len(p1):-4]
        l3 = f2[0][len(p2):-4]
        l4 = f2[1][len(p2):-4]

        f = plt.figure(figsize=figsize)
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        ax1.plot(x1, label=l1)
        ax1.plot(x2, label=l2)
        ax1.set_ylim(0, m)
        ax1.legend()
        ax1.set_title(title1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")

        ax2.plot(x3, label=l3)
        ax2.plot(x4, label=l4)
        ax2.set_ylim(0, m)
        ax2.legend()
        ax2.set_title(title2)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("loss")

        plt.tight_layout()
        if save is not None:
            plt.savefig(save + "loss.png")
        plt.show()


def plotExperimentf1(p1, p2, title1, title2, figsize=(10, 5), save=None):
    f1 = glob.glob(p1 + "*f1*.npy")
    f2 = glob.glob(p2 + "*f1*.npy")
    x1 = np.load(f1[0])
    x2 = np.load(f1[1])
    x3 = np.load(f2[0])
    x4 = np.load(f2[1])
    m = np.max([x1, x2, x3, x4]) + 0.02
    mi = np.min([x1, x2, x3, x4]) - 0.02

    l1 = f1[0][len(p1):-4]
    l2 = f1[1][len(p1):-4]
    l3 = f2[0][len(p2):-4]
    l4 = f2[1][len(p2):-4]

    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax1.plot(x1, label=l1)
    ax1.plot(x2, label=l2)
    ax1.set_ylim(mi, m)
    ax1.legend()
    ax1.set_title(title1)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("micro-f1 score")

    ax2.plot(x3, label=l3)
    ax2.plot(x4, label=l4)
    ax2.set_ylim(mi, m)
    ax2.legend()
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("micro-f1 score")
    ax2.set_title(title2)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save + "f1.png")
    plt.show()


def showExperimentInfo(path):
    f = glob.glob(path + "result*.pickle")
    r = pickle.load(open(f[0], "rb"))
    print("Training time:", round(r["train_time_minutes"] / 60, 1), "hours")
    print("Architecture:", r["architecture"])
    return r


def plotLoss(path, title=""):
    f = glob.glob(path + "*train_loss.npy")
    x = np.load(f[0])
    plt.plot(x, label="train_loss")

    f = glob.glob(path + "*val_loss.npy")
    x = np.load(f[0])
    plt.plot(x, label="val_loss")
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    # plt.savefig("4loss.png")
    plt.show()


def plotF1(path, title=""):
    f = glob.glob(path + "*f1_train.npy")
    x = np.load(f[0])
    plt.plot(x, label="f1_train")

    f = glob.glob(path + "*f1_val.npy")
    x = np.load(f[0])
    plt.plot(x, label="f1_val")
    plt.legend()
    plt.ylabel("micro f1 score")
    plt.xlabel("epoch")
    plt.title(title)
    # plt.savefig("4f1.png")
    plt.show()