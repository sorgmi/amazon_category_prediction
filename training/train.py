import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

from data import load_dataset
from data import confusion
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import glob, time, datetime, os, pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix, classification_report

'''
Model (tensorflow graph) definition. This class gets a list representing the architecture as parameter.
For exaple [False, 150, "r", "d"] means we use non-trainable XLING embeddings + fc layer with 150 units followed
by relu and dropout. The final classification layer is always a softmax layer.
'''
class Model:
    def __init__(self, data_X, data_Y, n_classes, params):
        self.params = params
        self.n_class = n_classes
        self.architecture = params["architecture"][1:]
        #print("Downloading xling...")
        self.xling = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1", trainable=params["architecture"][0])
        self.data_X = data_X
        self.data_Y = data_Y
        self.create_architecture(data_X, data_Y)

    def create_architecture(self, data_X, data_Y):
        self.logits = self.forward(data_X)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data_Y, logits=self.logits))
        self.vars   = tf.trainable_variables()
        self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.vars if 'bias' not in v.name ]) * 0.001
        
        if self.params["regularization"] == True:
          self.train_op = self.params["optimizer"].minimize(self.loss + self.lossL2)
        else:
          self.train_op = self.params["optimizer"].minimize(self.loss)

        self.predictions = tf.argmax(self.logits, 1)
        self.labels = data_Y

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.cast(data_Y, tf.int64)), tf.float32))

    def forward(self, X):
        output = self.xling(X)

        for index, x in enumerate(self.architecture):
            name = None
            if index == len(self.architecture)-1:
                name = "final_logits"  # specify a name for restoring a saved graph model
            if x == "bn":
                output = tf.layers.batch_normalization(output, training=True, name=name)
            elif x == "relu" or x == "r":
                output = tf.nn.relu(output, name=name)
            elif x == "dropout" or x == "d":
                output = tf.layers.dropout(output, name=name)
            else:
                output = tf.layers.dense(output, x, name=name)

        output = tf.layers.dense(output, self.n_class, name="final_output_prediction")

        return output


def plotResults(x1, label1, x2, label2, title, path, architecture, showPlot=False):
    plt.plot(x1, label=label1)
    plt.plot(x2, label=label2)
    plt.legend()
    plt.title(title)
    # plt.ylim(0,1)
    figpath = path + title + ".png"
    # figpath = figpath.replace()
    # print(figpath)
    plt.savefig(figpath)
    if showPlot == True:
        plt.show()
    else:
        plt.clf()

    np.save(path + str(architecture) + label1 + ".npy", x1)
    np.save(path + str(architecture) + label2 + ".npy", x2)


def plotAll(path, showPlot=False):
    for f in glob.glob(path + "*loss.npy"):
        x = np.load(f)
        plt.plot(x, label=f[len(path):-4])
    plt.legend()
    plt.ylim(0, 5)
    plt.savefig(path + "allloss.png")
    if showPlot == True:
        plt.show()
    else:
        plt.clf()

    for f in glob.glob(path + "*acc*.npy"):
        x = np.load(f)
        plt.plot(x, label=f[len(path):-4])
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(path + "allacc.png")
    if showPlot == True:
        plt.show()
    else:
        plt.clf()

    for f in glob.glob(path + "*f1*.npy"):
        x = np.load(f)
        plt.plot(x, label=f[len(path):-4])
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(path + "allf1.png")
    if showPlot == True:
        plt.show()
    else:
        plt.clf()

'''
This is the main training loop. Here we load the datasets and iterate over theem the respective number of epochs.
For each epoch we train and evaluate and additionally plot the loss and other metrics.
Optionally the results (loss, metrics...), model checkpoints and model checkpoints can be saved to a folder for later
evaluation.
'''
def trainModel(p):
    # init default params
    params = {}
    #params["trainData"] = "US"
    #params["testData"] = "DE"
    params["epochs"] = 15
    params["batchSize"] = 512
    params["optimizer"] = tf.train.AdamOptimizer(learning_rate=0.001)
    params["trainexamples"] = 1000 * 100
    #params["architecture"] = [False]
    params["f1modus"] = "micro"
    params["savelog"] = True
    params["checkpoint"] = True
    params["path"] = "blobs/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
    params["pathToCache"] = "data/"
    params["notebook"] = False
    params["showPlots"] = False
    params["includeReviewHeadline"] = False
    params["filterOtherLanguages"] = False
    params["regularization"] = False

    params.update(p)  # overwrite default parameter with passed parameter
    params["learning_rate"] = params["optimizer"]._lr

    if params["notebook"] == True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm


    tf.reset_default_graph()
    dataset_train, num_classes = load_dataset.getData(params["trainData"], shuffle=True, batchsize=params["batchSize"], pathToCache=params["pathToCache"], includeHeading=params["includeReviewHeadline"],maxrows=params["trainexamples"], filterOtherLangs=params["filterOtherLanguages"])
    dataset_val, num_classes2 = load_dataset.getData(params["testData"], shuffle=False, batchsize=params["batchSize"], pathToCache=params["pathToCache"], includeHeading=params["includeReviewHeadline"],maxrows=params["trainexamples"], filterOtherLangs=params["filterOtherLanguages"])

    if num_classes != num_classes2:
        raise Exception("number of classes do not match between train and test set")

    print("classes:", num_classes)
    params["num_classes"] = num_classes

    if params["savelog"] == True:
        print("saving to:", params["path"])
        if os.path.exists(params["path"]) is False:
            os.mkdir(params["path"])
        f = open(params["path"] + "info.txt", "w")
        for k in params:
            f.write(k + ": " + str(params[k]) + "\n")
        f.close()

    if params["trainexamples"] is not None:
        dataset_train = dataset_train.take(int(params["trainexamples"] / params["batchSize"]))
        dataset_val = dataset_val.take(int(params["trainexamples"] / params["batchSize"]))

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    train_iterator = iterator.make_initializer(dataset_train)
    val_iterator = iterator.make_initializer(dataset_val)
    text_input, label = iterator.get_next()

    model = Model(text_input, label, num_classes, params)

    init_op = tf.group([tf.local_variables_initializer(), tf.global_variables_initializer(), tf.tables_initializer()])
    sess = tf.Session()
    sess.run(init_op)

    loss_hist, acc_hist, val_loss_hist, val_acc_hist = [], [], [], []
    loss_hist_epoch, acc_hist_epoch, val_loss_hist_epoch, val_acc_hist_epoch, f1_train_epoch, f1_val_epoch = [], [], [], [], [], []

    val_labels_pred_hist = []

    saver = tf.train.Saver(max_to_keep=5)
    startTime = time.time()
    for epoch in tqdm(range(params["epochs"])):
        # print('\nEpoch: {}'.format(epoch + 1))
        train_loss, train_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0
        counter = 0

        train_predictions, train_labels, val_predictions, val_labels = [], [], [], []

        sess.run(train_iterator)

        try:
            with tqdm(total=params["trainexamples"]) as pbar:
                while True:
                    _, a, l, predictions, labels = sess.run(
                        [model.train_op, model.accuracy, model.loss, model.predictions, model.labels])

                    train_loss += l
                    train_accuracy += a
                    loss_hist.append(l)
                    acc_hist.append(a)
                    pbar.set_postfix_str((l, a))
                    pbar.update(params["batchSize"])

                    train_predictions.extend(predictions)
                    train_labels.extend(labels)

                    counter += 1
        except tf.errors.OutOfRangeError:
            pass

        loss_hist_epoch.append(train_loss / counter)
        acc_hist_epoch.append(train_accuracy / counter)
        train_f1 = f1_score(train_labels, train_predictions, average=params["f1modus"])
        f1_train_epoch.append(train_f1)

        # Validation
        counter = 0
        sess.run(val_iterator)
        try:
            with tqdm(total=params["trainexamples"]) as pbar:
                while True:
                    a, l, p, labels = sess.run([model.accuracy, model.loss, model.predictions, model.labels])
                    val_loss += l
                    val_accuracy += a
                    val_loss_hist.append(l)
                    val_acc_hist.append(a)
                    pbar.set_postfix_str((l, a))
                    pbar.update(params["batchSize"])

                    val_predictions.extend(p)
                    val_labels.extend(labels)

                    counter += 1
        except tf.errors.OutOfRangeError:
            pass
            # print("\tfinished after", counter, "batches.")

        val_loss_hist_epoch.append(val_loss / counter)
        val_acc_hist_epoch.append(val_accuracy / counter)
        val_f1 = f1_score(val_labels, val_predictions, average=params["f1modus"])
        val_labels_pred_hist.append( [val_labels, val_predictions] )
        f1_val_epoch.append(val_f1)

        print('\n\tEpoch {}: train_loss: {:.4f}, train_acc: {:.4f}, train_micro-f1: {:.4f} || val_loss: {:.4f}, val_acc: {:.4f}, val_micro-f1: {:.4f}'.format(
                epoch + 1, loss_hist_epoch[-1], acc_hist_epoch[-1], train_f1, val_loss_hist_epoch[-1],
                val_acc_hist_epoch[-1], val_f1))
        print(classification_report(val_labels, val_predictions))

        if params["showPlots"] == True:
            confusion.plotCM(val_labels, val_predictions, figsize=(30,20))


        # Epoch finished - update and save results
        trainingTime = time.time() - startTime
        result = {}
        result["architecture"] = params["architecture"]
        result["loss_hist_epoch"] = loss_hist_epoch
        result["acc_hist_epoch"] = acc_hist_epoch
        result["val_loss_hist_epoch"] = val_loss_hist_epoch
        result["val_acc_hist_epoch"] = val_acc_hist_epoch
        result["f1_train_epoch"] = f1_train_epoch
        result["f1_val_epoch"] = f1_val_epoch
        result["loss_hist_epoch"] = loss_hist_epoch
        result["loss_hist_epoch"] = loss_hist_epoch
        result["train_time_seconds"] = trainingTime
        result["train_time_minutes"] = trainingTime / 60
        result["epochs"] = epoch + 1

        if params["savelog"] == True:
            f = open(params["path"] + "result_" + str(params["architecture"]) +".txt", "w")
            for k in result:
                f.write(k + ": " + str(result[k]) + "\n")
            f.close()
            with open(params["path"] + "result_" + str(params["architecture"]) +".pickle","wb") as output_file:
                pickle.dump(result, output_file)

            np.save(params["path"] + "val_label_pred_hist" + str(params["architecture"]) +".npy", val_labels_pred_hist)

            # save plots
            path = params["path"]
            print("saving results to:", path)
            a = params["architecture"]
            plotResults(loss_hist_epoch, "train_loss", val_loss_hist_epoch, "val_loss", str(a) + " loss", path, a,params["showPlots"])
            plotResults(acc_hist_epoch, "acc_train", val_acc_hist_epoch, "acc_val", str(a) + " acc", path, a,params["showPlots"])
            plotResults(f1_train_epoch, "f1_train", f1_val_epoch, "f1_val", str(a) + " f1", path, a,params["showPlots"])

        # save epoch checkpoint
        if params["savelog"] == True and params["checkpoint"] == True:
            saver.save(sess, params["path"] + 'checkpoints/epoch', global_step=epoch+1)

    print(classification_report(val_labels, val_predictions))

    if params["savelog"] == True:
        with open(params["path"] + "classification_report_" + str(params["architecture"]) +".txt","w") as f:
            f.write(classification_report(val_labels, val_predictions))
        with open(params["path"] + "classification_report_" + str(params["architecture"]) +".pickle","wb") as f:
            pickle.dump(classification_report(val_labels, val_predictions), f)

    if params["savelog"] == True:
        confusion.plotCM(val_labels, val_predictions, savepath=params["path"], figsize=(30, 20), show=params["showPlots"])
        plotAll(path,params["showPlots"])
        if params["checkpoint"] == True:
            saver.save(sess, params["path"] + 'checkpoints/final')


    if params["savelog"] == True and params["checkpoint"] == True:
        builder = tf.saved_model.builder.SavedModelBuilder(params["path"] + 'finalsaved/')
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
        builder.save()

    sess.close()

    return result





if __name__== "__main__":

    from tqdm import tqdm as tqdm
    params = {}
    params["trainData"] = "us_balanced"
    params["testData"] = "german"
    params["checkpoint"] = False
    params["savelog"] = True
    params["path"] = "../blobs/test/"
    params["pathToCache"] = "../data/"
    params["architecture"] = [False]
    params["epochs"] = 2
    params["trainexamples"] = 100
    params["batchSize"] = 32

    params["includeReviewHeadline"] = False
    params["filterOtherLanguages"] = False

    result = trainModel(params)
