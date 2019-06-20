import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

from data import load_dataset
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import glob, time, datetime, os, pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


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
        # y_hot = tf.one_hot(data_Y, depth=self.n_class)
        self.logits = self.forward(data_X)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data_Y, logits=self.logits))
        self.train_op = self.params["optimizer"].minimize(self.loss)

        self.predictions = tf.argmax(self.logits, 1)
        self.labels = data_Y
        # self.acc, self.acc_op = tf.metrics.accuracy(labels=data_Y, predictions=self.predictions)

        # a = tf.cast(self.predictions, tf.float64)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.cast(data_Y, tf.int64)), tf.float32))

    def forward(self, X):
        output = self.xling(X)

        for x in self.architecture:
            if x == "bn":
                output = tf.layers.batch_normalization(output, training=True)
            elif x == "relu" or x == "r":
                output = tf.nn.relu(output)
            elif x == "dropout" or x == "d":
                output = tf.layers.dropout(output)
            else:
                output = tf.layers.dense(output, x)

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

    params.update(p)  # overwrite default parameter with passed parameter
    params["learning_rate"] = params["optimizer"]._lr

    if params["notebook"] == True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm


    tf.reset_default_graph()
    dataset_train, num_classes = load_dataset.getData(params["trainData"], shuffle=True, batchsize=params["batchSize"], pathToCache=params["pathToCache"])
    dataset_val, num_classes2 = load_dataset.getData(params["testData"], shuffle=False, batchsize=params["batchSize"], pathToCache=params["pathToCache"])

    if num_classes != num_classes2:
        raise Exception("number of classes do not match between train and test set")

    print("classes:", num_classes)
    params["num_classes"] = num_classes

    if params["savelog"] == True:
        '''
        if params["path"] is None:
            path = '/content/gdrive/My Drive/nlp/blobs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
            os.mkdir(path)
            params["path"] = path
        else:
            path = params["path"]
        '''

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
    train_predictions, train_labels, val_predictions, val_labels = [], [], [], []

    saver = tf.train.Saver()
    startTime = time.time()
    for epoch in tqdm(range(params["epochs"])):
        # print('\nEpoch: {}'.format(epoch + 1))
        train_loss, train_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0
        counter = 0

        sess.run(train_iterator)

        try:
            with tqdm(total=params["trainexamples"]) as pbar:
                while True:
                    _, a, l, predictions, labels = sess.run(
                        [model.train_op, model.accuracy, model.loss, model.predictions, model.labels])
                    # print(a,l)

                    '''
                    if l > 0 and l < 15:
                        pass
                    else:
                        print(l)
                        print(counter)
                        print(sess.run(model.data_X))
                        # print(tf.print(model.data_Y))
                    '''

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
            # print("\tfinished after", counter, "batches.")

        loss_hist_epoch.append(train_loss / counter)
        acc_hist_epoch.append(train_accuracy / counter)
        train_f1 = f1_score(train_labels, train_predictions, average=params["f1modus"])
        f1_train_epoch.append(train_f1)
        # print('\nEpoch: {}'.format(epoch + 1))

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
        f1_val_epoch.append(val_f1)
        print(
            '\n\tEpoch {}: train_loss: {:.4f}, train_acc: {:.4f}, train_micro-f1: {:.4f} || val_loss: {:.4f}, val_acc: {:.4f}, val_micro-f1: {:.4f}'.format(
                epoch + 1, loss_hist_epoch[-1], acc_hist_epoch[-1], train_f1, val_loss_hist_epoch[-1],
                val_acc_hist_epoch[-1], val_f1))


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

    if params["savelog"] == True:
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
    params["trainData"] = "organic_train_entity"
    params["testData"] = "organic_test_entity"
    params["checkpoint"] = False
    params["path"] = "../blobs/test/"
    params["pathToCache"] = "../data/"
    params["architecture"] = [False]
    params["epochs"] = 2
    params["trainexamples"] = 15
    params["batchSize"] = 5
    result = trainModel(params)