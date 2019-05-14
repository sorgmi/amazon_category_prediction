import tensorflow as tf
import os

import shutil, gzip
import pandas as pd

import numpy as np


def getCategory2IndexDict():
    d = {'Video DVD':                   0,
         'Music':                       1,
         'Books':                       2,
         'Mobile_Apps':                 3,
         'Digital_Video_Download':      4,
         'Digital_Music_Purchase':      5,
         'Toys':                        6,
         'Digital_Ebook_Purchase':      7,
         'PC':                          8,
         'Camera':                      9,
         'Wireless':                    10,
         'Electronics':                 11,
         'Video':                       12,
         'Sports':                      13,
         'Video Games':                 14,
         'Watches':                     15,
         'Shoes':                       16,
         'Home':                        17,
         'Musical Instruments':         18,
         'Baby':                        19,
         'Home Improvement':            20,
         'Home Entertainment':          21,
         'Office Products':             22,
         'Personal_Care_Appliances':    23,
         'Automotive':                  24,
         'Lawn and Garden':             25,
         'Luggage':                     26,
         'Kitchen':                     27,
         'Furniture':                   28,
         'Health & Personal Care':      29,
         'Software':                    30,
         'Grocery':                     31,
         'Pet Products':                32,
         'Beauty':                      33,

         #UK
         'Apparel':                     34,

         #US
         'Tools':                       35,
         'Outdoors':                    36,
         'Mobile_Electronics':          37,
         '2012-12-22':                  38,
        }
    return d

def generateFilenames(countryCode):
    x1 = "cache/amazon_multilingual_"+countryCode+"_category.npy"
    x2 = "cache/amazon_multilingual_"+countryCode+"_labels.npy"
    return (x1,x2)

def getAmazonDataFrame(filename):
    p = tf.keras.utils.get_file(filename,origin="https://s3.amazonaws.com/amazon-reviews-pds/tsv/" + filename,
                                extract=False, cache_dir=".", cache_subdir="cache")
    outDir = p[0:-3]
    if os.path.exists(outDir) == False:
        with open(outDir, 'wb') as f_out, gzip.open(p, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
    else:
        print(outDir, "already exists. Skipping unzipping")

    frame = pd.read_csv(p[0:-3], sep='\t', error_bad_lines=False)
    return frame


def dataToFile(data, countryCode):
    f1, f2 = generateFilenames(countryCode)
    d = getCategory2IndexDict()

    # TODO: also use headline (review_headline) ?
    x = data.review_body.values
    y = [d[x] for x in data.product_category.values]

    np.save(f1, x)
    np.save(f2, y)


def getData(countryCode, shuffle, buffer=None, batchsize=128):  # DE, UK, US

    f1, f2 = generateFilenames(countryCode)
    if os.path.exists(f1) and os.path.exists(f2):
        print(f1, "and", f2, " exist. Using saved npy files...")
    else:
        print(f1, "or", f2, " missing. Creating it...")
        frame = getAmazonDataFrame("amazon_reviews_multilingual_" + countryCode + "_v1_00.tsv.gz")
        frame.dropna(subset=['review_body'], inplace=True)
        dataToFile(frame, countryCode)
        del frame

    x = np.load(f1, allow_pickle=True)
    y = np.load(f2)

    print(x.shape, y.shape)
    print(x[:10])
    print(y[:10])
    #print(type(y[0]))
    return buildDataset(x, y, shuffle, buffer, batchsize)



def buildDataset(x, y, shuffle, buffer, batchsize):
    length = x.shape[0]

    if buffer is None:
        buffer = length

    dataset = tf.data.Dataset.from_tensor_slices( (x,y) )
    if shuffle == True:
        # tf.data.experimental.AUTOTUNE
        dataset = dataset.shuffle(buffer).batch(batchsize).prefetch(1)
    else:
        print("dataset is not shuffled and prefetched")

    iter = dataset.make_one_shot_iterator()

    return iter, length



if __name__== "__main__":

    iter, length = getData("DE", shuffle=True, buffer=3, batchsize=3)
    #iter = buildDataset(data_de)
    #quit()

    print("sample one example")

    with tf.Session() as sess:
        output = sess.run(iter.get_next())
        print("out:", output)