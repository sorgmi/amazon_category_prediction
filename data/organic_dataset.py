import csv
import tensorflow as tf
import os, shutil, gzip
import pandas


def getAttributeMapping():
    return {
	'g': 0,
	'p': 1,
	't': 2,
	'q': 3,
	's': 4,
	'h': 5,
	'c': 6,
	'll': 7,
	'or': 8,
	'l': 9,
	'e': 10,
	'av': 11,
	'a': 12,
	'pp': 13
    }

def getEntityMapping():
    return {
	'g': 0,
	'p': 1,
	'f': 2,
	'c': 3,

	'cg': 4,
	'cp': 5,
	'cf': 6,
	'cc': 7,

	'gg': 8
    }

def getRelevanceMapping():
    return {0:0, 9:1}

def downloadData(pathToCache):
    p = tf.keras.utils.get_file("processed.zip",
                                origin="https://syncandshare.lrz.de/dl/fiNDkYy3SNfH6M62mpfPV92c/processed.zip",
                                extract=True, cache_dir=pathToCache, cache_subdir="cache")
    return p


def csvGenerator(filename, countryCode):
    frame = pandas.read_csv(filename.decode("utf-8"), sep='|')
    # print(frame.head())

    if countryCode.lower().endswith(b"attribute"):
        target = "Attribute"
        frame = frame[frame["Attribute"].notnull()]
        mapping = getAttributeMapping()
    elif countryCode.lower().endswith(b"entity"):
        target = "Entity"
        frame = frame[frame["Entity"].notnull()]
        mapping = getEntityMapping()
    elif countryCode.lower().endswith(b"relevance"):
        target = "Domain_Relevance"
        frame = frame[frame["Domain_Relevance"].notnull()]
        mapping = getRelevanceMapping()
    else:
        raise NotImplementedError

    for row in frame.itertuples():
        sentence = getattr(row, "Sentence")
        value = getattr(row, target)
        label = mapping[value]
        yield sentence, label


def getData(countryCode, pathToCache):

    downloadData(pathToCache)

    if countryCode.startswith("organic_train"):
        filename = pathToCache + "cache/train_test_validation V0.2/train/dataframe.csv"
    elif countryCode.startswith("organic_test"):
        filename = pathToCache + "cache/train_test_validation V0.2/test/dataframe.csv"
    elif countryCode.startswith("organic_val"):
        filename = pathToCache + "cache/train_test_validation V0.2/validation/dataframe.csv"
    else:
        raise NotImplementedError


    dataset = tf.data.Dataset.from_generator(csvGenerator,
                                             (tf.string,tf.int32),
                                             args=(filename, countryCode) )

    if countryCode.lower().endswith("attribute"):
        mapping = getAttributeMapping()
    elif countryCode.lower().endswith("entity"):
        mapping = getEntityMapping()
    elif countryCode.lower().endswith("relevance"):
        mapping = getRelevanceMapping()
    else:
        raise NotImplementedError

    return dataset, len(mapping)
