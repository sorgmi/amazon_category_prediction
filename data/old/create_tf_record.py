import csv
import sys

import tensorflow as tf
import os, shutil, gzip

from tqdm import tqdm


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


def downloadData(filename):
    p = tf.keras.utils.get_file(filename,origin="https://s3.amazonaws.com/amazon-reviews-pds/tsv/" + filename,
                                extract=False, cache_dir=".", cache_subdir="cache")
    outDir = p[0:-3]
    if os.path.exists(outDir) == False:
        with open(outDir, 'wb') as f_out, gzip.open(p, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
    else:
        print(outDir, "already exists. Skipping unzipping")


def get_example_object(csv_row, mapping, labelindex, inputindex):
    int_list1 = tf.train.Int64List(value=[mapping[csv_row[labelindex]]])
    str_list1 = tf.train.BytesList(value=[csv_row[inputindex].encode('utf-8')])
    feature_key_value_pair = {
        'label': tf.train.Feature(int64_list=int_list1),
        'input': tf.train.Feature(bytes_list=str_list1),
    }

    # Create Features object with above feature dictionary
    features = tf.train.Features(feature=feature_key_value_pair)

    # Create Example object with features
    example = tf.train.Example(features=features)
    return example


def createData(filename, mapping, labelindex, inputindex):

    tfrecords = filename[:-4]+".tfrecord"

    with open(filename, encoding="utf8") as csv_file:
        csv.field_size_limit(131072*4)
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        line_count = 0
        with tf.python_io.TFRecordWriter(tfrecords) as tfwriter:
            for row in tqdm(csv_reader):
                try:
                    example = get_example_object(row, mapping, labelindex, inputindex)
                    tfwriter.write(example.SerializeToString())
                    line_count += 1
                except IndexError:
                    print("Index error for row:", row)
        print("processed", line_count, "rows.")

    return tfrecords


if __name__== "__main__":
    bs = 2000
    dataset_train = createData("DE", bs, shuffle=False)

