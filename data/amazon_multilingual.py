import csv
import tensorflow as tf
import os, shutil, gzip
import pandas


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


def downloadData(filename, pathToCache):
    p = tf.keras.utils.get_file(filename,origin="https://s3.amazonaws.com/amazon-reviews-pds/tsv/" + filename,
                                extract=False, cache_dir=pathToCache, cache_subdir="cache")
    outDir = p[0:-3]
    if os.path.exists(outDir) == False:
        with open(outDir, 'wb') as f_out, gzip.open(p, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
    else:
        pass
        #print(outDir, "already exists. Skipping unzipping")

    # shuffle Data
    csv_path = outDir + ".shuffled.csv"
    if os.path.exists(csv_path) == False:
        print("creating", csv_path, "...")
        frame = pandas.read_csv(outDir, error_bad_lines=False, delimiter="\t").sample(frac=1)
        frame.to_csv(csv_path)
    else:
        print(csv_path, "already exists. Using cached data")


    return csv_path


def csvGenerator(filename, labelindex, inputindex):
    mapping = getCategory2IndexDict()
    with open(filename, encoding="utf8") as csv_file:
        csv.field_size_limit(131072*4)
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_len = len(next(csv_reader))
        for row in csv_reader:
            if len(row) != line_len:
                #print("skipping row. different length.", row)
                continue

            yield row[inputindex], mapping[row[labelindex]]


def getData(countryCode, pathToCache):

    if countryCode == "TEST":
        filename = pathToCache + "cache/amazon_reviews_multilingual_TEST_v1_00.tsv"
        frame = pandas.read_csv(filename, error_bad_lines=False, delimiter="\t")
        filename = filename + ".shuffled.csv"
        frame.to_csv(filename)
    else:
        filename = "amazon_reviews_multilingual_" + countryCode + "_v1_00.tsv.gz"
        filename = downloadData(filename, pathToCache)

    dataset = tf.data.Dataset.from_generator(csvGenerator,
                                             (tf.string,
                                              tf.int32),
                                             args=(filename,7,14) )


    return dataset, len(getCategory2IndexDict())
