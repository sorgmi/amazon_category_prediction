import tensorflow as tf
import os, shutil, gzip


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


d = getCategory2IndexDict()
keys = list(d.keys())
values = [d[k] for k in keys]
table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int32), -1)

def preprocess(label, text):
    #print (text.shape)
    return text, table.lookup(label)



def getData(countryCode, batchsize, shuffle, buffer=None, filterOtherLangs=False):

    filename = "amazon_reviews_multilingual_" + countryCode + "_v1_00.tsv.gz"
    downloadData(filename)
    filename = "cache/" + filename[:-3]

    if buffer is None:
        buffer = batchsize*5

    dataset = tf.data.experimental.CsvDataset(
        filename,
        [tf.string,  tf.string,],
        select_cols=[6,13],
        header=True,
        field_delim="\t",
        use_quote_delim=False
    ).map(preprocess) #lambda x: fun(x, my_arg)

    #dataset = dataset.filter(lambda x,y: tf.size(y)>0)

    if shuffle == True:
        dataset = dataset.shuffle(buffer)

    dataset = dataset.batch(batchsize).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset




if __name__== "__main__":
    bs = 2000
    dataset_train = getData("UK", bs, shuffle=False)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    train_iterator = iterator.make_initializer(dataset_train)

    with tf.Session() as sess:
        sess.run(train_iterator)
        sess.run([tf.tables_initializer()])
        a = sess.run(iterator.get_next())
        print(a[0], a[1])

    while True:
        with tf.Session() as sess:
            sess.run(train_iterator)
            sess.run([tf.tables_initializer()])
            a = sess.run(iterator.get_next())
            #print(a[0], a[1])
