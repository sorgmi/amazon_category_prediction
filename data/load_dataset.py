import tensorflow as tf
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
         'Beauty':                      33
        }
    return d

def index2category(label):
    for key, value in getCategory2IndexDict().items():
        if value == label:
            return key

def numpyToDataset(csvfile, maxrows, includeHeading):
    frame = pandas.read_csv(csvfile, nrows=maxrows) #,sep=sep , usecols=["review_headline", "review_body", "product_category"]
    if includeHeading == True:
        features = frame["review_body"].values + frame["review_headline"].astype(str).values
    else:
        features = frame["review_body"].values
    c2i = getCategory2IndexDict()
    labels = [c2i[x] for x in frame["product_category"].values]
    return tf.data.Dataset.from_tensor_slices((features,labels)), len(c2i)


def getData(countryCode, batchsize, shuffle, pathToCache="./", buffer=None, filterOtherLangs=False,maxrows=None, includeHeading=False):

    if countryCode == "DE" or countryCode == "UK" or countryCode == "US" or countryCode == "TEST":
        import data.amazon_multilingual as amazon_multilingual
        dataset, num_classes = amazon_multilingual.getData(countryCode, pathToCache=pathToCache)
    elif countryCode.startswith("organic"):
        import data.organic_dataset as organic_dataset
        dataset, num_classes = organic_dataset.getData(countryCode, pathToCache=pathToCache)
    elif countryCode == "german":
        dataset, num_classes = numpyToDataset(pathToCache + "cache/amazon_reviews_multilingual_DE_v1_00.tsv.shuffled", maxrows, includeHeading)
    elif countryCode == "german_filtered":
        dataset, num_classes = numpyToDataset(pathToCache + "cache/amazon_reviews_multilingual_DE_v1_00.tsv.shuffled.filtered", maxrows, includeHeading)
    elif countryCode == "us_balanced":
        dataset, num_classes = numpyToDataset(pathToCache + "cache/amazon_reviews_us_balanced.csv.shuffled", maxrows, includeHeading)
    elif countryCode == "us_balanced_validation":
        dataset, num_classes = numpyToDataset(pathToCache + "cache/amazon_reviews_us_balanced_validation.csv.shuffled", None, includeHeading)

    else:
        raise NotImplementedError

    if buffer is None:
        buffer = batchsize * 5
    if shuffle == True:
        dataset = dataset.shuffle(buffer)

    dataset = dataset.batch(batchsize).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, num_classes


if __name__== "__main__":
    bs = 15
    #dataset_train = getData("organic_train_entity", bs, shuffle=False)
    dataset_train, _ = getData("us_balanced", bs, shuffle=False, maxrows=bs)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    train_iterator = iterator.make_initializer(dataset_train)

    with tf.Session() as sess:
        sess.run(train_iterator)

        a = sess.run(iterator.get_next())
        print(a[0], a[1])
