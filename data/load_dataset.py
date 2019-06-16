import tensorflow as tf

def getData(countryCode, batchsize, shuffle, pathToCache="./", buffer=None, filterOtherLangs=False):

    if countryCode == "DE" or countryCode == "UK" or countryCode == "US" or countryCode == "TEST":
        import data.amazon_multilingual as amazon_multilingual
        dataset, num_classes = amazon_multilingual.getData(countryCode, pathToCache=pathToCache)
    elif countryCode.startswith("organic"):
        import data.organic_dataset as organic_dataset
        dataset, num_classes = organic_dataset.getData(countryCode, pathToCache=pathToCache)
    else:
        raise NotImplementedError

    if buffer is None:
        buffer = batchsize * 5
    if shuffle == True:
        dataset = dataset.shuffle(buffer)

    #dataset=dataset.map(preprocess)
    dataset = dataset.batch(batchsize).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, num_classes


if __name__== "__main__":
    bs = 5
    dataset_train = getData("organic_train_entity", bs, shuffle=False)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    train_iterator = iterator.make_initializer(dataset_train)

    with tf.Session() as sess:
        sess.run(train_iterator)

        a = sess.run(iterator.get_next())
        print(a[0], a[1])
