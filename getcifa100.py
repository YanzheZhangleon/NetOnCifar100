
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getTrainData():
    file = 'F:\cifar-100-python\\train'
    dic = unpickle(file)
    return dic

def getTestData():
    file = 'F:\cifar-100-python\\test'
    dic = unpickle(file)
    return dic

def getLabelData():
    file = 'F:\cifar-100-python\meta'
    dic = unpickle(file)
    return dic[b'fine_label_names'],dic[b'coarse_label_names']

def showSomeImage():
    fineName,coarseName = getLabelData()
    imageData = getTestData()
    fileName = imageData[b'filenames']
    batch_label = imageData[b'batch_label']
    fine_labels = imageData[b'fine_labels']
    coarse_labels = imageData[b'coarse_labels']
    data = imageData[b'data']
    for j in range(10):
        i = np.random.randint(0,1000)
        ima = np.reshape(data[i], [3, 32, 32])
        ima = np.transpose(ima, [1, 2, 0])
        plt.subplot(2,5,j+1)
        plt.imshow(ima)
        plt.title(fineName[fine_labels[i]])

    plt.show()

def getData():
    train = getTrainData()
    test = getTestData()
    train_labels = train[b'fine_labels']
    train_data = train[b'data']
    train_data = np.reshape(train_data, [-1, 3, 32, 32])
    train_data = np.transpose(train_data, [0, 2, 3, 1])
    test_labels = test[b'fine_labels']
    test_data = test[b'data']
    test_data = np.reshape(test_data, [-1, 3, 32, 32])
    test_data = np.transpose(test_data, [0, 2, 3, 1])
    return train_data,train_labels,test_data,test_labels





