from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def array2binary(yarray, num):
    #convert the label to binary classification for specific digit
    y=[]
    for i in yarray:
        if i[num] == 1:
            y.append(1)
        else:
            y.append(-1)
    return y

def HOG(images, orientations, pixels_per_cell, cells_per_block):
    features=[]
    for img in images:
        imgresized = img.reshape(28,28)
        fd,hog_image = hog(imgresized, orientations, pixels_per_cell, cells_per_block, visualize=True)
        features.append(fd)
    #normalaized the feature
    scaler = StandardScaler()
    train = scaler.fit_transform(features)
    # Make an instance of the Model
    return train

def pca(train,test, maxvar):
    pca = PCA(maxvar)
    pca.fit(train)
    train = pca.transform(train)
    #transform the test but fit on the train
    test= pca.transform(test)
    return train, test



