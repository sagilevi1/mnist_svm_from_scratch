from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def array2binary(arr, num):
    # convert the label to binary classification for specific digit
    y = []
    for i in arr:
        if i[num] == 1:
            y.append(1)
        else:
            y.append(-1)
    return y


def hog_feature(images, orientations, pixels_per_cell, cells_per_block):
    features = []
    for img in images:
        resized = img.reshape(28, 28)
        fd, hog_image = hog(resized, orientations, pixels_per_cell, cells_per_block, visualize=True)
        features.append(fd)
    # normalized the feature
    scale = StandardScaler()
    train = scale.fit_transform(features)
    # Make an instance of the Model
    return train


def pca(train, test, v_max):
    principal = PCA(v_max)
    principal.fit(train)
    train = principal.transform(train)
    # transform the test but fit on the train
    test = principal.transform(test)
    return train, test



