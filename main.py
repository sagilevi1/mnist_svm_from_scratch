from load import download_mnist
from preprocess import hog_feature
from preprocess import pca
from preprocess import array2binary
import numpy as np
from svm import SVM
from results import cnf_matrix
import matplotlib.pyplot as plt


def main():
    # choose how much training and testing data set
    train_examples = 2000  # from 1 to 60000
    test_examples = 100  # from 1 to 10000

    # download the full data set
    train_images, train_labels, test_images, test_labels = download_mnist(path=None)

    # preprocess - get the hog and pca features
    train_features = hog_feature(train_images[0:train_examples], orientations=9, pixels_per_cell=(2, 2),
                                 cells_per_block=(1, 1))
    test_features = hog_feature(test_images[0:test_examples], orientations=9, pixels_per_cell=(2, 2),
                                cells_per_block=(1, 1))
    train_features, test_features = pca(train_features, test_features, v_max=0.9)

    # multiclass svm - one to all
    train_predicted = -1 * np.ones(train_examples)
    test_predicted = -1 * np.ones(test_examples)
    for i in range(10):
        # move the labels to binary representation for every digit
        trn_labels = array2binary(train_labels[0:train_examples], num=i)
        clf = SVM()
        # fit on the training set
        clf.fit(train_features, trn_labels)
        # predict the training and the test
        train_predictions = clf.predict(train_features)
        test_predictions = clf.predict(test_features)
        # Check if not predicted in the earlier digits
        for j in range(train_examples):
            if train_predicted[j] == -1 and train_predictions[j] == 1:  # and trn_labels[j]==1:
                train_predicted[j] = i
        for j in range(test_examples):
            if test_predicted[j] == -1 and test_predictions[j] == 1:  # and tst_labels[j]==1:
                test_predicted[j] = i

    # calculate the statistics and visualize
    cnf_matrix(train_examples, train_labels, train_predicted, f"Training Set - {train_examples} Examples")
    cnf_matrix(test_examples, test_labels, test_predicted, f"Testing Set - {test_examples} Examples")
    plt.show()


if __name__ == '__main__':
    main()
