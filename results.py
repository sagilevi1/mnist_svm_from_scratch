import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def cnf_matrix(examples, labels, predicted, string):
    # calculate confusion matrix
    cm = np.zeros((11, 13))
    for i in range(examples):
        num = np.argmax(labels[i][:])
        # place the additional example
        if predicted[i] != -1:

            cm[int(predicted[i]), num] += 1.0
        else:
            # make a column of miss class
            cm[num, 10] += 1.0

    # calculate the total accuracy
    cm[10, 12] = np.trace(cm) / np.sum(cm)

    # calculate statistics for every digits
    for i in range(10):
        tp = cm[i, i]  # match point on the diagonal
        fn = np.sum(cm[i, :]) - tp  # predict other digits instead
        fp = np.sum(cm[:, i]) - tp  # wrong predict the digit
        tn = np.trace(cm) - tp  # all the diagonal line except for the match point
        if tp != 0:
            r = tp / (tp + fn)  # recall : predict true/all the true
            p = tp / (tp + fp)  # precision  :guess true/all my guess
            a = (tp + tn) / (tp + tn + fp + fn)  # accuracy for the specific digit
        else:  # if the tp=0 prevent division by zero
            r = 0
            p = 0
            a = 0
        cm[10, i] = p
        cm[i, 11] = r
        cm[i, 12] = a

    # build the labels for the plot
    names_x = list(map(str, list(range(0, 10)))) + ["miss", "r", "a"]
    names_y = list(map(str, list(range(0, 10)))) + ["p"]

    # plot the heatmap
    plt.figure(num=string)
    ax = sns.heatmap(cm, square=True, annot=True, cbar=False, xticklabels=names_x, yticklabels=names_y)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_title(f'Confusion Matrix Of The {string}', y=-0.1)
    plt.ylabel('actual')
    plt.xlabel('predicted')

