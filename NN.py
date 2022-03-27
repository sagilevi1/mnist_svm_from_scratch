import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import utils as np_utils
from keras.models import sequential, Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
# makes the random numbers constant
np.random.seed(0)


def main():
    # load data
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)

    # Visualize Examples

    num_classes = 10
    # f, ax = plt.subplots(1, num_classes, figsize=(20, 20))
    #
    # for i in range(0, num_classes):
    #     sample = x_train[y_train == i][0]
    #     ax[i].imshow(sample, cmap='gray')
    #     ax[i].set_title("Label: {}".format(i), fontsize=16)

    # change the labels to "one hot"
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    # Normalize Data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape Data - Flatten (28X28) to (1X784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Create Model - Fully Connected Neural Network

    model = Sequential()

    model.add(Dense(units=128, input_shape=(784,), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25)) # prevent over-fitting
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train

    batch_size = 512
    epochs = 10
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    # Evaluate

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

    y_prediction = model.predict(x_test)
    y_prediction_classes = np.argmax(y_prediction, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Single Example
    # random_idx = np.random.choice(len(x_test))
    # x_sample = x_test[random_idx]
    # y_sample_true = y_true[random_idx]
    # y_sample_prediction_class = y_prediction_classes[random_idx]
    #
    # plt.title("Predicted: {}, True: {}".format(y_sample_prediction_class, y_sample_true), fontsize=16)
    # plt.imshow(x_sample.reshape(28, 28), cmap='gray')

    # Confusion Matrix

    confusion_mtx = confusion_matrix(y_true, y_prediction_classes)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix');

    # Investigate Some Errors

    errors = (y_prediction_classes - y_true != 0)
    y_prediction_classes_errors = y_prediction_classes[errors]
    y_prediction_errors = y_prediction[errors]
    y_true_errors = y_true[errors]
    x_test_errors = x_test[errors]

    # Max probability of error
    y_prediction_errors_probability = np.max(y_prediction_errors, axis=1)
    true_probability_errors = np.diagonal(np.take(y_prediction_errors, y_true_errors, axis=1))
    diff_errors_prediction_true = y_prediction_errors_probability - true_probability_errors

    # Get list of indices of sorted differences
    sorted_idx_diff_errors = np.argsort(diff_errors_prediction_true)
    top_idx_diff_errors = sorted_idx_diff_errors[-5:]  # 5 last ones

    # Show Top Errors
    num = len(top_idx_diff_errors)
    f, ax = plt.subplots(1, num, figsize=(30, 30))

    for i in range(0, num):
        idx = top_idx_diff_errors[i]
        sample = x_test_errors[idx].reshape(28, 28)
        y_t = y_true_errors[idx]
        y_p = y_prediction_classes_errors[idx]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Predicted label :{}\nTrue label: {}".format(y_p, y_t), fontsize=22)

    plt.show()


if __name__ == '__main__':
    main()


