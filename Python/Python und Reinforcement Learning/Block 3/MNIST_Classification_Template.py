from idlelib import history

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)

NUMBER_OF_CLASSES = 10
EPOCHS = 2
BATCH_SIZE = 256
VALIDATION_SPLIT = .2
NUMBER_OF_HIDDEN_NEURONS = 32


class MNIST_data:
    def __init__(self):
        # Load MNIST dataset
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = keras.datasets.mnist.load_data()
        (self.X_train_preprocessed, self.Y_train_preprocessed, self.X_test_preprocessed, self.Y_test_preprocessed) = self.preprocess_data()
        print(self.X_train)

    def plot_examples(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - generate a tuple of 6 random indices of images within self.X_train
        #   - create a figure with 6 subplots and get all possible indices for subplots (already done)
        #   - use a for-loop to plot the 6 images

        ### Enter your code here ###
        image_idx = tuple(np.random.randint(len(self.X_train), size=6))


        ### End of your code ###

        fig, axs = plt.subplots(2, 3)
        axs_idx = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))

        ### Enter your code here ###
        plot_idx = 0
        for i in image_idx:
            axs[axs_idx[plot_idx]].imshow(self.X_train[i])
            plot_idx += 1


        ### End of your code ###

        plt.show()

    def preprocess_data(self):
        # Input:
        #   - self
        # Return:
        #   - X_train_preprocessed
        #   - Y_train_preprocessed
        #   - X_test_preprocessed
        #   - Y_test_preprocessed
        # Function:
        #   - Reshape each image from 28x28 to 784x1 (input shape: (X, 28, 28), output shape: (X, 784))
        #   - The value of each pixel is in range [0, 255] -> Normalize those values to a range [0, 1]
        #   - Preprocess Y-values:
        #       - Neural net returns probability for each class -> output shape (1, 10)
        #       - Y-value for each picture has to be an array consisting of zeros and one 1, not only a single integer
        #       - e.g. Y-value for digit '3': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        #       - Y input shape: (X, ), output shape: (X, 10)
        #   - Assign the result of the operations to the returned variables

        # X_train_preprocessed = np.ndarray(shape=(len(self.X_train), 784,))
        # for i in range(len(self.X_train)):
        #     X_train_preprocessed[i] = np.reshape(self.X_train[i], 784)
        #
        # Y_train_preprocessed = np.ndarray(shape=(len(self.Y_train), 10,))
        # for i in range(len(self.Y_train)):
        #     Y_train_preprocessed[i, self.Y_train[i]] = 1
        #
        # X_test_preprocessed = np.ndarray(shape=(len(self.X_test), 784,))
        # for i in range(len(self.X_test)):
        #     X_test_preprocessed[i] = np.reshape(self.X_test[i], 784)
        #
        # Y_test_preprocessed = np.ndarray(shape=(len(self.Y_test), 10,))
        # for i in range(len(self.Y_test)):
        #     Y_test_preprocessed[i, self.Y_test[i]] = 1
        X_train_preprocessed = self.X_train.reshape(self.X_train.shape[0], -1)
        X_test_preprocessed = self.X_test.reshape(self.X_test.shape[0], -1)

        Y_train_preprocessed = keras.utils.to_categorical(self.Y_train, NUMBER_OF_CLASSES)
        Y_test_preprocessed = keras.utils.to_categorical(self.Y_test, NUMBER_OF_CLASSES)

        X_train_preprocessed = X_train_preprocessed / 255
        X_test_preprocessed = X_test_preprocessed / 255

        return X_train_preprocessed, Y_train_preprocessed, X_test_preprocessed, Y_test_preprocessed





class TrainSolver:
    def __init__(self):
        self.dataset = MNIST_data()
        self.dataset.plot_examples()
        self.mnist_model = self.create_net()

    def create_net(self):
        # Input:
        #   - self
        # Return:
        #   - model
        # Function:
        #   - Create a Sequential model with 3 Layers:
        #       - Input Layer: pixels from image (hint: use model.add(keras.Input()))
        #       - Hidden Layer: Type: Dense, Activation function: ReLu, number of neurons: NUMBER_OF_HIDDEN_NEURONS
        #       - Output Layer: Type: Dense, Activation function: Softmax, number of neurons: Number of classes
        #   - Compile model:
        #       - Optimizer: Adam
        #       - Loss: Mean-squared-error
        #       - Metrics: Accuracy
        #   - Print model summary and return model

        mnist_model = keras.models.Sequential(name="simple_2_layer_mnist")
        mnist_model.add(keras.Input(shape=(784,)))
        mnist_model.add(keras.layers.Dense(units=NUMBER_OF_HIDDEN_NEURONS, activation='relu'))
        mnist_model.add(keras.layers.Dense(units=NUMBER_OF_HIDDEN_NEURONS, activation='relu'))
        mnist_model.add(keras.layers.Dense(units=NUMBER_OF_CLASSES, activation='softmax'))

        mnist_model.compile(optimizer='Adam',
                            loss='mean_squared_error',
                            metrics=['accuracy'])

        # mnist_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003),
        #                     loss=keras.losses.MeanSquaredError(),
        #                     metrics=keras.metrics.Accuracy())

        mnist_model.summary()

        return mnist_model


    def train(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - train model and store result to variable history:
        #       - Use model.fit()
        #       - Specify batch_size, epochs, verbose=2 and validation_split (use global variables)
        #   - evaluate model with test data
        #       - hint: use model.evaluate()
        #   - plot loss and accuracy (already done)

        ### Enter your code here ###
        global VALIDATION_SPLIT

        history = self.mnist_model.fit(self.dataset.X_train_preprocessed, self.dataset.Y_train_preprocessed,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  validation_split=VALIDATION_SPLIT)

        self.mnist_model.evaluate(self.dataset.X_test_preprocessed, self.dataset.Y_test_preprocessed)
        print(self.dataset.X_test_preprocessed)
        #print(self.mnist_model([self.dataset.X_test_preprocessed[1], self.dataset.X_test_preprocessed[2]]))

        ## End of your code ###

        # Code taken from: https://newbiettn.github.io/2021/04/07/MNIST-with-keras/
        # Convert this object to a dataframe
        df_loss_acc = pd.DataFrame(history.history)
        df_loss_acc.describe()

        # Extract to two separate data frames for loss and accuracy
        df_loss = df_loss_acc[['loss', 'val_loss']]
        df_loss = df_loss.rename(columns={'loss': 'Loss', 'val_loss': 'Validation Loss'})
        df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
        df_acc = df_acc.rename(columns={'accuracy': 'Accuracy', 'val_accuracy': 'Validation Accuracy'})

        # Plot the data frames
        df_loss.plot(title='Training vs Validation Loss', figsize=(10, 6))
        df_acc.plot(title='Training vs Validation Accuracy', figsize=(10, 6))
        plt.show()


if __name__ == "__main__":
    session = TrainSolver()
    session.train()
