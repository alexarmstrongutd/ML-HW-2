#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error



class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        # No categories and mainly interested in dropping null values and removing duplicate entries
        # to clean the data
        self.raw_input.dropna(inplace=True)
        self.raw_input.drop_duplicates(inplace=True)
        self.raw_input['Class'] = pd.factorize( self.raw_input['Class'])[0] + 1
        self.processed_data = self.raw_input

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics
        for activation in activations:
            for learn in learning_rate:
                for iteration in max_iterations:
                    for hidden_layer in num_hidden_layers:
                        regression = MLPRegressor(hidden_layer_sizes=tuple([100]* hidden_layer), activation=activation,learning_rate_init=learn,max_iter=iteration).fit(X_train,y_train)
                        train_accuracy = accuracy_score(y_train, regression.predict(X_train))
                        test_accuracy = accuracy_score(y_test, regression.predict(X_test))
                        train_error = mean_squared_error(y_train, regression.predict(X_train))
                        test_error = mean_squared_error(y_test, regression.predict(X_test))
                        print(f'Activation_Function = {activation}, Learning_Rate = {learn}, Iterations = {iteration}, Num_hidden_layers = {hidden_layer}')
                        print(f'Train_Accuracy = {train_accuracy}, Train_Error = {train_error}')
                        print(f'Test_Accuracy = {test_accuracy}, Test_Error = {test_error}')

        exit()



        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/alexarmstrongutd/ML-HW-2/master/Concrete_Data.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
