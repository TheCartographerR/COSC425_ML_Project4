# Ryan Pauly
# CS 425 - Machine Learning
# Project 4
#
# Neural Networks Project Description:  Apply multilayer artificial neural nets (ANNs) to spam email classification
#                                           More specifically, use back-propagation
#
#
#
#######################################################################################################################


from random import random
import numpy as np
import pandas as pd
import math
import random
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


#   The activation function defines the output of the node and for this project is flexible to a degree.
#   Can accept linear, sigmoid, and a softmax output type to be specific by the user. (In this case though it is just
#   to find which performs the best on the given data.)
def activation_function(rowdata, weight, bias, function_type):

    node_activation = weight[-1]
    for i in range(len(weight) - 1):
        node_activation = node_activation + (weight[i] * rowdata[i])

    if function_type == "linear":
        # print("linear")
        return node_activation

    if function_type == "sigmoid":
        # print("sigmoid")
        return 1 / (1 + np.exp(-node_activation))

    if function_type == "softmax":
        # print("softmax")
        # node_activation = rowdata * weight
        return np.exp(node_activation) / np.sum(np.exp(node_activation))


#   Find derivatives of the activation function for back-propagation
def find_slope(outbound, function_type):
    #   Compute the derivative with respect to the function type
    derivative = 0

    if function_type == "linear":
        # print("derivative of linear")
        return 1

    if function_type == "sigmoid":
        # print("derivative of sigmoid")
        derivative = outbound * (1 - outbound)
        return derivative

    if function_type == "softmax":
        # print("derivative of softmax")
        temp = np.diag(outbound)
        temp_matrix = np.tile(temp.reshape(temp.shape[0], 1), temp.shape[0])
        derivative = temp - (temp_matrix * np.transpose(temp_matrix))
        return derivative


def softmax_deriv(outputs):
    # print("outputs = ", outputs)
    temp = np.diag(outputs)
    # temp_matrix = np.tile(temp.reshape(temp.shape[0], 1), temp.shape[0])
    # derivative = temp - (temp_matrix * np.transpose(temp_matrix))

    for i in range(len(temp)):

        for j in range(len(temp)):

            if i != j:
                temp[i][j] = -outputs[i] * outputs[j]
            else:
                temp[i][j] = outputs[i] * (1.0 - outputs[i])

    # print("temp = ", temp)
    return [temp[0][0], temp[1][1]]


def build_neural_network(input_dimension, hidden_dimension, output_dimension):
    myNeuralNetwork = []

    hidden_layer = []
    second_hidden_layer = []
    output_layer = []

    bias = 1

    #   Hidden Layer Weights:
    for i in range(hidden_dimension):

        hidden_layer_neuron = {
            "Connection_Weight": [],
            "Bias": random.uniform(0, 0.01)
        }

        #   Each weight is a connection to this neuron from the input layer
        for j in range(input_dimension):
            #   Initialize the weight to a value between -0.01 and 0.01 (we want a very small number close to 0)
            #   for the initial weight (page 289 in psuedo code)
            hidden_layer_neuron["Connection_Weight"].append(random.uniform(0, 0.01))

        #   Keep this particular neuron and all its "connections" or weights.
        hidden_layer.append(hidden_layer_neuron)

    for i in range(output_dimension):

        output_layer_neuron = {
            "Connection_Weight": [],
            "Bias": random.uniform(0, 0.01)

        }

        #   Again, each weight is a connection to this neuron except this time from the hidden layer:
        for j in range(hidden_dimension):
            #   Like before, initialize to a value between -0.01 and 0.01
            output_layer_neuron["Connection_Weight"].append(random.uniform(0, 0.01))

        #   Once finished, append into the output_layer list to keep all of its "connections" or weights
        output_layer.append(output_layer_neuron)

    #   Now put both the hidden and output layer into myNeuralNetwork list
    myNeuralNetwork.append(hidden_layer)
    # myNeuralNetwork.append(second_hidden_layer)
    myNeuralNetwork.append(output_layer)

    return myNeuralNetwork


#   Forward propagation calculates the output for each neuron in each layer of the neural network
def forward(neural_network, rowdata, function_type):
    myInputs = rowdata

    inFirstLayer = True
    softmax_outputs = []

    for network_layer in neural_network:

        nextInputs = []

        #   For each node in the current network_layer (i.e. a node in say.. the hidden layer)
        for node in network_layer:
            #   Use the activation function to find the output of the neuron node

            #   First check to see which layer we're in to limit softmax function
            #   In the first hidden layer
            if inFirstLayer:

                if function_type == "softmax":

                    #   Force the sigmoid activation function instead of the softmax since softmax is only for the
                    #   output layer in the neural network
                    node["Output"] = activation_function(myInputs, node["Connection_Weight"], node["Bias"], "sigmoid")

                    softmax_outputs.append(node["Output"])

                    nextInputs.append(node["Output"])

                else:
                    node["Output"] = activation_function(myInputs, node["Connection_Weight"], node["Bias"],
                                                         function_type)
                    nextInputs.append(node["Output"])

            else:
                node["Output"] = activation_function(myInputs, node["Connection_Weight"], node["Bias"], function_type)
                nextInputs.append(node["Output"])
            #   Append the result of the activation function into a temp variable
            # temp.append(node["Output"])

        inFirstLayer = False
        myInputs = nextInputs  # update myInputs with the updated list for the following layer

    if function_type == "softmax":
        myInputs = []

        mySoftmax = np.exp(softmax_outputs) / np.sum(np.exp(softmax_outputs))

        for node in neural_network[len(neural_network) - 1]:

            for i in range(len(mySoftmax)):
                node["Output"] = mySoftmax[i]
                myInputs.append(mySoftmax[i])

    output_layer_outputs = myInputs
    return output_layer_outputs


#   Backward propagation trains and optimizes weight values for each connection in the neural network
#   by using the chain rule, iterating backwards one layer at a time
def backward(neural_network, actual_classifier_output, function_type):
    inTheOutputLayer = False

    for network_layer in reversed(range(len(neural_network))):

        error_check = []
        check_layer = neural_network[network_layer]

        if network_layer != len(neural_network) - 1:

            for i in range(len(check_layer)):
                error = 0
                for node in neural_network[network_layer + 1]:
                    # error = error + (node["Connection_Weight"][i] * node["BackPropError"])
                    error = error + (node["Connection_Weight"][i] * node["BackPropError"])  # * node["Bias"]
                error_check.append(error)
        else:
            inTheOutputLayer = True

            for i in range(len(check_layer)):
                node = check_layer[i]
                #   Error Check will store the difference of the actual classifier less the Output
                #   Which will ultimately show how close the "approximation" is

                # print("i = ", i)
                # print("actual[i] = ", actual_classifier_output[i])
                # print("node[Output] = ", node["Output"])
                # print("Error = ", actual_classifier_output[i] - node["Output"])

                error_check.append(actual_classifier_output[i] - node["Output"])

        #   Calculate the back propagation error
        for i in range(len(check_layer)):
            node = check_layer[i]

            #   The Softmax function is only used for the output layer
            if inTheOutputLayer:

                if function_type == "softmax":

                    x = []

                    for node in check_layer:
                        x.append(node["Output"])

                    shift = x - np.max(x)
                    temp = np.exp(shift)
                    mySoftmax = temp / np.sum(temp)

                    # mySoftmax = np.exp(x) / np.sum(np.exp(x))

                    mySoftmaxPartial = softmax_deriv(mySoftmax)

                    for node in check_layer:

                        for k in range(len(check_layer)):
                            node["BackPropError"] = error_check[k] * mySoftmaxPartial[k]

                else:
                    #   The function is sigmoid or linear
                    node["BackPropError"] = error_check[i] * find_slope(node["Output"], function_type)

                #   Since we've just entered the output layer we update the flag to reflect that
                #   We've been in here.
                inTheOutputLayer = False

            else:
                if function_type == "softmax":
                    #   If we're using softmax and this is not the output layer, default to using sigmoid
                    node["BackPropError"] = error_check[i] * find_slope(node["Output"], "sigmoid")

                else:
                    #   Otherwise the function type is either sigmoid or linear
                    node["BackPropError"] = error_check[i] * find_slope(node["Output"], function_type)

        inTheOutputLayer = False


# Update network weights with error
def update(network, rowdata, user_defined_learning_rate):
    for i in range(len(network)):

        #   Get all the row features except for the classifier
        inputData = rowdata[:len(rowdata) - 1]

        # print("InputData = ", inputData)
        #   If we're in the output layer
        if i == 1:
            #   If we're in the output layer we instead grab the final outputs of the nodes
            inputData = [node["Output"] for node in network[i - 1]]
            # print("inputData = ", inputData)

        for node in network[i]:

            for j in range(len(inputData)):
                node["Connection_Weight"][j] += user_defined_learning_rate * node["BackPropError"] * inputData[j]
                #node["Bias"] += user_defined_learning_rate * node["BackPropError"] * inputData[j]

            node["Connection_Weight"][-1] += user_defined_learning_rate * node["BackPropError"]
            #node["Bias"] += user_defined_learning_rate * node["BackPropError"]


def find_optimal_weights(neural_network, training_data, user_defined_epoch, user_defined_learning_rate,
                         user_defined_error_threshold, output_dim, function_type):
    errors_y_axis = []
    epoch_x_axis = []

    final_error = 0
    for i in range(user_defined_epoch):

        error_counter = 0
        break_epoch_count = 0

        for rowdata in training_data:

            myOutput = forward(neural_network, rowdata, function_type)
            actualOutput = [0] * output_dim
            #   Depending on the classifier value present in the actual rowdata, will determine where the 1 goes
            #   where a 1 says yes the value is 0 or that the classifier is a 1
            actualOutput[int(rowdata[-1])] = 1

            for j in range(len(actualOutput)):
                error_counter = error_counter + (actualOutput[j] - myOutput[j]) ** 2

            #   Backward Propagation:
            backward(neural_network, actualOutput, function_type)

            #   Update the weights with respect to the backward propagation
            #   weights are updated with respect to the error we found, where a small error is desirable

            update(neural_network, rowdata, user_defined_learning_rate)

            #error_counter = np.divide(error_counter, len(actualOutput))
            error_counter = error_counter / len(actualOutput)

        if i % 10 == 0:
            # print every 10,000 steps
            print("Epoch = " + str(i) + "\t Error = " + str(error_counter))

        if error_counter <= user_defined_error_threshold:
            print("Error is less than user_defined_error_threshold!")
            print("Epoch = " + str(i) + "\t Error = " + str(error_counter) + " at threshold = "
                  + str(user_defined_error_threshold))

            errors_y_axis.append(error_counter)
            epoch_x_axis.append(i)

            break

        errors_y_axis.append(error_counter)
        epoch_x_axis.append(i)

        np.random.shuffle(training_data)
        final_error = error_counter

    if (error_counter > user_defined_error_threshold):
        print("Epoch Limit Reached:")
        print("Epoch = " + str(user_defined_epoch) + "\t Error = " + str(error_counter))


    error_plot = {
        "x": epoch_x_axis,
        "y": errors_y_axis
    }

    plt.plot(epoch_x_axis, errors_y_axis)
    plt.title("Error versus Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

    return final_error


def neural_net_analysis(neural_network, testing_data, function_type):
    actual = []
    myPredictions = []

    for row in testing_data:
        results = forward(neural_network, row, function_type)
        prediction = results.index(max(results))

        myPredictions.append(prediction)
        # print("row[-1] = ", row[-1])
        actual.append(row[-1])

        # print("Actual = " + str(row[-1]) + "\t Prediction = " + str(prediction))

    TN = []
    TP = []
    FN = []
    FP = []

    #   0 == Benign
    #   1 == Malignant

    for i in range(len(myPredictions)):
        if myPredictions[i] == 0:
            if actual[i] == 0:
                TN.append(1)  # True Negative
            else:
                FN.append(1)  # False Negative
        else:
            if actual[i] == 0:
                FP.append(1)  # False Positive
            else:
                TP.append(1)  # True Positive

    num_TN = len(TN)
    num_TP = len(TP)
    num_FN = len(FN)
    num_FP = len(FP)

    accuracy = ((num_TN + num_TP) / (num_TN + num_TP + num_FN + num_FP)) * 100

    if (num_TP + num_FN) == 0:
        TPR = 0
    else:
        TPR = num_TP / (num_TP + num_FN)

    if (num_TP + num_FP) == 0:
        PPV = 0
    else:
        PPV = num_TP / (num_TP + num_FP)

    if (num_TN + num_FP) == 0:
        TNR = 0
    else:
        TNR = num_TN / (num_TN + num_FP)

    if (PPV + TPR) == 0:
        F_1_Score = 0
    else:
        F_1_Score = 2 * PPV * TPR / (PPV + TPR)


    print("\nAccuracy = ", accuracy)
    print("\tTPR = ", TPR)
    print("\tPPV = ", PPV)
    print("\tTNR = ", TNR)
    print("\tF_1_Score = ", F_1_Score)

    # Create Confusion Matrix

    table = PrettyTable()
    table.field_names = ["...", "Predicted Class", 'Benign', 'Malignant']
    table.add_row((["True", "Benign", num_TN, num_FP]))
    table.add_row([" Class", "Malignant", num_FN, num_TP])
    print(table)

    return accuracy


def determine_best_setup(training, testing, validation):
    num_input_neurons = len(training[0]) - 1
    num_output_neurons = 2
    num_hidden_layer_neurons = int((num_input_neurons + num_output_neurons) / 3)

    output_function_types = ["linear", "sigmoid", "softmax"]

    errorThreshold = 0.20
    epoch = 200
    learning_rate = 0.07

    # myNeuralNet = build_neural_network(num_input_neurons, num_hidden_layer_neurons, num_output_neurons)
    myNeuralNet = build_neural_network(num_input_neurons, 3, num_output_neurons)

    find_optimal_weights(myNeuralNet, training, epoch, learning_rate, errorThreshold, num_output_neurons, "sigmoid")
    print("\n\nTRAINING\n")
    neural_net_analysis(myNeuralNet, validation, "sigmoid")
    # find_optimal_weights(myNeuralNet, training, epoch, learning_rate, errorThreshold, num_output_neurons, "softmax")
    # find_optimal_weights(myNeuralNet, training, epoch, learning_rate, errorThreshold, num_output_neurons, "linear")

    print("\nCROSS-VALIDATION:")
    print("\terrorThreshold = " + str(errorThreshold) + " (Error)")
    print("\tEpoch Limit = ", epoch)
    print("\tLearning Rate = ", learning_rate)

    maxAccuracy = 0
    bestFunctionType = "Oops"
    for i in range(len(output_function_types)):
        # temp = find_optimal_weights(myNeuralNet, training, epoch, learning_rate, errorThreshold, num_output_neurons,
        #                             output_function_types[i])
        # temp = find_optimal_weights(myNeuralNet, training, epoch, learning_rate, errorThreshold, num_output_neurons,
        #                                                       output_function_types[i])

        print("\n\tActivation Function Type = ", output_function_types[i])

        accuracy = neural_net_analysis(myNeuralNet, validation, output_function_types[i])

        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestFunctionType = output_function_types[i]

    print("\n\nThe best function type is ", bestFunctionType)
    print("Which returned an accuracy of = ", maxAccuracy)

    print("\n Test with best activation function: ", bestFunctionType)

    accuracy = neural_net_analysis(myNeuralNet, testing, bestFunctionType)


if __name__ == "__main__":
    fileName = "spambase.data"

    #   Read in the CSV file
    myData = pd.read_csv(fileName, header=None)

    #   Search for any Nan values and replace them with "", or a blank spot without a space.
    myData = myData.replace(to_replace={"\?", "\ ", " ", "\-"}, value="", regex=True)
    myData = myData.replace("", np.nan).astype(float)

    #   Next drop the rows which contain a np.nan value, essentially deleting rows with incomplete data.
    myData = pd.DataFrame.dropna(myData, axis=0, how='any', thresh=None, subset=None, inplace=False)

    #   Shuffle the rows of the DataFrame->numpyArray
    myData = myData.to_numpy()
    np.random.shuffle(myData)

    myData = myData.astype(int)

    #   NORMALIZE DATA
    myData[:, 0:-1] = (myData[:, 0:-1] - np.mean(myData[:, 0:-1])) / np.std(myData[:, 0:-1])
    print("myData standardized= ", myData)

    # myData = myData[:, 40:len(myData)]

    trainingAmount = int(round(len(myData) * 0.7))
    validationAmount = int(round(len(myData) * 0.1))
    testingAmount = int(round(len(myData) * 0.2))

    print("\ntotal training rows = ", trainingAmount)
    print("total validation rows = ", validationAmount)
    print("total testing rows = ", testingAmount)

    training = myData[:trainingAmount, :]
    validation = myData[trainingAmount:(trainingAmount + validationAmount), :]
    testing = myData[(trainingAmount + validationAmount):, :]

    #   This is a classifier problem where the output is either a 0 or a 1 (2 neurons)
    #   We have 57 columns thus 57 input neurons

    determine_best_setup(training, testing, validation)

    exit()
