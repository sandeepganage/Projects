#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# imorting dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# We will consider all the parameters from Credit score(index 3) upto EstimatedSalary(index 12)
X = dataset.iloc[:, 3:13].values

# Actual Output for every entry from the dataset
y = dataset.iloc[:, 13].values


# Split the dataset into training set and test set
# Before that we have some categorical variables in our matrix. Therefore we have to encode them
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Lets encode country feature first which is index 1 and encode it
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Lets encode Gender feature which is index 2 and encode it
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Excluded one variable to eliminate risk of dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# It Split arrays or matrices into random train and test subsets
# test size  = if float then it represents the proportion of the dataset to include in the test split
# test size  = if int then it represents the absolute number of the test samples
# if none then the value is set to the complement of the train size. If train size is
# also none then it will be set to 0.25


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Part 2 - Lets make an ANN
# Now, As we discussed the steps to train the ANN

# Step 1 -> Randomly initialize the weights to the each nodes (small nuber close to 0).
# This will be done by Dense function

# Step 2 -> Input the first observation of your dataset in the input layer,
# each feature in one input node. As we have 11 number of features in our problem. Number of
# nodes in input layer is nothing but the number of independent variables we have in our matrix of features.

# Step 3 -> Forward propagation, from left to right neurons are activated in a way that the impact of
# each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result y
# We will choose Rectifier Activation function for the hidden layer and Sigmoid activation function for the output layer

# Step 4 -> Compare the predicted result to the actual result. Measure the generated error

# Step 5 -> Back - Propagation : From right to left

# Step 6 -> Repeat step 1 to 5 and update the weights after each observation (Reinforcement Learning). Or:
#           Repeat step 1 to 5 and update the weights only after a batch of observation (Batch Learning)

# Step 7 -> When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

# Import keras library and required packages
import keras

# Also import modules
# Sequential modules which is required to initialize the neural network
# Dense module required to build the layers of ANN

from keras.models import Sequential
from keras.layers import Dense

# Now initialize the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Parameter : 1 - Output Dim.
#             2 - init. Initialize the weight to a small num
#             3 - activation function. We use rectifier activation layer for hidden layer
#             4 - input dim.

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# So, here our first hidden layer and at the same time our input layer is very well added in our ANN
# Now, next step, add a new hidden layer, which is actually not necessarily useful for our dataset

# Now, Adding second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
# We don't need 'input_dim' parameter in this above layer since its a second hidden layer


# Now, Adding output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# Compiling the ANN
# Arg1 : 1 - Optimizer
#        2 - Loss function
#        3 - Criterion that you choose to evaluate your model. We use accuracy criterion to improve model's
#            performance.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Till now we have created the NN but haven't made any connection to our training set and test set
# Now. Lets fit the ANN to the training set

# Parameters : 1 - Training input
#              2 - Output (dependant variable)
#              3 - Remember 6th step while training ANN with Stochastic Gradient Descent
#              Step 6 -> Repeat step 1 to 5 and update the weights after each observation (Reinforcement Learning). Or:
#              Repeat step 1 to 5 and update the weights only after a batch of observation (Batch Learning)
#

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Making the predictions and evaluating the model
# Predicting the test set results
y_pred = classifier.predict(X_test)
# Now this will give us probabilities. We need actual binary result (true / false)
# We have to convert it to binary form
# So we need to choose threshold
y_pred = (y_pred > 0.5)

# To check the prediction we will make the Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


# Now test Khandu's data

dataset1 = pd.read_csv('Churn_Modelling1.csv')

# We will consider all the parameters from Credit score(index 3) upto EstimatedSalary(index 12)
X1 = dataset1.iloc[:, 3:13].values

# Actual Output for every entry from the dataset
y1 = dataset1.iloc[:, 13].values


# Lets encode country feature first which is index 1 and encode it
labelencoder_X_11 = LabelEncoder()
X1[:, 1] = labelencoder_X_11.fit_transform(X1[:, 1])

# Lets encode Gender feature which is index 2 and encode it
labelencoder_X_11 = LabelEncoder()
X1[:, 2] = labelencoder_X_11.fit_transform(X1[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X1 = onehotencoder.fit_transform(X1).toarray()

# Excluded one variable to eliminate risk of dummy variable trap
X1 = X1[:, 1:]

# from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)



print('Done!')