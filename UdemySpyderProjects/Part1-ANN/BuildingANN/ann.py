#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('D:\\Udemy\ArtificialNeuralNetwork\Artificial_Neural_Networks\Churn_Modelling.csv')

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

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Excluded one variable to eliminate risk of dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
# It Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
# Import keras library and required packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Now, As we discussed the steps to train the ANN

# Step 1 -> Randomly initialize the weights to the each nodes (by some small number close to 0).
# This will be done by Dense function

# Step 2 -> Input the first observation of your dataset in the input layer, 
# each feature in one input node. As we have 11 number of features in our problem. 
# Number of nodes in input layer is nothing but the number of independent variables 
# we have in our matrix of features.


# Step 3 -> Forward propagation, from left to right neurons are activated in a way that the impact of
# each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result y
# We will choose Rectifier Activation function for the hidden layer and Sigmoid activation function for the output layer

# Step 4 -> Compare the predicted result to the actual result. Measure the generated error

# Step 5 -> Back - Propagation : From right to left

# Step 6 -> Repeat step 1 to 5 and update the weights after each observation (Reinforcement Learning). Or:
#           Repeat step 1 to 5 and update the weights only after a batch of observation (Batch Learning)

# Step 7 -> When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Parameter : 1 - Output Dim.
#             2 - init. Initialize the weight to a small num
#             3 - activation function. We use rectifier activation layer for hidden layer
#             4 - input dim.

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Now, Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# We don't need 'input_dim' parameter in this above layer since its a second hidden layer


# Now, Adding output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
# Arg1 : 1 - Optimizer (Here adam is kind of Stochastic Gradient Descent Algorithm)
#        2 - Loss function (binary_crossentropy - Logarithmic loss function for binary output. 
#                 categorical_crossentropy  - Logarithmic loss function for categorical output)
#        3 - Criterion that you choose to evaluate your model. 
#            We use accuracy criterion to improve model's performance.
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

# Part 3 -> Making the predictions and evaluating the model
# Predicting the test set results
y_pred = classifier.predict(X_test)


# Now this will give us probabilities. We need actual binary result (true / false)
# We have to convert it to binary form
# So we need to choose threshold
y_pred = (y_pred > 0.5)


# To check the prediction we will make the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

""" Now lets find out for the following customer
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000 """

new_prediction = classifier.predict(sc.fit_transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

# Now this will give us probabilities. We need actual binary result (true / false)
# We have to convert it to binary form
# So we need to choose threshold
new_prediction = (new_prediction > 0.5)


# Part 4 -> Evaluating, Improving and Tuning the ANN
# Evaluating the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs=1)

mean = accuracies.mean()
variance = accuracies.std()


# Improving the ANN
# Dropout Regularization to reduce overfitting if needed



#Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[20,25],
              'nb_epoch':[100,110],
              'optimizer':['adam','rmsprop']}

grid_Search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_Search = grid_Search.fit(X_train, y_train)
best_parameters = grid_Search.best_params_
best_accuracy = grid_Search.best_score_
