# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#PART - 1 : Data Pre-processing
# Read the csv data file
dataset = pd.read_csv('C:\\Users\Admin\DeepLearningA-Z\SuperVised\Part1-ANN\BuildingANN\Churn_Modelling.csv')


# We will consider all the parameters from Credit Score to EstimatedSalary(column 3 to column12)
X = dataset.iloc[:,3:13].values
# Actual output for every entry of dataset
y = dataset.iloc[:,13].values


#Encode the categorical variables in our dataset
from sklearn.preprocessing import LabelEncoder
#Lets encode the coulmn 1 (country) data
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])
#Lets encode the coulmn 2 (Gender) data
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_1.fit_transform(X[:,2])


#Split categorical encoded variables(column 1) into multiple columns
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[1]) #1->For column 1 : Country
X = oneHotEncoder.fit_transform(X).toarray()


#Exclude one feature to avoid risk of dummy variable trap
X = X[:, 1:]

# Split the arrays of matrix into test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#PART - 2 : Building ANN
# Import keras library ind required packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()
# Adding input layer and the first hidden layer
# Parameters :  1 - Output Dim
#               2 - init. Initialize the weight to some small number close to 0
#               3 - Activation function. We use Rectifier Activation function for hidden layer
#                   And Sigmoid activation function for the output layer
#               4 - input Dim
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))


# Compiling the ANN
# Arg1 : 1 - Optimizer
#        2 - Loss function
#        3 - Criterion that you choose to evaluate your model. We use accuracy criterion to improve model's
#            performance.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Connect the created NN to the training data set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Making the prediction and evaluating the model
# Predicting the test set resultfrom sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
y_pred = classifier.predict(X_test)
# This gives us the probabilities. We need binary result(true/false)
y_pred = y_pred > 0.5


# To check the prediction accuracy we need to make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Confusion matrix is the 2X2 matrix 