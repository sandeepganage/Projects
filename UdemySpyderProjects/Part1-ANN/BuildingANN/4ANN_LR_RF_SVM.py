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


# Train and Test on SVM classifier
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)


# Train and Test on LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# Train and Test on RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)