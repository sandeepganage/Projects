# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def binarylab(labels, classes):
   numSamples = len(labels)
   #Define an Empty Array
   x = np.zeros([numSamples, classes], dtype="uint8")
   
   #Read Each pixel label and put it into corresponding label plane
   for i in range(numSamples):
       x[i, labels[i]]=1
   return x


#PART - 1 : Data Pre-processing
# Read the csv data file
dataset = pd.read_csv('D:\TestesStaging\ModelBuild\Feature_Matrix.csv')


# We will consider all the parameters from Credit Score to EstimatedSalary(column 3 to column12)
X = dataset.iloc[:,:28].values
# Actual output for every entry of dataset
y = dataset.iloc[:,28].values


#Encode the categorical variables in our dataset
from sklearn.preprocessing import LabelEncoder
#Lets encode the coulmn 1 (country) data
labelEncoder_y = LabelEncoder()
y[:] = labelEncoder_y.fit_transform(y[:])

max = np.max(y)

y = binarylab(y, 10)


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
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(output_dim = 28, init='uniform', activation='relu', input_dim=28))
classifier.add(Dense(output_dim = 14, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 14, init='uniform', activation='relu'))
classifier.add(Dense(output_dim = 10, init='uniform', activation='softmax'))
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#   return classifier
# classifier.fit(X_train, y_train, batch_size=10, epochs=500, validation_split=0.1)
history = classifier.fit(X_train, y_train, batch_size=10, epochs=1000, validation_split=0.1)

#_, train_accu = classifier.evaluate(X_train, y_train, verbose=0)
#_, test_accu = classifier.evaluate(X_test, y_test, verbose=0)


plt.plot(history.history['acc'], label='train accu')
plt.plot(history.history['val_acc'], label='test accu')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.legend()
plt.show()


y_pred = classifier.predict(X_test)

y_pred_class = np.argmax(y_pred, axis=1)

y_test_class = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_class, y_pred_class)

corrVal = 0
totVal = 0
for i in range(10):
    corrVal = corrVal + cm[i,i]
    for j in range(10):
        totVal = totVal + cm[i,j]
    

from keras.utils import plot_model
plot_model(classifier,to_file="D:\TestesStaging\ModelBuild\model.png")