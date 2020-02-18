'''
Created on 25-Jul-2019

@author: Rohit.Garg
'''

import os
import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
import pickle
import sklearn.tree as tree
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
print('*andu')

model = pickle.load(open('D:/RandomForest/randomForest_model_95_18_87pt1.sav', 'rb'))
dt1 = model.estimators_[1]
dot_data = StringIO()
class_names = ['1', '2-3', '4-5-6', '7', '8', '9', '10', '11', '12-13', '14']
feature_names = list(range(28))
tree.export_graphviz(dt1,
 out_file=dot_data,
 class_names=class_names, # the target names.
 feature_names=feature_names, # the feature names.
 filled=True, # Whether to fill in the boxes with colours.
 rounded=True, # Whether to round the corners of the boxes.
 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
img = Image(graph.create_png())