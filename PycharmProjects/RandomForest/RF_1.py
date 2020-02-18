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

clf = pickle.load(open('D:/RandomForest/randomForest_model_95_18_87pt1.sav', 'rb'))
dt0 = clf.estimators_[0]
dot_data = StringIO()
class_names = ['1', '2-3', '4-5-6', '7', '8', '9', '10', '11', '12-13', '14']
feature_names = list(range(28))
features  = [feature_names[i] for i in range(len(feature_names))]
impurity=dt0.tree_.impurity
importances = dt0.feature_importances_
SqlOut=""

#global Conts
global ContsNode
global Path
#Conts=[]#
ContsNode=[]
Path=[]
global Results
Results=[]

def print_decision_tree(tree, feature_names, offset_unit='' ''):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value

    if feature_names is None:
        features = ['f%d'%i for i in tree.tree_.feature]
    else:
        features = [feature_names[i] for i in tree.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0,ParentNode=0,IsElse=0):
        global Conts
        global ContsNode
        global Path
        global Results
        global LeftParents
        LeftParents=[]
        global RightParents
        RightParents=[]
        for i in range(len(left)): # This is just to tell you how to create a list.
            LeftParents.append(-1)
            RightParents.append(-1)
            ContsNode.append("")
            Path.append("")


        for i in range(len(left)): # i is node
            if (left[i]==-1 and right[i]==-1):
                if LeftParents[i]>=0:
                    if Path[LeftParents[i]]>" ":
                        Path[i]=Path[LeftParents[i]]+" AND " +ContsNode[LeftParents[i]]
                    else:
                        Path[i]=ContsNode[LeftParents[i]]
                if RightParents[i]>=0:
                    if Path[RightParents[i]]>" ":
                        Path[i]=Path[RightParents[i]]+" AND not " +ContsNode[RightParents[i]]
                    else:
                        Path[i]=" not " +ContsNode[RightParents[i]]
                Results.append(" case when  " +Path[i]+"  then ''" +"{:4d}".format(i)+ " "+"{:2.2f}".format(impurity[i])+" "+Path[i][0:180]+"''")

            else:
                if LeftParents[i]>=0:
                    if Path[LeftParents[i]]>" ":
                        Path[i]=Path[LeftParents[i]]+" AND " +ContsNode[LeftParents[i]]
                    else:
                        Path[i]=ContsNode[LeftParents[i]]
                if RightParents[i]>=0:
                    if Path[RightParents[i]]>" ":
                        Path[i]=Path[RightParents[i]]+" AND not " +ContsNode[RightParents[i]]
                    else:
                        Path[i]=" not "+ContsNode[RightParents[i]]
                if (left[i]!=-1):
                    LeftParents[left[i]]=i
                if (right[i]!=-1):
                    RightParents[right[i]]=i
                ContsNode[i]= "( "+ str(features[i]) + " <= " + str(threshold[i]) + " ) "

    recurse(left, right, threshold, features, 0,0,0,0)


print_decision_tree(dt0, features)
SqlOut=""
for i in range(len(Results)):
    SqlOut=SqlOut+Results[i]+ " end,"+chr(13)+chr(10)
print("*andu again!")