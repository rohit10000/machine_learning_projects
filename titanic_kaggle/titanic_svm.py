#importing libraries
from sklearn import cross_validation
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt

#importing training data 
titanic_train_data = pd.read_csv("train.csv")

#importing testing data
titanic_test_data = pd.read_csv("test.csv")

#droping the bug features
titanic_train_data = titanic_train_data.drop(columns = ["Name","Ticket","Fare","Cabin","Embarked","PassengerId"])
titanic_test_data = titanic_test_data.drop(columns =["Name","Ticket","Fare","Cabin","Embarked","PassengerId"])

#replacing female and male with 1 and 0 respectively
titanic_train_data = titanic_train_data.replace("female", 1)
titanic_train_data = titanic_train_data.replace("male", 0)
titanic_test_data = titanic_test_data.replace("female", 1)
titanic_test_data = titanic_test_data.replace("male", 0)

# filling up the nan values of titanic_test_data["Age"] with the mean of the mode of the rest of the
# non nan values . This is for training_data
train_nan_fillup_value = titanic_train_data["Age"].mode().mean()
titanic_train_data["Age"] = titanic_train_data["Age"].fillna(train_nan_fillup_value)

# filling up the nan values of titanic_test_data["Age"] with the mean of the mode of the rest of the
# non nan values . This is for testing_data
test_nan_fillup_value = titanic_test_data["Age"].mode().mean()
titanic_test_data["Age"]=titanic_test_data["Age"].fillna(test_nan_fillup_value)

#list_of_columns = list(titanic_train_data.columns.values)

#prepairing features and target using iloc on DataFrame
features_train = titanic_train_data.iloc[:,1:6]
labels_train = titanic_train_data.iloc[:,0]
features_test = titanic_test_data.iloc[:,1:6]

#Selecting the best 5 traing_features
features = SelectKBest(chi2, k=5).fit_transform(features_train,labels_train)
# using svm for prediction
rnn = svm.SVC()
rnn.fit(features_train, labels_train)
pred = rnn.predict(features_test)

print(accuracy_score(labels_train, rnn.predict(features_train)))
#saving the result to filemane
filename="prediction.csv"
np.savetxt(filename, pred, delimiter=',')

##########################
# Below is the rough.
##########################
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,target, test_size = 0.1, random_state = 42)

'''
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
#pickle.
print len(features_test)

'''


'''
parameters= {'kernel':('linear','rbf'),'C':[1,10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)
'''
