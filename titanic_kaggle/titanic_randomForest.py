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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


titanic_train_data = pd.read_csv("train.csv")

titanic_train_data['child'] = 1
titanic_train_data['child'][titanic_train_data['Age'] >= 18] = 0
titanic_train_data['no_of_family_members'] = titanic_train_data['SibSp'] + titanic_train_data[ 'Parch']
titanic_train_data = titanic_train_data.drop(columns = ["Name","Ticket","Cabin","PassengerId","Age","SibSp","Parch"])
#dropping all row_entries with NaN
#titanic_train_data = titanic_train_data.dropna()
#titanic_test_data = titanic_test_data.dropna()
#replacing female and male with 1 and 0 respectively
titanic_train_data = titanic_train_data.replace("female", 1)
titanic_train_data = titanic_train_data.replace("male", 0)
titanic_train_data = titanic_train_data.replace("S", 0)
titanic_train_data = titanic_train_data.replace("Q", 1)
titanic_train_data = titanic_train_data.replace("C", 2)
titanic_train_data["Fare"] = titanic_train_data["Fare"].astype('int32')
titanic_train_data=titanic_train_data.dropna();

#print (titanic_train_data)
#train_nan_fillup_value = 26
#titanic_train_data["Age"] = titanic_train_data["Age"].fillna(train_nan_fillup_value)
#prepairing features and target
features = titanic_train_data.iloc[:,1:]
labels = titanic_train_data.iloc[:,0]

#features = SelectKBest(chi2, k=2).fit_transform(features,labels)
#features = pd.DataFrame(features)

#print features.columns.values
#features = np.array(features)
#labels = np.array(labels)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels, test_size = 0.1, random_state  = 42)


forest = RandomForestClassifier(max_depth = 50, min_samples_split = 2, n_estimators = 10, random_state = 1)
my_forest = forest.fit(features_train,labels_train)
pred = my_forest.predict(features_test)
print(accuracy_score(labels_test, pred))


'''
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test,pred)
'''

'''
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print (accuracy_score(labels_test, pred))
'''


'''
rnn = svm.SVC(kernel= 'linear',gamma=10,C=10)
rnn.fit(features_train, labels_train)
pred = rnn.predict(features_test)
print (accuracy_score(labels_test, pred))
'''
'''
parameters= {'kernel':('linear','rbf'),'C':[1,10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)
'''
