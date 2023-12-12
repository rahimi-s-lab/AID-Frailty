#Frailty detection
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc , precision_score, recall_score, classification_report
from sklearn.metrics import f1_score, balanced_accuracy_score
df = pd.read_excel(r'/Users/vania/Desktop/AID.xlsx')
df = df.interpolate()
df.isnull().sum().sum()
df.dropna(inplace=True)
df.isnull().sum().sum()
df.dtypes
X= df.drop ('Frailtystatus', axis=1).copy()
y= df['Frailtystatus'].copy()
print(len(X))
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=10)
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
forward_feature_selection = SequentialFeatureSelector(RandomForestClassifier()).fit(X_train,y_train)
from sklearn.decomposition import PCA
Pca=PCA(0.9)
Pca.fit(X_train)
PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
Pca.n_components_
X_train = Pca.transform(X_train)
X_test = Pca.transform(X_test)
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf2=clf.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
from sklearn.metrics import mean_squared_error
import numpy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import numpy as np
k_fold = KFold(n_splits=3) 
accuracies = cross_val_score(clf2, X, y, cv=k_fold)
print("Average accuracy:", accuracies.mean())
print("Accuracy standard deviation:", accuracies.std())
acc=clf2.score(X_test,y_test)
print("acc:",acc)
y_scores = cross_val_predict(clf2, X, y, cv=k_fold, method='decision_function')
auc_roc = roc_auc_score(y, y_scores)
print("AUC-ROC:", auc_roc)
f1 = f1_score(y, y_pred)
print("F1-score:", f1)