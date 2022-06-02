import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

data = pd.read_excel('Iris.xlsx')

data_copy = data.copy()
"""
print(data_copy.head())

print(data_copy.isnull().sum())

print(data_copy.describe())
    
print(data_copy.corr())
"""

x = data_copy.iloc[ :,:4]
y = data_copy.iloc[:,4:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)#split data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# ----- Logistic Regression -----
from sklearn.linear_model import LogisticRegression
log_r = LogisticRegression(random_state = 0)
log_r.fit(x_train, y_train)

y_pred = log_r.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression")
print(cm)


# ----- KNN -----
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", algorithm="auto")

knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("KNN")
print(cm)


# ----- SVC -----
from sklearn.svm import SVC
svc = SVC(kernel = "rbf")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)  

cm = confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)
 

# ----- Naive Bayes - CategoricalNB -----
from sklearn.naive_bayes import CategoricalNB
cnb = CategoricalNB()
cnb.fit(x_train, y_train)

y_pred = cnb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("CNB")
print(cm)
 

# ----- Decision Tree -----
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy", min_samples_split=10)
dtc.fit(x_train, y_train)

y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("DTC")
print(cm)


# ----- Random Forest -----
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("RFC")
print(cm)
 





