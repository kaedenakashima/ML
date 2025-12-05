import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
training_data = pd.read_csv('sales.csv')
training_data.describe()
x = training_data.iloc[:, :-1].values
y = training_data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =.20,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# minkowski is for ecledian distance
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# Model training
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]
print(y_pred)
print(y_prob)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))
new_pred = classifier.predict(sc.transform(np.array([[40,20000]])))
new_pred_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]
# # Picking the Model and Standard Scaler
# import pickle
# model_file = "classifier.pickle"
# pickle.dump(classifier, open(model_file,'wb'))
# scaler_file = "sc.pickle"
# pickle.dump(sc, open(scaler_file,'wb'))