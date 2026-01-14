#  npm i serve -D
#  npx serve -s build
# conda deactivate
# rm -rf venv
# python3 -m venv venv
# source venv/bin/activate
# pip
# lsof -n | grep LISTEN
# lsof -i :5000
# kill -9 <PIN>
#pip install numpy pandas scikit-learn flask tensorflow
#K-Nearest Neighbor Classification
#h
#| o  x  x
#|o  ? x
#-----------w o=cat,x=dog
import numpy as np
import pandas as pd
import statistics
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import tensorflow as tf
sc=StandardScaler()

# 1 Analyse sales data
# f1=pd.read_csv('sales.csv')
# print(f1.describe())
#              Age        Salary  Purchased
# count  40.000000     40.000000  40.000000
# mean   38.100000  49525.000000   0.550000
# std    12.557151  19046.484971   0.503831
# min    18.000000  20000.000000   0.000000
# 25%    27.750000  35000.000000   0.000000
# 50%    37.500000  48500.000000   1.000000
# 75%    47.250000  60000.000000   1.000000
# max    60.000000  95000.000000   1.000000

# 2.1 Create dataset x=age_salary, y=purchased_list
# f1=pd.read_csv('sales.csv')
f1=pd.read_csv('sales2.csv')
age_salary=f1.iloc[:,:-1].values
purchased_list=f1.iloc[:,-1].values
# print(age_salary)
# print(purchased_list)

# 2.2 find deviation with stdev of 1* *avr btw next item
x_train, x_test, y_train, y_test = train_test_split(age_salary,purchased_list,test_size=.20,random_state=0)
x_train=sc.fit_transform(x_train)#80% of items are used for training
x_test=sc.transform(x_test)#20% of items are used for test
# print(x_train) 
# print(x_test)
# sample1 stdev  res:1.52 
#print(statistics.stdev([1,2,4]))
# sample2 dev with stev of 1 res:[-1.3333333333333335, -0.3333333333333335, 1.6666666666666665]
# avr = sum([1,2,4]) / len([1,2,4]) # statistics.mean([1,2,4])
# x2=[]
# for x1 in [1,2,4]:
#     x2.append(x1-avr)
# print(x2)

#-------------------
# Ch3 Use KNeighbor Classification
#--------------------
# 3.1 Find purchase rate per customer with KNeighbor Classification
# classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
# classifier.fit(x_train,y_train)
# y_pred=classifier.predict(x_test)
# y_prob=classifier.predict_proba(x_test)[:,1]
# print(y_pred)#[1   1   1 0   1   0   0 0]
# print(y_prob)#[1 0.6 0.8 0 0.8 0.2 0.4 0] percentage customer buy the product

# 3.2 Find accuracy level by confusion matrix result:0.875
# actual  [1 1]
# estimate[1 0] 
# score   [T F]
# cm=confusion_matrix(y_test,y_pred)
# print(cm)#[[3 0][1 4]] T ok, F wrong, F wrong, T ok (7 items are correct out of 8)
# print(accuracy_score(y_test,y_pred)) #0.875
# print(classification_report(y_test,y_pred))

# ⭐︎3.3.1 way1 estimate purchase rate mannuary
# example1 age 40, salary 20000 
# answer1=classifier.predict(sc.transform(np.array([[40,20000]])))
# prob1=classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]
# print(answer1,prob1) #ML prediction: not buy For 20%, this customer buy the product.
# example2 age 42, salary 50000
# answer2=classifier.predict(sc.transform(np.array([[42,50000]])))
# prob2=classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]
# print(answer2,prob2) #ML prediction: buy For 80%, this customer buy the product.

# ⭐︎3.3.2 way2 create local file and estimate purchase rate
# Create local file
# pickle.dump(classifier, open('classifier.pickle', 'wb'))
# pickle.dump(sc,open('sc.pickle','wb'))
# Estimate purchase rate
# f2=pickle.load(open('classifier.pickle','rb'))
# f3=pickle.load(open('sc.pickle','rb'))
# # example1 age 40, salary 20000 
# answer1=f2.predict(f3.transform(np.array([[40,20000]])))
# prob1=f2.predict_proba(f3.transform(np.array([[40,20000]])))[:,-1]
# print(answer1, prob1)
# # example2 age 42, salary 50000
# answer2=f2.predict(f3.transform(np.array([[42,50000]])))
# prob2=f2.predict_proba(f3.transform(np.array([[42,50000]])))[:,-1]
# print(answer2, prob2)


#-------------------
# Ch4 Use neural network with pytorch
#--------------------
# # 4.1 Create base of neural network
# Xtrain_=torch.from_numpy(x_train).float()
# Xtest_=torch.from_numpy(x_test).float()
# ytrain_=torch.from_numpy(y_train)
# ytest_=torch.from_numpy(y_test)
# # print(Xtrain_.shape, ytrain_.shape)
# # print(Xtest_.shape, ytest_.shape)
# input_size=2
# output_size=2
# hidden_size=10
# class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.fc1 = torch.nn.Linear(input_size, hidden_size)
#        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
#        self.fc3 = torch.nn.Linear(hidden_size, output_size)
#    def forward(self, X):
#        X = torch.relu((self.fc1(X)))
#        X = torch.relu((self.fc2(X)))
#        X = self.fc3(X)
#        return F.log_softmax(X,dim=1)
# model1 = Net() #mannual

# # 4.2 Train the neural network
# optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)
# loss_fn = nn.NLLLoss()
# epochs = 100
# for epoch in range(epochs):
#   optimizer.zero_grad()
#   Ypred = model1(Xtrain_)
#   loss = loss_fn(Ypred,  ytrain_)
#   loss.backward()
#   optimizer.step()
# #   print('Epoch',epoch, 'loss',loss.item())
# # Epoch 0 loss 0.7977757453918457
# # Epoch 1 loss 0.7583175897598267 ...
# list(model1.parameters())

# 4.3.1 way1 Find purchase rate with pytorch mannuary
# # # example1 age 40, salary 20000 
# torch.from_numpy(sc.transform(np.array([[40,20000]]))).float()
# pred1 = model1(torch.from_numpy(sc.transform(np.array([[40,20000]]))).float())
# _, answer1 = torch.max(pred1.data,-1)
# print(pred1)#tensor([[-0.0054, -5.2190]] each vector of possibility rate
# print(answer1) #tensor([0])
# # example2 age 42, salary 50000
# torch.from_numpy(sc.transform(np.array([[42,50000]]))).float()
# pred2 = model1(torch.from_numpy(sc.transform(np.array([[42,50000]]))).float())
# _, answer2 = torch.max(pred2.data,-1)
# print(pred2) #tensor([[-0.9004, -0.5216]])
# print(answer2) # tensor([1])

# ⭐︎4.3.2 way2 create local file and estimate purchase rate
# Create local file
# torch.save(model.state_dict(),'pt_dataset')
# model2 = Net() #use file
# model2.load_state_dict(torch.load('pt_dataset'))
# # # example1 age 40, salary 20000 
# torch.from_numpy(sc.transform(np.array([[40,20000]]))).float()
# pred1 = model2(torch.from_numpy(sc.transform(np.array([[40,20000]]))).float())
# _, answer1 = torch.max(pred1.data,-1)
# print(pred1)#tensor([[-0.0054, -5.2190]] each vector of possibility rate
# print(answer1) #tensor([0])
# # example2 age 42, salary 50000
# torch.from_numpy(sc.transform(np.array([[42,50000]]))).float()
# pred2 = model2(torch.from_numpy(sc.transform(np.array([[42,50000]]))).float())
# _, answer2 = torch.max(pred2.data,-1)
# print(pred2) #tensor([[-0.9004, -0.5216]])
# print(answer2) # tensor([1])
#-------------------
# Ch5 Use tesnsorflow kerass only works at google colab
#--------------------
# go to google colab and access ml_tf.ipynb

# # 5.   ?
# # df = pd.DataFrame(data)
# # scaler = StandardScaler()
# # scaled_data = scaler.fit_transform(df)
# # scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
# # print("Original Data:\n", df)
# # print("\nScaled Data:\n", scaled_df)


# data = {
#     'Age': [25, 30, 35, 40, 45],
#     'Salary': [50000, 60000, 75000, 90000, 120000]
# }
# stock = {
#     'A': [12397.91,11559.36,11138.66,10395.18,9446.01,8928.29,8870.16,8839.91,8695.06,9006.78,8542.73,9520.89,10083.56,9723.24,8802.51,8455.35,8434.61,8988.39,8700.29,8955.20,9833.03,9816.09,9693.73,9849.74,9755.10,10624.09,10237.92,10228.92,9937.04,9202.45,9369.35,8824.06,9537.30,9382.64,9768.70,11057.40,11089.94,10126.03,10198.04,10546.44,9345.55,10034.74,10133.23,10492.53,10356.83,9958.44,9522.50,8828.26,8109.53,7568.42,7994.05,8859.56,8512.27,8576.98,11259.86,13072.87,13376.81,13481.38,14338.54,13849.99],
#     'B': [4860,4765,4365,4005,3535,3065,3040,3095,3020,3190,3040,3305,3570,3355,2810,2565,2509,2644,2688,2734,3155,3300,3400,3230,3350,3820,3400,3220,3220,2859,2998,2860,3050,3080,3280,3665,3745,3330,3490,3880,3440,3660,3570,3990,3990,3670,3810,3850,3120,3180,2925,2905,3000,3730,4380,4930,4660,5010,5370,5270],
#     'C': [4015,3850,3980,3595,3550,3795,3770,3870,3725,3585,3535,3980,4125,3725,3525,3670,3460,3360,3440,3230,3340,3450,3490,3380,3400,3725,3670,3755,3860,3855,4010,3760,3840,4015,4200,4285,4020,4010,3790,3920,3900,4000,4380,4570,4060,3880,3950,3740,3550,3830,3880,3940,4000,4240,4300,4780,4740,4290,4250,4540],
#     'D': [30650,25430,24080,21840,18720,17780,18150,18270,16150,15880,17490,17930,18860,16830,15150,14000,12320,14230,13990,14480,13680,12970,11840,12710,10410,12760,11950,12930,13240,10530,11760,11580,12930,13520,12800,14370,16250,15000,15020,17470,15680,15120,11360,11150,12290,12600,11290,10300,11170,9950,11540,12980,10950,10260,10550,11130,12110,10060,9120,9680]
# }

