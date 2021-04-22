"""
Mithun Manivannan
CS 370 Project
Predicting Used Car Values Using Machine Learning
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# read in dataset
n=10
data = pd.read_csv("vehicles-2.csv",usecols=[4,5,6,7,8,11,13,18,22], header=0, skiprows=lambda i: i % n != 0)
# select columns
#data = data[[""price","manufacturer","model","condition","transmission","paint_color","state"]]

# filter data
data = data[data['price'] < 50000]
data = data[data['price'] > 500]
data = data[data['odometer'] < 400000]
data = data[data['odometer'] > 1000]
data = data[data['year'] > 2004]

# use categorical encoding
dum_df = pd.get_dummies(data)
data = data.merge(dum_df,how='left')

# drop columns without numerical values
predict="price"
dropData = data.drop([predict,"manufacturer","model","condition","transmission","paint_color","state"],axis=1)

# train model
x = np.array(dropData)
y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

# create linear model and fit
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)

# calculate score for linear fit
acc=linear.score(x_test,y_test)
print("Score: " + str(acc))

# print coefficients and intercepts
#print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# print predictions with test parameters and compare it to actual price
predictions = linear.predict(x_test)
for i in range(10):
    print("Prediction: " + str(predictions[i]))
    idx = 0;
    for val in x_test[i]:
        if val:
            print(dropData.columns[idx] + ": " + str([val]), end=' ')
        idx+=1
    print("\nActual: " + str(y_test[i]) + "\n\n\n\n")