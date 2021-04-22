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
data = pd.read_csv("cars.csv")
# select columns
data = data[["manufacturer_name", "model_name","transmission","color","odometer_value","year_produced","price_usd","engine_fuel","body_type","drivetrain"]]

# filter data
data = data[data['price_usd'] < 15000]
data = data[data['odometer_value'] < 400000]
data = data[data['year_produced'] > 1988]

# use categorical encoding
dum_df = pd.get_dummies(data)
data = data.merge(dum_df,how='left')

# drop columns without numerical values
predict="price_usd"
dropData = data.drop([predict,"manufacturer_name","model_name","transmission","color","engine_fuel","body_type","drivetrain"],1)

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
print("Co: \n", linear.coef_)
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
