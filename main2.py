"""
Mithun Manivannan
CS 370 Project
Predicting Used Car Values Using Machine Learning
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import locale
locale.setlocale( locale.LC_ALL, 'en_CA.UTF-8' )


import time
start_time = time.time()
# read in dataset
n=10
data = pd.read_csv("vehicles-2.csv",usecols=[4,5,6,7,11,13,18,22], header=0, skiprows=lambda i: i % n != 0)
#data = pd.read_csv("vehicles-2.csv",usecols=[4,5,6,7,9,10,11,15], header=0, skiprows=lambda i: i % n != 0)

# filter data
data = data[data['price'] < 40000]
data = data[data['price'] > 1000]
data = data[data['odometer'] < 200000]
data = data[data['odometer'] > 1000]
data = data[data['year'] > 1995]


# use categorical encoding
dum_df = pd.get_dummies(data)
data = data.merge(dum_df,how='left')

# drop columns without numerical values
predict="price"
dropData = data.drop([predict,"manufacturer","model","transmission","paint_color","state"],axis=1)
#dropData = data.drop([predict,"manufacturer","model","cylinders","drive","fuel"],axis=1)

# train model
x = np.array(dropData)
y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

# create linear model and fit
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)

# calculate score for linear fit
acc=linear.score(x_test,y_test)
print("Score: " + str(acc))
print("--- %s seconds ---" % (time.time() - start_time))

# print predictions with test parameters and compare it to actual price
print("Welcome. Enter your car data and I will calculate its value")
repeat = 1
while(True):
    try:
        make = input("Enter the make of your car: \n").lower()
        make = "manufacturer_" + make

        type = input("Enter the model of your car: \n").lower()
        type = "model_" + type

        yr = int(input("Enter the year of your car:  \n"))
        miles = int(input("Enter the number of miles on your car: \n"))

        trans = input("Enter the transmission of your car(automatic, manual or other): \n").lower()
        trans = "transmission_" + trans

        color = input("Enter the color of your car: \n").lower()
        color = "paint_color_" + color

        place = input("Enter what state your car is located in(2 letter abbreviation): \n").lower()
        place = "state_" + place

        addArray = [0] * len(dropData.columns)
        addArray[dropData.columns.get_loc("year")] = yr
        addArray[dropData.columns.get_loc("odometer")] = miles
        addArray[dropData.columns.get_loc(make)] = 1
        addArray[dropData.columns.get_loc(trans)] = 1
        addArray[dropData.columns.get_loc(color)] = 1
        addArray[dropData.columns.get_loc(place)] = 1
        addArray[dropData.columns.get_loc(type)] = 1

        predictions=linear.predict([addArray])
    except KeyError:
        print("Invalid data. Please try Again\n\n\n")
        continue
    except ValueError:
        print("Invalid data. Please try Again\n\n\n")
        continue

    value = locale.currency(predictions[0], grouping=True)
    print("Your car is worth " + str(value))
    repeat = int(input("Would you like to enter another car?\nEnter 1 to try again or 0 to stop: \n\n\n\n"))
    if repeat == 1:
        continue
    elif repeat == 0:
        break

