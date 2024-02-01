#IMN Lesson 4: Guide on Pandas, Matplotlib, and NumPy (+ a variety of packages for model building)

#https://www.w3schools.com/python/python_pip.asp

#Data Collection + Cleaning: Pandas
import pandas as pd

#Reading in + exploring data.
alzheimers_data = pd.read_csv("/Users/brian/Documents/Python/Data Science/alzheimer_data.csv")

print(alzheimers_data.info())

print(alzheimers_data[:6])
print(alzheimers_data.head())
print(alzheimers_data.tail())
print(type(alzheimers_data))

#df = pd.read_json('test.json') #Can also read JSON files.

#Cleaning data (preprocessing)
alzheimers_data.dropna(inplace = True)
print(alzheimers_data.info())

alzheimers_data["age"].fillna(69, inplace = True)

print(alzheimers_data[7:8])
alzheimers_data.loc[7, 'age'] = 80
print(alzheimers_data[7:8])

print(alzheimers_data[5:6])
alzheimers_data.drop(5, inplace = True) #Removes a row.
print(alzheimers_data[5:6])

#To reset the data frame, just read it again:
alzheimers_data = pd.read_csv("/Users/brian/Documents/Python/Data Science/alzheimer_data.csv")
print(alzheimers_data.info())

#Removing duplicates
print(alzheimers_data.duplicated())
alzheimers_data.drop_duplicates(inplace = True)

#Converting data types.
#1.
pythonData = {
  "brainVolume": [100, 280, 390],
  "alzheimers": [1, 1, 0]
}
df = pd.DataFrame(pythonData)
print(df)

#2.
brainVolume = {1: 100, 2: 280, 3: 390}
series = pd.Series(brainVolume)


print(dir(pd)) #Lists all functions of the panda library

#Pandas for data analysis:
print(alzheimers_data.corr(numeric_only=True))
#Returns relatoinship between every variable, between -1 and 1. Closer to -1 or 1 means it is a strong relationship.



#Matplotlib Package (Plotting + Data Visualization):
import matplotlib as mpl
import numpy as np #General package that works with numbers and mathematical functions. Will be used heavily in conjunction with matplotlib, but can also be used for basically any part of the data science workflow.

import matplotlib.pyplot as plt #Lots of functions also come from this submodule.


#Before using matplotlib on our dataset, I will showcase the basics of matplotlib and pyplot when used in data visualization
xpoints = np.array([1, 8, 10])
ypoints = np.array([3, 10, 15])

plt.ion() #Allows for changes to be made to the plots in real time. MAKE SURE YOU INCLUDE THIS STEP.

plt.title("Graph Name")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.plot(xpoints, ypoints, linestyle = 'dotted', marker = 'o')
plt.show()

#Different types of plots are better for different types of variables:

#SCATTERPLOT:
plt.clf() #Clears everything on the plot w/o deleting.

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = '#88c999', cmap = 'viridis') #Built-in color map. Look online for more that you like.
plt.show()


#BAR GRAPH:
plt.clf()

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()

plt.clf()
plt.barh(x, y, color = "red", width = 0.5)
plt.show()


#HISTOGRAMS:
x = np.random.normal(170, 10, 250) #Makes an array of 250 numbers with mean 170 and average deviation of 10 from the mean.

plt.hist(x) #hist() function makes a histogram out of an array.
plt.show() 


#PIE CHARTS
y = np.array([35, 25, 25, 15])

plt.clf()
plt.pie(y, labels = ["Apples", "Bananas", "Cherries", "Dates"], explode = [0.2, 0, 0, 0], colors = ['black', 'blue', 'red', 'green'])
plt.legend(title = "Four Fruits:")
plt.show() 

#In-depth guide on how to customize the plot:
#https://www.w3schools.com/python/matplotlib_line.asp
#https://www.w3schools.com/python/matplotlib_markers.asp
#https://www.w3schools.com/python/matplotlib_labels.asp

#But to use it well, we must use it in conjunction with our Alzheimer's data:
plt.clf()
alzheimers_data.plot(kind = 'scatter', x = 'age', y = 'diagnosis')
plt.show()

#Some types of graphs are worse at portraying the same data. You must know which graphs to use for different data types. In this case, a bar graph would return a very confusing data visualization due to the large amount of ages.
plt.clf()
alzheimers_data.plot(kind = 'bar', x = 'age', y = 'diagnosis')
plt.show()

#You can also plot frequencies of different occurrences of each diagnosis.
plt.clf()
alzheimers_data["diagnosis"].plot(kind = 'hist')
plt.show()

#Play around with matplotlib by yourself in your own data science projects - there are lots of possibilities.



#NumPy

#MANY ADVANCED MATHEMATICAL FUNCTIONS.
#But mathematical functions here aren't our focus. Here are some examples anyway:

print(np.mean(alzheimers_data['age']))
print(np.median(alzheimers_data['age']))
print(np.std(alzheimers_data['age']))
print(np.percentile(alzheimers_data['age'], 75))


#To look at the mathematical functions available with NumPy, refer to:
#https://www.w3schools.com/python/numpy/default.asp


#It has a large role in model building, the last phase in the data science workflow. But for model building we will look at a variety of different packages as the various types of models cannot be constrained to one package:

#For the most simple, simple linear regression, it actually involves SciPy instead of NumPy (but the rest involves NumPy). Here it is in action, comparing age and Alzheimer's diagnosis.
from scipy import stats

slope, intercept, r, p, std_err = stats.linregress(alzheimers_data["age"], alzheimers_data["diagnosis"]) #This type of assignment assigns the 5 values to the 5 values returned by the linregress() function

print(slope)
print(intercept)

#You can also plot this model:
def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.clf()
plt.scatter(alzheimers_data["age"], alzheimers_data["diagnosis"])
plt.plot(alzheimers_data["age"], mymodel)
plt.show()

#Makes the relationship between variables clear and visualized.


#For multiple linear regression, with multiple independent variables (such as comparing age and education to diagnosis), we use the sklearn package.
from sklearn import linear_model as lm

X = alzheimers_data[['age', 'educ']]
y = alzheimers_data['diagnosis']

model = lm.LinearRegression()
model.fit(X, y)

predictedDiag = model.predict([[70, 16]]) #We can use this model with a predict() function (using age 70 and 16 years of education) to predict alzheimer's diagnosis
print(predictedDiag)

#For splitting data into training and testing data (training is what the model is trained with, then testing data will be new data the newly-trained model will be tested with)
np.random.seed(2) #Setting the seed so that the training/testing split stays the same.
x = alzheimers_data["age"]
y = alzheimers_data["diagnosis"]

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))

from sklearn.metrics import r2_score

r2 = r2_score(test_y, mymodel(test_x)) #Tests the model with new testing data, and gives an r2 score, which is a measure of accuracy.

print(r2)