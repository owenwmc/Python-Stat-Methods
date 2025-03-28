# Softmax Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Here MinMaxScaler is to convert pixel values to [0,1] scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_excel('/Users/owen/Desktop/School 24-25/MTH 461/Digits.xlsx', header=None) # File path for OS X
print(data.head())

# Read the first 49 rows besides the last column
data_49 = data.iloc[:49, 0:-1]
print(data_49.head())

#Reshape each row as 8x8
data_49_sample = data_49.values.reshape(49, 8, 8)
print(data_49_sample)

#display these 49 images using for loop
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(data_49_sample[i], cmap='gray')
    plt.axis('off')
#plt.show()

first_obs = data.iloc[0, 0:-1] # First row, all columns except the last one
#print(first_obs)

# Reshape first_obs as 8x8 matrix
first_obs = first_obs.values.reshape(8, 8)
#print(first_obs)

# Use img to display the matrix
plt.imshow(first_obs, cmap='gray')
#plt.show()

# Split this data into train and test in 7:3 ratio
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=461)

#train a softmax model using logistic method
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
softmax_model.fit(X_train, y_train)

#print probabilities of responses
print(softmax_model.predict_proba(X_test))

#print dimension of the predicted probabilities
print(softmax_model.predict_proba(X_test).shape)

#first row in this prob table
print(softmax_model.predict_proba(X_test).shape[0])

#first column in this prob table
print(f"prob values of first obs is: \n {softmax_model.predict_proba(X_test)[0]}")

#print the predicted y value of the first observation in the test data
print(softmax_model.predict(X_test)[0])

#capture the index at which softmax_model.predict(X_test)[0] attains its maximum
print(f"argmax index is {np.argmax(softmax_model.predict_proba(X_test)[0])}")

print(f"the digit 4 has the largest probability with ")

#print the y value of the first observation in the test data
print(f"the true label of the first observation in y_test is {y_test.iloc[0]}")

#prediction of classes
y_hat = softmax_model.predict(X_test)

#plot a confusion matrix
cm = confusion_matrix(y_test, y_hat)
ConfusionMatrixDisplay(cm).plot()
plt.show()

