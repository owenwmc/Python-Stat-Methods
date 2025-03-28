# Logistic Regression on CHD data
# Code copied from MTH_461_Softmax.py

#FN = false negative. Predicting one as zero.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Here MinMaxScaler is to convert pixel values to [0,1] scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#data = pd.read_csv('/Users/owen/Desktop/School 24-25/MTH 461/Midterm/CHD.csv') # Midterm CSV file path for OS X
data = pd.read_csv('/home/oren/Desktop/School 24-25/MTH 461/Midterm/CHD.csv') # Midterm CSV file path for Linux
#data = pd.read_csv('/Users/owen/Desktop/School 24-25/MTH 461/CHD.csv', header=None) # Old CSV file path for OS X
print(data.head())
print(data.columns)

data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})
print(data.head())

# Check for missing values with isnull
print(f'The number of missing cells is {data.isnull().sum().sum()}')
print(data.head())

# Shape of data
print(f'The shape of data is {data.shape}')

# Data info
print(data.info())

# Barplot of CHD
data['chd'].value_counts().plot(kind='bar')
plt.title('CHD')
plt.show()

# identify target (response) and features (predictors)
y=data['chd']
X=data.drop('chd', axis=1)

# Split data into train and test in 6:4 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=461)

#train a logistic model
logit_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
logit_model.fit(X_train, y_train)

#print probabilities of responses
print(logit_model.predict_proba(X_test))

#print dimension of the predicted probabilities
print(logit_model.predict_proba(X_test).shape)

#first row in this prob table
print(logit_model.predict_proba(X_test).shape[0])

#first column in this prob table
print(f"prob values of first obs is: \n {logit_model.predict_proba(X_test)[0]}")

#print the predicted y value of the first observation in the test data
print(logit_model.predict(X_test)[0])

#capture the index at which logit_model.predict(X_test)[0] attains its maximum
print(f"argmax index is {np.argmax(logit_model.predict_proba(X_test)[:5])}")

print(f"the digit 4 has the largest probability with ")

#print the y value of the first observation in the test data
print(f"the true label of the first observation in y_test is {y_test.iloc[:5]}")

#prediction of classes
y_hat =logit_model.predict(X_test)

#plot a confusion matrix
cm = confusion_matrix(y_test, y_hat)
ConfusionMatrixDisplay(cm).plot()
plt.show()

#print first ten rows of probabilities
logit_model.predict_proba(X_test)[:10]

#prediction using threshhold
threshold = 0.5

#predict y_pred (the classes: 0 and 1) based on the threshold
y_pred = np.where(logit_model.predict_proba(X_test)[:,1] > threshold,1,0)

#create a dataframe to present the four: the two column prob table, y_pred and y_test
df = pd.DataFrame({'prob_0':logit_model.predict_proba(X_test)[:,0], 'prob_1':logit_model.predict_proba(X_test)[:,1], 'y_pred':y_pred, 'y_test':y_test})
print(f'prob table and stuff \n {df.head(10)}')

#create a confusion matrix and display
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

#filter the observations of FN (false negative) in test data. That is, the cases where y_test is 1, but y_pred is 0.
FN = df[(df['y_test']==1) & (df['y_pred']==0)]
if FN.empty:
    print('No false negatives')
else:
    print(FN)
print(f'The number of false negatives is {len(FN)}')

# In medical diagnosis, FN is a critical error for those who actually have cancers.
# To prevent the chance of FN, threshold value needs to be less than 0.5.
# Now update the threshold value to 0.1 and recompute the confusion matrix.
# We can either create a new threshold value or update the previous threshold value.

threshold = 0.1 # Update the threshold value
y_pred = np.where(logit_model.predict_proba(X_test)[:,1] > threshold,1,0) # Predict the classes based on the updated threshold value
# New dataframe to present the four: the two column prob table, y_pred and y_test
df = pd.DataFrame({'prob_0':logit_model.predict_proba(X_test)[:,0], 'prob_1':logit_model.predict_proba(X_test)[:,1], 'y_pred':y_pred, 'y_test':y_test})

# Find indexes in the test data where FN happens with the default threshold 0.1
FN = df[(df['y_test']==1) & (df['y_pred']==0)]
print(f'Indexes of FN with threshold 0.1: {FN.index}')
print(f'The number of false negatives with threshold 0.1 is {len(FN)}')

# Create a confusion matrix and provide the display and its diagnostic outputs addressing accuracy, sensitivity, and specificity.
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
print(f'Accuracy: {(cm[0,0]+cm[1,1])/cm.sum()}')
print(f'Sensitivity: {cm[1,1]/(cm[1,0]+cm[1,1])}')
print(f'Specificity: {cm[0,0]/(cm[0,0]+cm[0,1])}')
print(f'Precision: {cm[1,1]/(cm[0,1]+cm[1,1])}')



