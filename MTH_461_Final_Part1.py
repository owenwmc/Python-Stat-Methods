# Logistic regression on breastcancer data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from warnings import simplefilter # This is to get rid of a FutureWarning for multinomial parameter for the Logistic Regression model
simplefilter(action='ignore', category=FutureWarning)

# Import the breast cancer CSV file
#data = pd.read_csv('/home/oren/Desktop/School 24-25/MTH 461/Final/breastcancer.csv') # File path for Linux
data = pd.read_csv('/Users/owen/Desktop/School 24-25/MTH 461/Final/breastcancer.csv') # File path for OS X

# Head and info of the data
print(data.head())
print(data.info())

# Check for missing values with isnull
print(f'The number of missing cells is {data.isnull().sum().sum()}')

# Shape and head of the data
print(f'The shape of data is {data.shape}')
print(data.head())

# Bar graph of the frequency distribution of breast cancer (target variable)
data['breast cancer'].value_counts().plot(kind='bar')
plt.title('breast cancer')
plt.show()

# Transform the target varaible data into numeric: 0 if benign, 1 if malignant
data['breast cancer'] = data['breast cancer'].map({'malignant': 1, 'benign': 0})

# Use seaborn library heatmap() function to provide a plot of the correlation matrix of the data.
sns.heatmap(data.corr(), annot=False, square=True) # annot=True will write data value in each cell
plt.title('Heatmap Correlation Matrix')
plt.show()

# Identify target (response) and features (predictors)
y = data['breast cancer']
X = data.drop('breast cancer', axis=1)

# Split the data into X_train, X_test, y_train, y_test in 6:4 ratio for train and test, respectively, using random_state=461.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=461)

# Fit logistic model with max_iter=10000. Provide the predicted probabilities of the first five observations in the test data.

logit_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000) # Train model
logit_model.fit(X_train, y_train) # Fit model

print(logit_model.predict_proba(X_test)[:5]) # Printing .head() will also give the same result

# Print dimension of the predicted probabilities
print(logit_model.predict_proba(X_test).shape)

# First row in this prob table
print(logit_model.predict_proba(X_test).shape[0])

# First column in this prob table
print(f"prob values of first obs is: \n {logit_model.predict_proba(X_test)[0]}")

# Print the predicted y value of the first observation in the test data
print(logit_model.predict(X_test)[0])

# Capture the index at which logit_model.predict(X_test)[0] attains its maximum
print(f"argmax index is {np.argmax(logit_model.predict_proba(X_test)[:5])}")

# Print the y value of the first observation in the test data
print(f"the true label of the first observation in y_test is {y_test.iloc[:5]}")

# Prediction of classes
y_hat = logit_model.predict(X_test)

# Print first ten rows of probabilities
logit_model.predict_proba(X_test)[:10]

# Predict using the threshold of 0.5
threshold = 0.5

# Predict y_pred (the classes: 0 and 1) based on the threshold
y_pred = np.where(logit_model.predict_proba(X_test)[:,1] > threshold,1,0)

# Create a dataframe with the probabilities and the predicted classes, prob_0, prob_1, y_pred, and y_test
df = pd.DataFrame({'prob_0':logit_model.predict_proba(X_test)[:,0], 'prob_1':logit_model.predict_proba(X_test)[:,1], 'y_pred':y_pred, 'y_test':y_test})
print(f'prob table and stuff \n {df.head(10)}')

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

#filter the observations of FN (false negative) in test data. That is, the cases where y_test is 1, but y_pred is 0.
FN = df[(df['y_test']==1) & (df['y_pred']==0)]
FP = df[(df['y_test']==0) & (df['y_pred']==1)] # False positives
print(f'The number of false negatives is {len(FN)}')
print(f'The number of false positives is {len(FP)}')

# FN = false negative. Predicting one as zero.

# In the context of our data and logistic model, a False Negative (FN) happens when our model incorrectly predicted a malignant case as benign.
# More specifcally, predicting one as zero. The true label y_test (malignant) is 1, but the predicted label y_pred (benign) is 0.

# Similarly, in this context, a False Positive (FP) means that the model incorrectly predicted a benign case as malignant.
# More specifically, predicting zero as one. The true label y_test (benign) is 0, but the predicted label y_pred (malignant) is 1.

# In this context, FN is more critical than FP because it means that a malignant case was predicted as benign.
# This could lead to a cancer patient not getting life saving treatment.

# Therefore, in our context of cancer diagnosis we need to be more conservative in our predictions.
# Lowering the threshold value from 0.5 can help reduce the number of false negatives, but it may also increase number of false positives.
# However, with cancer diagnosis it would be more important to to have less FNs.

# Find indexes in the test data where FN happens with the default threshold 0.5
FN = df[(df['y_test']==1) & (df['y_pred']==0)]
print(f'Indexes of FN with threshold 0.5: {FN.index}')
print(f'The number of false negatives with threshold 0.5 is {len(FN)}')

threshold = 0.1 # Update the threshold value
y_pred = np.where(logit_model.predict_proba(X_test)[:,1] > threshold,1,0) # Predict the classes based on the updated threshold value
# Create a new dataframe to present the four: the two column prob table, y_pred and y_test
df = pd.DataFrame({'prob_0':logit_model.predict_proba(X_test)[:,0], 'prob_1':logit_model.predict_proba(X_test)[:,1], 'y_pred':y_pred, 'y_test':y_test})

# Find indexes in the test data where FN happens with the threshold 0.1
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



