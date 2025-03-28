# MTH 461 - Final Project Part 2
# Digit Pattern Recognitions using Classifications using Classification Trees

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

#df = pd.read_csv('/Users/owen/Desktop/School 24-25/MTH 461/Final/Digits.csv') # File path for OS X
df = pd.read_csv('/home/oren/Desktop/School 24-25/MTH 461/Final/Digits.csv') # File path for Linux
print(df.head())
print(df.info())

print(f'The number of missing cells is {df.isnull().sum().sum()}')

# Identify the target (digit) column (the last one) and the features in terms of y and X.
# Use test_size=0.3 and random_state=461 to split the data into X_train, X_test, y_train, y_test. Print their shapes.
y = df['y']
X = df.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=461)
print(f'The shape of X_train is {X_train.shape}')
print(f'The shape of X_test is {X_test.shape}')
print(f'The shape of y_train is {y_train.shape}')
print(f'The shape of y_test is {y_test.shape}')

# Use DecisionTreeClassifier(max_depth=2) to fir a single decision tree model. Plot the tree.
tree_model = DecisionTreeClassifier(max_depth=2)
tree_model.fit(X_train, y_train)

# Define feature names and class names
feat_names = X.columns.tolist()  # Feature names from the dataset. Names of input features used to train (column names of X).
names = y.unique().astype(str).tolist()  # Class names as strings. Unique values in the target variable (y).

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, class_names=names, feature_names=feat_names, impurity=True, filled=True)
plt.show()

# Least amt of impurity = gini index min
# Indicate (with explanation) which terminal node (at the bottom) has the least amount of impurity.
# In addition, describe precisely the feature region at this terminal node in terms of the feature variables.

# The terminal node at the bottom with class = 0 (digit 0) and where gini = 0.274 has the least amount of impurity.
# The feature region at this terminal node is where the features x37 <= 0.5 and x43 > 5.5.

# Use RandomForestClassifier(n_estimators=1000, max_depth=4) to find a prediction model of the data.
rf = RandomForestClassifier(n_estimators=1000, max_depth=4)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Provide the display of the confusion matrix of the Random Forest model
cm_rf = confusion_matrix(
    y_test,
    y_pred_rf,
    labels=rf.classes_
    )
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
disp_rf.plot()
plt.show()

# the feature importance plot
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# classification report
report_rf = classification_report(
    y_test,
    y_pred_rf,
    target_names=names, # Earlier defined class names variable
    zero_division=0
    )
print("Classification Report for Random Forest:")
print(report_rf)

# Regarding RF ("randomforst") in (4) above, answer the followiing:
#(a) What is the overall accurancy rate?
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"The overall accuracy rate of the Random Forest model is: {accuracy_rf:.4f}")
# Overall accuracy rate of the random forest model is 0.92.
#(b) Which two features (variables) have top feature imporatnce values? Is one of these used in the single decision tree plot in the prior problem?
top_two_features = X.columns[indices[:2]].tolist()
print(f"The two features with the highest importance values are: {top_two_features[0]} and {top_two_features[1]}")
# x22 and x37 have top feature importance values. Both x22 and x37 are used in the single decision tree plot, with x37 being used in the first split.
#(c) Discuss which two digits have least amount of accuracies and explain why it makes sense or not.
# Looking at the confusion matrix, to me it appears that the two digits with the least amount of accuracies are 3 and 8.
# This would make sense, because they are both visually similar and it could be hard to distinguish between the two.
# (Also, there is no random_state being used in RandomForestClassifier(), but we are using random_state=461 in train_test_split().)
# (Changing this will yeild slightly different results in terms of the splits, but accuracy is barely different.)

# Use GradientBoostingClassifier(n_estimators=1000, max_depth=4) to find a prediction model of the data.
gbc = GradientBoostingClassifier(n_estimators=1000, max_depth=4)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)

cm_gbc = confusion_matrix(
    y_test,
    y_pred_gbc,
    labels=gbc.classes_
    )
disp_gbc = ConfusionMatrixDisplay(confusion_matrix=cm_gbc, display_labels=gbc.classes_)
disp_gbc.plot()
plt.show()

gbc_importances = gbc.feature_importances_
gbc_indices = np.argsort(gbc_importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Gradient Boosting Classifier")
plt.bar(range(X.shape[1]), gbc_importances[gbc_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[gbc_indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

report_gbc = classification_report(
    y_test,
    y_pred_gbc,
    target_names=names, # Earlier defined class names variable
    zero_division=0
    )
print("Classification Report for Gradient Boosting Classifier:")
print(report_gbc)

# Looking at the confusion matrix of the GBC, it now appears the two digits that it has the most difficulty with are 3 and 6.
# I assume this is also due to them being visually similar, like 3 and 8.
# Looking at the feature importance plot, the top three features are now x22, x43, and x44.
# From the classification report for the GBC, the overall accuracy rate has now increased to 0.98.

# Use a LogisticRegression(max_iter=1000) method to fit the data.
logit = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class='auto') # No specified values for solver given, nor random_state.
logit.fit(X_train, y_train)
y_pred_logit = logit.predict(X_test)

cm_logit = confusion_matrix(
    y_test,
    y_pred_logit,
    labels=logit.classes_
    )
disp_logit = ConfusionMatrixDisplay(confusion_matrix=cm_logit, display_labels=logit.classes_)
disp_logit.plot()
plt.show()

# For feature importance plot, LogisticRegression() does not have built in feature_importances_ function
# Using logit.coef_ doesn't work for plotting as it is a 2D array, indexing it will give an error.
# Instead we have to sum abs vals of coefficients across all classes then sort them.
logit_importances = np.mean(np.abs(logit.coef_), axis=0) # Aggregate feature importances across all classes
logit_indices = np.argsort(logit_importances)[::-1] # Sort indices based on the aggregated importances

plt.figure(figsize=(10, 6))
plt.title("Feature Importances for Logistic Regression")
plt.bar(range(X.shape[1]), logit_importances[logit_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[logit_indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

report_logit = classification_report(
    y_test,
    y_pred_logit,
    target_names=names, # Earlier defined class names variable
    zero_division=0
    )
print("Classification Report for Logistic Regression:")
print(report_logit)

# Looking at the confusion matrix for logistic regression, it appears that much like GBC it also has the most difficulty with 3 and 6.
# Overall accuracy score has dropped slightly from GBC model to 0.96.
# Feature importance plot shows top three features as x43, x44, and x39. x43 and x44 were also in the top for GBC, but not x39.

# Run RF method again this time with max_depth=10. Compare the overall accuracy to the Logistic Regression model.
rf2 = RandomForestClassifier(n_estimators=1000, max_depth=10)
rf2.fit(X_train, y_train)
y_pred_rf2 = rf2.predict(X_test)

cm_rf2 = confusion_matrix(
    y_test,
    y_pred_rf2,
    labels=rf2.classes_
    )
disp_rf2 = ConfusionMatrixDisplay(confusion_matrix=cm_rf2, display_labels=rf2.classes_)
disp_rf2.plot()
plt.show()

report_rf2 = classification_report(
    y_test,
    y_pred_rf2,
    target_names=names, # Earlier defined class names variable
    zero_division=0
    )
print("Classification Report for Random Forest with max_depth=10:")
print(report_rf2)

# Compare the overall accuracy to the Logistic Regression model
accuracy_rf2 = accuracy_score(y_test, y_pred_rf2)
print(f"The overall accuracy rate of the Random Forest model with max_depth=10 is: {accuracy_rf2:.2f}")

# The new RFC model with max_depth=10 has an overall accuracy score of 0.98, similar to what we saw with the GBC model.
# This is sligtly higher than the logit models overall accuracy of only 0.96.