import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os

# Import CHD.csv from MTH 461 Midterm folder
#df = pd.read_csv('/Users/owen/Desktop/School 24-25/MTH 461/Midterm/CHD.csv') # File path for OS X
df = pd.read_csv('/home/oren/Desktop/School 24-25/MTH 461/Midterm/CHD.csv') # File path for Linux

# 1) Check for missing values with isnull
print(f'The number of missing cells is {df.isnull().sum().sum()}')
print(df.head())

# Zero missing cells

# Quantify family history: Present=1, Absent=0
df['famhist'] = df['famhist'].map({'Present': 1, 'Absent': 0}) # Map Present to 1, Absent to 0
X = df[['adiposity']] # X is the adiposity column
y = df['ldl'] # y is the ldl column

# Split the data into train and test in the ratio using the train_test_split() module using random_state=461
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=461)
# 2) Find the shapes of the train and test data
print(f'The shape of X_train is {X_train.shape}')
print(f'The shape of X_test is {X_test.shape}')
print(f'The shape of y_train is {y_train.shape}')
print(f'The shape of y_test is {y_test.shape}')

# 3) Find the mean and variance values of ldl of the train and test data, respectively.
print(f'The mean of ldl y_train is {np.mean(y_train)}')
print(f'The variance of ldl y_train is {np.var(y_train)}')
print(f'The mean of ldl y_test is {np.mean(y_test)}')
print(f'The variance of ldl y_test is {np.var(y_test)}')

# 4) Let x=adiposity and y=ldl.
# Fit the six decision tree models on the train data:
# Linear Regression, Decision Tree, Bagging, Random Forest, Gradient Boosting, and XGBoosting.
# Provide a 3x2 grid plot of the test data and the fitted model.

# Linear Regression
lin_reg = LinearRegression() # Linear regression model
lin_reg.fit(X_train, y_train) # Fit the model on the train data
y_pred_lin_reg = lin_reg.predict(X_test) # Predict the ldl values of the test data using the trained model
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg) # Mean squared error of the test data
r2_lin_reg = r2_score(y_test, y_pred_lin_reg) # R^2 (coefficiant of determination) score of the test data

# Decision Tree
d = 2 # Max depth value
reg_tree = DecisionTreeRegressor(max_depth=d) # Decision tree model
reg_tree.fit(X_train, y_train)
y_pred_reg_tree = reg_tree.predict(X_test)
mse_reg_tree = mean_squared_error(y_test, y_pred_reg_tree)
r2_reg_tree = r2_score(y_test, y_pred_reg_tree)

# Bagging with Decision Tree
bagging = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=d), n_estimators=100)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# Random Forest
random_forest = RandomForestRegressor(max_depth=d, n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Gradient Boosting
gradient_boosting = GradientBoostingRegressor(n_estimators=100, max_depth=d)
gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# XGBoosting
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_reg.fit(X_train, y_train)
y_pred_xg = xgb_reg.predict(X_test)
mse_xg = mean_squared_error(y_test, y_pred_xg)
r2_xg = r2_score(y_test, y_pred_xg)

# Comparison table
df_compare = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Bagging', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'MSE': [mse_lin_reg, mse_reg_tree, mse_bagging, mse_rf, mse_gb, mse_xg],
    'R^2': [r2_lin_reg, r2_reg_tree, r2_bagging, r2_rf, r2_gb, r2_xg]
})
print(df_compare)

# In order to provide 3x2 grid plot of the test data and the fitted model we need to properly sort the data.
# Our adiposity values are unsorted, so if we plot the unsorted data the plot will chase the points around from first to second, etc.
# This reults in jagged lines and a poor visual representation of the data.

# Sort the data
sorted_indices = np.argsort(X_test['adiposity']) # Sort the indices of the test data
X_test_sorted = X_test.iloc[sorted_indices] # Sort the test data
y_test_sorted = y_test.iloc[sorted_indices] # Sort the test data
y_pred_lin_reg_sorted = y_pred_lin_reg[sorted_indices]
y_pred_reg_tree_sorted = y_pred_reg_tree[sorted_indices]
y_pred_bagging_sorted = y_pred_bagging[sorted_indices]
y_pred_rf_sorted = y_pred_rf[sorted_indices]
y_pred_gb_sorted = y_pred_gb[sorted_indices]
y_pred_xg_sorted = y_pred_xg[sorted_indices]

# Plot the data and models over 3x2 grid in a single figure using axes
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

plt.suptitle('Comparison of Models')
axes[0, 0].scatter(X, y, label='CHD Data', color='blue', alpha=.5) # Scatter plot of the CHD data
axes[0, 0].plot(X_test_sorted, y_pred_lin_reg_sorted, label='Linear Regression', linestyle='dashed', color='red', linewidth=2) # Plot the linear regression model
axes[0, 0].set_xlabel('Adiposity') # Set the x-axis label
axes[0, 0].set_ylabel('LDL') # Set the y-axis label
axes[0, 0].legend()

axes[0, 1].scatter(X, y, label='CHD Data', color='blue', alpha=.5)
axes[0, 1].plot(X_test_sorted, y_pred_reg_tree_sorted, label='Decision Tree', linestyle='dashed', color='red', linewidth=2) # Plot the decision tree model
axes[0, 1].set_xlabel('Adiposity')
axes[0, 1].set_ylabel('LDL')
axes[0, 1].legend()

axes[1, 0].scatter(X, y, label='CHD Data', color='blue', alpha=.5)
axes[1, 0].plot(X_test_sorted, y_pred_bagging_sorted, label='Bagging', linestyle='dashed', color='red', linewidth=2) # Plot the bagging model
axes[1, 0].set_xlabel('Adiposity')
axes[1, 0].set_ylabel('LDL')
axes[1, 0].legend()

axes[1, 1].scatter(X, y, label='CHD Data', color='blue', alpha=.5)
axes[1, 1].plot(X_test_sorted, y_pred_rf_sorted, label='Random Forest', linestyle='dashed', color='red', linewidth=2) # Plot the random forest model
axes[1, 1].set_xlabel('Adiposity')  # Set the x-axis label
axes[1, 1].set_ylabel('LDL')  # Set the y-axis label
axes[1, 1].legend()

axes[2, 0].scatter(X, y, label='CHD Data', color='blue', alpha=.5)
axes[2, 0].plot(X_test_sorted, y_pred_gb_sorted, label='Gradient Boosting', linestyle='dashed', color='red', linewidth=2) # Plot the gradient boosting model
axes[2, 0].set_xlabel('Adiposity')
axes[2, 0].set_ylabel('LDL')
axes[2, 0].legend() # Show the legend

axes[2, 1].scatter(X, y, label='CHD Data', color='blue', alpha=.5)
axes[2, 1].plot(X_test_sorted, y_pred_xg_sorted, label='XGBoost', linestyle='dashed', color='red', linewidth=2) # Plot the XGBoost model
axes[2, 1].set_xlabel('Adiposity')  # Set the x-axis label
axes[2, 1].set_ylabel('LDL')  # Set the y-axis label
axes[2, 1].legend()  # Show the legend

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()

# Plot trees of bagging, random forest, and gradient boosting with feature names over 3x1 grid in a single figure using axes
fig, axes = plt.subplots(3, 1, figsize=(9, 15))

# the last tree in bagging
plot_tree(bagging.estimators_[-1], feature_names=X.columns.tolist(), filled=True, rounded=True, ax=axes[0])
axes[0].set_title('Last Tree in Bagging')

# the last tree in random forest
plot_tree(random_forest.estimators_[-1], feature_names=X.columns.tolist(), filled=True, rounded=True, ax=axes[1])
axes[1].set_title('Last Tree in Random Forest')

# the last tree in gradient boosting
plot_tree(gradient_boosting.estimators_[-1][0], feature_names=X.columns.tolist(), filled=True, rounded=True, ax=axes[2])
axes[2].set_title('Last Tree in Gradient Boosting')

plt.show()

# 5) Include all the potential predictors (besides ldl that is the response), use random forest model with max_depth=3 to provide the plot of the feature importances.
# According to this feature selection model, indicate the three predictors in order with most effects on the response variable.

# Get all the potential predictors
X = df.drop(columns=['ldl']) # Drop the ldl column
y = df['ldl'] # ldl is the response variable

# Train Random Forest model
random_forest_feat = RandomForestRegressor(max_depth=3, n_estimators=100, random_state=461)
random_forest_feat.fit(X, y)

# Get feature importances
importances = random_forest_feat.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Top three predictors
top_three_predictors = [X.columns[i] for i in indices[:3]]
print(f"Top three predictors in order with most effects on the response variable: {top_three_predictors}")

# 6) Use the three predictors (adiposity, alcohol, and obsesity) to fit the six tree models on the train data
# (“Linear Regression, Decision Tree, Bagging, Random Forest, Gradient Boosting, and XGBoosting”)
# To have simplicity and consistency, use max_depth=3 and n_estimators =100 for tree models.
# In all ensemble methods, use seed 461 to get consistent outcomes.

# Get the three predictors
X = df[['adiposity', 'alcohol', 'obesity']]
y = df['ldl']

# Split the data into train and test in the ratio using the train_test_split() module using random_state=461
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=461)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
r2_lin_reg = r2_score(y_test, y_pred_lin_reg)

# Decision Tree
reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_train, y_train)
y_pred_reg_tree = reg_tree.predict(X_test)
mse_reg_tree = mean_squared_error(y_test, y_pred_reg_tree)
r2_reg_tree = r2_score(y_test, y_pred_reg_tree)

# Bagging with Decision Tree
bagging = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=461)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# Random Forest
random_forest = RandomForestRegressor(max_depth=3, n_estimators=100, random_state=461)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Gradient Boosting
gradient_boosting = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=461)
gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# XGBoosting
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=461)
xgb_reg.fit(X_train, y_train)
y_pred_xg = xgb_reg.predict(X_test)
mse_xg = mean_squared_error(y_test, y_pred_xg)
r2_xg = r2_score(y_test, y_pred_xg)

# a) Evaluate the mean values of the three variables (adiposity, alcohol, obesity) in the test data.
mean_adiposity = X_test['adiposity'].mean()
mean_alcohol = X_test['alcohol'].mean()
mean_obesity = X_test['obesity'].mean()

print(f"The mean value of adiposity in the test data is {mean_adiposity}")
print(f"The mean value of alcohol in the test data is {mean_alcohol}")
print(f"The mean value of obesity in the test data is {mean_obesity}")

# b) Use the Random Forest Regressor to predict ldl of the mean feature with respect the three predictors (adiposity, alcohol, obesity).
mean_features_df = pd.DataFrame({'adiposity': [mean_adiposity], 'alcohol': [mean_alcohol], 'obesity': [mean_obesity]})
predicted_ldl = random_forest.predict(mean_features_df)

print(f"The predicted LDL value for the mean features is {predicted_ldl[0]}")

# c) Use the fitted XGBoosting Regressor to predict ldl of the mean values of the three predictors of the test data.
predicted_ldl_xg = xgb_reg.predict(mean_features_df)

print(f"The predicted LDL value for the mean features using XGBoosting is {predicted_ldl_xg[0]}")

# d) Provide the plot of the last Random Forest Regressor tree.
# And, give interpretations of the tree top to bottom narrative details in terms of the predictors.
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from xgboost import plot_tree as xgb_plot_tree

# Plot the last tree in the Random Forest Regressor
plt.figure(figsize=(20, 10))
plot_tree(random_forest.estimators_[-1], feature_names=['adiposity', 'alcohol', 'obesity'], filled=True, rounded=True)
plt.title('Last Tree in Random Forest Regressor')
plt.show()

# e) Provide the plot of the last Gradient Boosting tree
plt.figure(figsize=(20, 10))
plot_tree(gradient_boosting.estimators_[-1][0], feature_names=['adiposity', 'alcohol', 'obesity'], filled=True, rounded=True)
plt.title('Last Tree in Gradient Boosting Regressor')
plt.show()

# Extra: Provide the plot of the last XGBoosting tree
#plt.figure(figsize=(20, 10))
#xgb_plot_tree(xgb_reg, num_trees=xgb_reg.best_iteration)
#plt.title('Last Tree in XGBoosting Regressor')
#plt.show()

# 7) Provide a dataframe output of SSE, MSE, bias, and R^2 of the six models on the test data.
models = [lin_reg, reg_tree, bagging, random_forest, gradient_boosting, xgb_reg]
titles = ['Linear Regression', 'Decision Tree', 'Bagging', 'Random Forest', 'Gradient Boosting', 'XGBoost']
SSE = []
MSE = []
bias = []
R2 = []
for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    SSE.append(np.sum((y_pred - y_test)**2))
    MSE.append(SSE[i]/len(y_test))
    bias.append(np.mean(y_pred - y_test))
    R2.append(r2_score(y_test, y_pred))

df_output = pd.DataFrame({
    'Model': titles,
    'SSE': SSE,
    'MSE': MSE,
    'Bias': bias,
    'R^2': R2
})
print(df_output)