import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/owen/Desktop/School 24-25/MTH 461/boston_housing.csv')
X = df.drop('MEDV', axis=1)
y = df['MEDV']
print(X.shape)
print(y.shape)

#Partition the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#fit the six models

# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
SSE_lin_reg = np.sum((y_pred_lin_reg - y_test)**2)
MSE_lin_reg = SSE_lin_reg/len(y_test)

# Regression Tree
reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_train, y_train) # Fit
y_pred_reg_tree = reg_tree.predict(X_test) # Predicted values
print(y_pred_reg_tree)
SSE_reg_tree = np.sum((y_pred_reg_tree - y_test)**2)
MSE_reg_tree = SSE_reg_tree/len(y_test)

# Bagging
bag_reg = BaggingRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=100, bootstrap=True, n_jobs=-1)
#bag_reg = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, bootstrap=True, oob_score=True, n_jobs=-1)
bag_reg.fit(X_train, y_train)
y_pred_bag = bag_reg.predict(X_test)
SSE_bag = np.sum((y_pred_bag - y_test)**2)
MSE_bag = SSE_bag/len(y_test)

# Random Forest
rand_forest = RandomForestRegressor(n_estimators=100, max_depth=2, bootstrap=True, n_jobs=-1)
rand_forest.fit(X_train, y_train)
y_pred_rand_forest = rand_forest.predict(X_test)
SSE_rand_forest = np.sum((y_pred_rand_forest - y_test)**2)
MSE_rand_forest = SSE_rand_forest/len(y_test)

# Gradient Boosting
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2)
gb_reg.fit(X_train, y_train)
y_pred_gb = gb_reg.predict(X_test)
SSE_gb = np.sum((y_pred_gb - y_test)**2)
MSE_gb = SSE_gb/len(y_test)

# XGBoost
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=2)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)
SSE_xgb = np.sum((y_pred_xgb - y_test)**2)
MSE_xgb = SSE_xgb/len(y_test)

# Plot the data and the six models over 3x2 grid in one graph
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
fig.suptitle('Regression Models')

models = [lin_reg, reg_tree, bag_reg, rand_forest, gb_reg, xgb_reg]
titles = ['Linear Regression', 'Regression Tree', 'Bagging', 'Random Forest', 'Gradient Boosting', 'XGBoost']
for i, ax in enumerate(axs.flat): #outcome of this is two arguments
    model = models[i]
    title = titles[i]
    ax.scatter(X_test, y_test, color='black')
    ax.plot(X, model.predict(X_test), color='red', linewidth=2)
    ax.set_title(titles[i])
    ax.set_xlabel('X')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()

# In Dr. Kim's graph, boost graphs are perfectly fitted

#SSE, MSE, bias, and R^2 of the six models using for loop and print outcomes in dataframe
models = [lin_reg, reg_tree, bag_reg, rand_forest, gb_reg, xgb_reg]
titles = ['Linear Regression', 'Regression Tree', 'Bagging', 'Random Forest', 'Gradient Boosting', 'XGBoost']
SSE = []
MSE = []
bias = []
R2 = []
for i, model in enumerate(models):
    y_pred = model.predict(X.reshape(-1, 1))
    SSE.append(np.sum((y_pred - y)**2))
    MSE.append(SSE[i]/len(y))
    bias.append(np.mean(y_pred - y))
    R2.append(r2_score(y, y_pred))
    #SSE = np.sum((y_pred - y)**2)
    #MSE = SSE/len(y)
    #bias = np.mean(y_pred - y)
    #R2 = r2_score(y, y_pred)
    #R2 =1 - SSE / np.sum((y - np.mean(y))**2) #TSS model

df_output = pd.DataFrame({'Model': titles, 'SSE': SSE, 'MSE': MSE, 'Bias': bias, 'R2': R2})
print(df_output)