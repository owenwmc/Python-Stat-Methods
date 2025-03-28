import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import f, t

# Read in CSV file
file = '/Users/owen/Desktop/School 24-25/MTH 461/Longevity.csv'
df = pd.read_csv(file)

# How many cells have missing values in df
print(f'The number of missing cells is {df.isnull().sum().sum()}')

print(df.head())

# Map categorical variables to numerical values
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})
df['Smoke'] = df['Smoke'].map({'Smoker': 1, 'Non-Smoker': 0})
print(df.head())

# Define features and target variable
X = df[['Mother', 'Father', 'GMother', 'GFather', 'Gender', 'Smoke']]
y = df['Longevity']

# Train linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(f"Coefficients: {lin_reg.coef_}")
print(f"Intercept: {lin_reg.intercept_}")

# Predict y values
y_pred = lin_reg.predict(X)

# Degrees of freedom
df1 = len(X.columns)
df2 = len(y) - df1 - 1

# Total Sum of Squares (TSS)
TSS = np.sum((y - np.mean(y))**2)

# Sum of Squares Error (SSE)
SSE = np.sum((y - y_pred)**2)

# Sum of Squares Regression (SSR)
SSR = np.sum((y_pred - np.mean(y))**2)

# TSS decomposition
print(f'TSS = {TSS}, SSE = {SSE}, SSR = {SSR}')

# Mean Square Regression (MSR)
MSR = SSR / df1

# Mean Square Error (MSE)
MSE = SSE / df2

# Coefficient of determination (R^2)
R2 = 1 - (SSE / TSS)
print(f"Coefficient of determination (R^2) is {R2}")

# Adjusted coefficient of determination
adj_R2 = 1 - (1 - R2) * (len(y) - 1) / (len(y) - df1 - 1)
print(f"Adjusted coefficient of determination is {adj_R2}")

# F-statistic
F = MSR / MSE

# p-value
p_value = 1 - f.cdf(F, df1, df2)
print(f"p-value is {p_value}")

# ANOVA table
df_anova = pd.DataFrame({
    'source': ['model', 'error', 'total'],
    'df': [df1, df2, len(y) - 1],
    'SS': [SSR, SSE, TSS],
    'MS': [MSR, MSE, np.nan],
    'F': [F, np.nan, np.nan],
    'p-value': [p_value, np.nan, np.nan]
})
print(df_anova)

# Small p-value implies that the regression model has a strong correlation to the true y values

# Beta_hat are statistics because beta_hat=((X.T)X)^-1(X.T)Y, where X=design matrix.
# It turns out that var(beta_hat)=MSE*((X.T)X)^-1
# To test Ho:beta_i=0 or not, we compute t test statistic and its p-value. The t score is obtained by beta_hat/SE(beta_hat).
# SE(beta_hat)=sqrt(MSE*((X.T)X)^-1)
# So, let's compute t scores and p-values for the predictors

# Add intercept to design matrix
X_design = np.concatenate((np.ones((len(X), 1)), X), axis=1)

beta_hat = np.concatenate(([lin_reg.intercept_], lin_reg.coef_))
print(f"beta_hat is {beta_hat}")

se_beta = np.sqrt(MSE * np.diag(np.linalg.inv(np.dot(X_design.T, X_design))))
print(f"SE(beta_hat) is {se_beta}")

t_scores = beta_hat / se_beta
print(f"t_scores are {t_scores}")

p_values = 2 * (1 - t.cdf(np.abs(t_scores), df2))
print("t_scores and p_values are \n", pd.DataFrame({'t_scores': t_scores, 'p_values': p_values}))

# Now drop insignificant variables GMother and GFather
X_new = X.drop(['GMother', 'GFather'], axis=1)

# Train new linear regression model
lin_reg_new = LinearRegression()
lin_reg_new.fit(X_new, y)
print(f"New Coefficients: {lin_reg_new.coef_}")
print(f"New Intercept: {lin_reg_new.intercept_}")

# Predict a new data set
new_data = pd.DataFrame({'Mother': [80, 80], 'Father': [75, 75], 'Gender': [0, 1], 'Smoke': [1, 0]})
print(f"Predicted longevity is {lin_reg_new.predict(new_data)}")

