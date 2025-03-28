### Chapter 1: ANOVA
# Import necessary libraries and functions

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, f
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# In order to simulate data
np.random.seed(123)
mu1 = [45, 50, 47, 46, 51]
mu2 = [48, 52, 49, 55, 53]
mu3 = [42, 44, 46, 43, 47]
sigma = 3
n = 5 # For each group, so balanced data
G1 = np.random.normal(mu1[0], sigma, n)
G2 = np.random.normal(mu2[1], sigma, n)
G3 = np.random.normal(mu3[2], sigma, n)

# Print the group means rounded to 2 decimal places
print(f"Group means are {np.mean(G1):.2f}, {np.mean(G2):.2f}, {np.mean(G3):.2f}")

# Print the grand mean
grand_mean = np.mean([np.mean(G1), np.mean(G2), np.mean(G3)])
print("Grand mean is ", grand_mean)

# Boxplots of these three
plt.boxplot([G1, G2, G3])
#print(plt.show())

# Use f_oneway to do basic ANOVA
f, p = f_oneway(G1, G2, G3) # This will generate two outcomes - the F-statistic and the p-value
print('f =', f, 'p-value =', p)

# This time I want to use statsmodels to have more details
# For this end, I want to create a dataframe of the data
Y = np.concatenate([G1, G2, G3]) # Concatenate the data
print(Y)
X = ['G1']*n + ['G2']*n + ['G3']*n # Create the group labels
df = pd.DataFrame({'X': X, 'Y': Y}) # Create the dataframe

# Now use ols to do ANOVA
model = ols('Y ~ X', data=df).fit() # Fit the model
anova_table = sm.stats.anova_lm(model, typ=2) # Generate the ANOVA table
print(anova_table) # Print the table
print("Data type of anova_table is ", type(anova_table))

# MSB and MSE (mean square between and mean square error)
MSB = anova_table['sum_sq'][0]/(anova_table['df'][0]-1) # Subtract 1?
MSE = anova_table['sum_sq'][1]/(anova_table['df'][1]-1) # "

#anova_table['MS'] = [MSB, MSE] # Add these to the table
anova_table.insert(2, 'MS', [MSB, MSE])
print(anova_table)

# Calculate total amount of response values
TSS = np.sum((Y - np.mean(Y))**2)
print("Total sum of squares is ", TSS)
SSB = anova_table['sum_sq'][0]
SSE = anova_table['sum_sq'][1]
#print(TSS == SSB + SSE)
print('SSB + SEE is', SSB + SSE)

# Coefficient of determination
R2 = SSB/TSS
print("coeff of determination is ", R2)

# Tukey HSD plot
mc = pairwise_tukeyhsd(df['Y'], df['X'])
print(mc)
mc.plot_simultaneous()
plt.show()

# Set up critical region at alpha = 0.05
alpha = 0.05
dfn = 2
dfd = 12
f_crit = f.ppf(1-alpha, dfn, dfd)
print("Critical value is ", f_crit)

# Compare the critical value with the F-statistic
if f > f_crit:
    print("Reject the null hypothesis: There is a significant difference between the group means.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the group means.")