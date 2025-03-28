import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

G1 = [45, 50, 47, 46, 51]
G2 = [48, 52, 49, 55, 53]
G3 = [42, 44, 46, 43, 47]
n = 5

# Don't simulate data!

# Print the group means
print(f"Group means are {np.mean(G1)}, {np.mean(G2)}, {np.mean(G3)}")

# Print the grand mean
Y = np.concatenate([G1, G2, G3]) # Concatenate the data
print(f"The grand mean is {np.mean(Y)}")

# Use f_oneway to do ANOVA
f_stat, p_value = f_oneway(G1, G2, G3)
print('f =', f_stat, 'p-value =', p_value) # Print f-statistic and p-value
X = ['G1']*n + ['G2']*n + ['G3']*n # Create the group labels
df = pd.DataFrame({'X': X, 'Y': Y}) # Create the dataframe

# Use ols to do ANOVA
model = ols('Y ~ X', data=df).fit() # Fit the model
anova_table = sm.stats.anova_lm(model, typ=2) # Generate the ANOVA table
print(anova_table) # Print the ANOVA table

# Set up critical region at alpha = 0.05 level of significance
alpha = 0.05
dfn = anova_table['df']['X'] # Equal to 2
dfd = anova_table['df']['Residual'] # Equal to 12
critical_value = 1-scipy.stats.f.ppf(alpha, dfn, dfd)
print(f"Critical value at alpha = {alpha} is {critical_value}")

if f_stat > critical_value:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# MSB and MSE (mean square between and mean square error)
MSB = anova_table['sum_sq']['X']/anova_table['df']['X']
MSE = anova_table['sum_sq']['Residual']/anova_table['df']['Residual']
print(f"MSB is {MSB} and MSE is {MSE}")

# Add MSB and MSE to the ANOVA table
anova_table.insert(2, 'MS', [MSB, MSE])
print(anova_table)

# Total Sum of Squares (TSS)
TSS = np.sum((Y - np.mean(Y))**2)
print(f"Total sum of squares is {TSS}")

# Sum of Squares Between (SSB) and Sum of Squares Error (SSE)
SSB = anova_table['sum_sq']['X']
SSE = anova_table['sum_sq']['Residual']
print(f"SSB is {SSB} and SSE is {SSE}")
print(f"SSB + SEE is {SSB + SSE}")
print(f"TSS == SSB + SSE is {TSS == SSB + SSE}")

# Coefficient of determination (R^2)
R2 = SSB / TSS
print(f"R^2 is {R2}")

# Percentages of TSS due to between-group variation and within-group variation
percent_SSB = (SSB / TSS) * 100
percent_SSE = (SSE / TSS) * 100
print(f"Percentage of TSS due to between-group variation: {percent_SSB:.2f}%")
print(f"Percentage of TSS due to within-group variation: {percent_SSE:.2f}%")

# Tukey's HSD
mc = pairwise_tukeyhsd(df['Y'], df['X'])
print(mc)
mc.plot_simultaneous()
# plt.show()