import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, f, chi2

df = pd.read_csv('/Users/owen/Desktop/School 24/MTH 361/CHD.csv') # Read in the data
#print(f"shape is {df.shape}")
#print(df.iloc[2:5,1:4])asdf.
#print(df.sbp)
#print(df['sbp'])
df.head()

## print statement were commented out to maintain cleanliness of file
print(df.head())
# print(df.tail())
# print(df[2:9])
# print(df.iloc[2:5,1:4])

# print(df.sbp)
# print(df['sbp'])#equivalent as above

#1
# how many cells have missing values in df
print(f'The number of missing cells is {df.isnull().sum().sum()}') #isnull is a function that looks at each cell for emptiness

# the list of row index for the missing values
missing_rows = df[df.isnull().any(axis=1)].index.tolist()
print(f"The list of row index for the missing values is: {missing_rows}")

# Impute the missing values by their matching column means if data was quantitative
# If the data were categorical or numeric binary, use mode to impute
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']: # Check for quantitative data types
        # Impute with mean for quantitative columns
        df[column] = df[column].fillna(df[column].mean())
    elif column in ['chd', 'famhist']: # Check for binary and categorical columns
        # Impute with mode for binary and categorical columns
        df[column] = df[column].fillna(df[column].mode()[0])

print(f"Number of missing cells after imputation is {df.isnull().sum().sum()}")

#3/4/5
x = df.sbp
TSS = np.sum((x - np.mean(x))**2)
print(f'TSS is {TSS.round(2)}')
print(f'The mean and variance are {x.mean().round(3)} and {x.var()}')
print(f'25th and 75th percentile of sbp is \n{x.quantile([0.25, 0.75]).round(3)}')

#6
#mean and variance of sbp grouped by chd
grouped_data = df.groupby('chd')['sbp'].agg(['mean', 'var'])
print(grouped_data)

#7
#25th and 75th percentile of sbp grouped by chd
grouped_data = df.groupby('chd')['sbp'].agg([lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
print(grouped_data)

df.groupby('chd')['sbp'].describe()

#8
df_subset = df[['sbp', 'ldl', 'adiposity']]
df_subset.head()
print(f'mean vector is \n {df_subset.mean()}')
print(f'covariance matrix is \n {df_subset.cov()}')

#9
print(f'one way frequency distribution of famhist is \n {df["famhist"].value_counts()}')

#10
print(f'two way frequency distribution of chd and famhist is \n {pd.crosstab(df["chd"], df["famhist"])}')

#11
print(f'percent of data having chd and famhist is {np.sum((df["chd"]==1) & (df["famhist"]=="Present"))/df.shape[0]*100}')

#12
print(f'percent of data having chd or famhist is {np.sum((df["chd"]==1) | (df["famhist"]=="Present"))/df.shape[0]*100}')

#13. Since there are 2*96 subjects with famhist and 96 of these have chd, its percent value is 50%

#14
temp = df[df.chd==1].sbp
print(np.sum((temp>120)|(temp<80))/df[df.chd==1].shape[0]*100)

#15
print(np.sum((100 < temp) & (temp < 130))/df[df.chd==1].shape[0]*100)

#16
#sns.pairplot(df.drop('IDN', axis=1))


#17 T distributions
#a
# Consider Z~N(0,1) and Chi-square with df
# the sampling distribution of Z/sqrt(Chi-square/df) is T distribution with df.
# Explain: draw a random z score and a random chi-square score with df. Then
# The sampling distribution of what Z/sqrt(chi-square/df) can be is governed by T distribution.
# T distribution is almost similar to N(0,1) distribution, except that
# T distribution is slightly more variable than N(0,1).

#b
print(t.mean(6, loc=0, scale=1)) # mean of T distribution with 10 df
print(t.var(6, loc=0, scale=1)) # variance of T distribution with 10 df

#c
print(1-t.cdf(1, 6)) # P(T>1) with 6 df

#d
# T_alpha,6 represents the upper critical value of T distribution with 6 df
# Find T_0.05,6
t.ppf(0.95, 6)

#18 About F distribution
#a. Definition
# consider two independent chi-square values say x1 and x2 with df1 and df2, respectively.
# The sampling distribution of (x1`/df1)/(x2`/df2) has F_df1,df2 distribution.

#b
print(f'mean and variance of F_2,3 are {f.mean(2, 3, loc=0, scale=1)}') # mean of F distribution with 2 and 3 df

#c
print(f"P(f_2,3>=5) is {1-f.cdf(5, 2, 3)}") # P(F>5) with 2 and 3 df

#alpha = 0.05
#F_alpha,2,3 represents the upper critical value of F distribution with 2 and 3 df
print(f"F_0.05,2,3 is {f.ppf(0.95, 2, 3)}")



# Experiment - create the list of .01 to .10 with .01 as increment
a = np.arange(0.01, 0.11, 0.01)
t.ppf(a,6) # Find the critical values of T distribution with 6 df
