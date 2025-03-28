from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
a1=1.5 # Define the first shape parameter
a2=3.5 # Define the second shape parameter
Beta=beta(a1,a2) # Assign the beta distribution to Beta
x=Beta.rvs(size=10000,random_state=123) # Generate 10000 random samples from the beta distribution
print(f"mean = {np.mean(x)}") # Calculate the sample mean
print(f"sec moment = {np.mean(x**2)}") # Calculate the sample second moment
print(f"std = {np.std(x)}") # Calculate the sample standard deviation
print(f"var = {np.var(x)}") # Calculate the sample variance
x[:10] # Display the first 10 samples
len(x[x<Beta.mean()])/len(x) # Calculate the probability of x < mean
print(f"var={Beta.var()}") # Calculate the variance of the beta distribution
print(f"sample variance= {np.var(x)}") # Calculate the sample variance
print(f"95th percentile={Beta.ppf([.95,a1,a2])}") # Calculate the 95th percentile of the beta distribution
print(f"sample 10th and 90th= {np.percentile(x,[.1,.9])}") # Calculate the 10th and 90th percentiles of the sample
Prob=beta.cdf(0.5,a1,a2) # Calculate the probability that x < 0.5
Prob_X_greater_than_0_5=1-Prob # Calculate the probability that x > 0.5
print(f"Prob_X_greater_than_0_5={Prob_X_greater_than_0_5}") # Display the probability that x > 0.5
