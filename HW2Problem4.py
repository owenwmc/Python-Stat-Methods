from scipy.stats import beta
import numpy as np

a1 = 1.5  # Define the first shape parameter
a2 = 3.5  # Define the second shape parameter
Beta = beta(a1, a2)  # Assign the beta distribution to Beta
x = Beta.rvs(size=10000, random_state=123)  # Generate 10000 random samples from the beta distribution

# Calculate the probability that x is greater than 0.5
prob_greater_than_0_5 = 1 - Beta.cdf(0.5)
percentile_95 = Beta.ppf(0.95)

print(f"mean = {np.mean(x)}")  # Calculate the sample mean
print(f"sec moment = {np.mean(x**2)}")  # Calculate the sample second moment
print(f"std = {np.std(x)}")  # Calculate the sample standard deviation
print(f"var = {np.var(x)}")  # Calculate the sample variance
print(f"Probability that x > 0.5 = {prob_greater_than_0_5}")  # Print the probability
print(f"95th percentile = {percentile_95}")  # Print the 95th percentile