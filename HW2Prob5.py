from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

# Given parameters
alpha = 1
beta_param = 5
n = 75
k = 8

# Update the parameters for the posterior distribution
alpha_post = alpha + k
beta_post = beta_param + n - k

# Calculate the CDF at 0.1
cdf_0_1 = beta.cdf(0.1, alpha_post, beta_post)

# Calculate the probability that X > 0.1
prob_X_greater_than_0_1 = 1 - cdf_0_1

print(f"Probability that X > 0.1: {prob_X_greater_than_0_1}")
