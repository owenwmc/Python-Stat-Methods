from scipy.stats import norm
import numpy as np

# Define the normal distribution
X = norm(loc=1500, scale=100)

cdf_1500 = X.cdf(1500)
cdf_1700 = X.cdf(1700)

probability = cdf_1700 - cdf_1500
percentile_90 = X.ppf(0.90)

print(probability)  # Probability that X is between 1500 and 1700
print(percentile_90)  # 90th percentile of the standard normal distribution