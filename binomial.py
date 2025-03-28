from scipy.stats import binom

# Example parameters
n = 30  # Number of trials
p = 0.53  # Probability of success

# Probability Mass Function (PMF) at x = 2
pmf_value = binom.pmf(2, n, p)

# Cumulative Distribution Function (CDF) up to x = 24
cdf_value = binom.cdf(24, 30, 0.53)

# Mean and Variance
mean, var = binom.mean(n, p), binom.var(n, p)

# Random Variates (e.g., generate 10 random values)
random_variates = binom.rvs(n, p, size=10)

print(f"PMF at x = 2: {pmf_value}")
print(f"CDF up to x = 24: {cdf_value}")
print(f"Mean: {mean}, Variance: {var}")
print(f"Random Variates: {random_variates}")