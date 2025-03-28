import numpy as np
from scipy.stats import beta, dirichlet

# Define the parameters for the Dirichlet distribution
alpha_prior = np.array([2, 2, 2])

# Calculate the mean of the Dirichlet distribution
mean = dirichlet.mean(alpha_prior)

# Calculate the variance of the Dirichlet distribution
variance = dirichlet.var(alpha_prior)

print("Mean of the Dirichlet distribution:", mean)
print("Variance of the Dirichlet distribution:", variance)

# Set the random seed for reproducibility
np.random.seed(123)

# Define the proportions for red, green, and blue marbles
proportions = np.array([1/3, 1/3, 1/3])

# Draw a random sample of size 10 with replacement
sample_size = 10
sample = np.random.choice(['red', 'green', 'blue'], size=sample_size, p=proportions)

# Count the frequencies of each color in the sample
unique, counts = np.unique(sample, return_counts=True)
frequencies = dict(zip(unique, counts))

# Print the frequencies in the order of red, green, and blue
print("Frequencies of the outcomes:")
print("Red:", frequencies.get('red', 0))
print("Green:", frequencies.get('green', 0))
print("Blue:", frequencies.get('blue', 0))

import numpy as np

# Observed counts from the sample
counts = np.array([frequencies.get('red', 0), frequencies.get('green', 0), frequencies.get('blue', 0)])

# Update the posterior parameters
alpha_posterior = alpha_prior + counts

# Calculate the mean of the posterior distribution
mean_posterior = alpha_posterior / np.sum(alpha_posterior)

# Calculate the variance of the posterior distribution
total_alpha = np.sum(alpha_posterior)
variance_posterior = (alpha_posterior * (total_alpha - alpha_posterior)) / (total_alpha**2 * (total_alpha + 1))

print("Posterior mean:", mean_posterior)
print("Posterior variance:", variance_posterior)

