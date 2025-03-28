import numpy as np

# Set the random seed for reproducibility
np.random.seed(123)

# Define the probabilities for each color
probabilities = [0.2, 0.3, 0.5]  # Example probabilities for red, green, and blue

# Simulate drawing a sample of size 20 with replacement
sample_size = 10
sample = np.random.choice(['red', 'green', 'blue'], size=sample_size, p=probabilities)

# Count the frequencies of each color in the sample
unique, counts = np.unique(sample, return_counts=True)
frequencies = dict(zip(unique, counts))

# Print the frequencies in the order of red, green, and blue
print("Red:", frequencies.get('red', 0))
print("Green:", frequencies.get('green', 0))
print("Blue:", frequencies.get('blue', 0))

# Define the prior parameters for the Dirichlet distribution
alpha_prior = np.array([2, 2, 2])  # Example prior parameters for red, green, and blue

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
