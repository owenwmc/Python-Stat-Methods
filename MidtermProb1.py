import matplotlib.pyplot as plt
import numpy as np

# Define the likelihood
def Likelihood_Bernoulli(p, data):
    return p**sum(data) * (1-p)**(len(data)-sum(data)) # X bar is the sum of the data

# Define the log likelihood
def log_Likelihood_Bernoulli(p, data):
    return sum(data) * np.log(p) + (len(data)-sum(data)) * np.log(1-p)

#Given data set from the problem
data = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1])

#Calculate the MLE
phat = np.mean(data)

#Print the results
print('phat:', phat)
print('Likelihood:', Likelihood_Bernoulli(phat, data))
print('Log Likelihood:', log_Likelihood_Bernoulli(phat, data))

#Generate varying values of p
p = np.arange(0.01, 1, 0.01)

#Create a plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#Plot the likelihood
ax1.plot(p, Likelihood_Bernoulli(p, data))
ax1.set_xlabel('p')
ax1.set_title('Likelihood')
ax1.axvline(phat, color='red', linestyle='--') #Add a vertical line at the MLE

#Plot the log likelihood
ax2.plot(p, log_Likelihood_Bernoulli(p, data))
ax2.set_xlabel('p')
ax2.set_title('Log Likelihood')
ax2.axvline(phat, color='red', linestyle='--') #Add a vertical line at the MLE

plt.show()