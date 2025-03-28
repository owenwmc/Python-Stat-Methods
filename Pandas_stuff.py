import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.normal(0, 1, 10000) # Generate 10000 random numbers from a normal distribution with mean 0 and standard deviation 1

x = pd.Series(x) # Convert the numpy array to a pandas Series
x = x.rename("X") # Rename the Series to "X"
x.describe() # Display summary statistics of the Series
x.quantile([0.1, 0.9]) # Calculate the quartiles of the Series

y = np.random.choice(x, 100, replace=True) # Draw a random sample of size 10000 with replacement from the Series


#Create a dataframe using dictionaries
data = {
    "Name": ['Mike', 'Susan', 'Vincent'],
    "Age": [40, 50, 60],
    "Gender": ['M', 'F', 'M']
}
df=pd.DataFrame(data)
print(df.describe())
print('Gender', df["Gender"].unique()) # Display summary statistics of the DataFrame