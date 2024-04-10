import pandas as pd
from sklearn.datasets import load_diabetes


# Load dataset from sklearn
X,y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)

# Convert to pandas dataframe
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# # Standardize data
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()

# Rename columns
X.rename(columns={'s1':'tc', 's2': 'ldl', 's3': 'hdl', 's4': 'tch', 's5': 'ltg', 's6': 'glu'}, inplace=True)

# Add Offset
X['Offset'] = 1

# Move offset to first position
cols = X.columns.tolist()
cols = cols[-1:] + cols[:-1]
X = X[cols]

# Number of samples and features: N = samples, M = features
N, M = X.shape

# attribute names
attributeNames = X.columns

# Convert pandas dataframe to numpy array
X = X.to_numpy()
y = y.to_numpy().flatten()