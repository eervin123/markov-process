# %% Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the data
file_path = 'data/btc_daily_data.csv'  # Updated file path
data = pd.read_csv(file_path)
#set columns to lowercase
data.columns = [column.lower() for column in data.columns]
# Data Preprocessing
data['close'] = pd.to_numeric(data['close'], errors='coerce')
data = data.dropna(subset=['close'])

# Calculate the logarithmic returns
data['log_return'] = np.log(data['close'] / data['close'].shift(1))
data['log_return'] = data['log_return'].fillna(0)

# Create features
data['high_low'] = data['high'] - data['low']
data['close_open'] = data['close'] - data['open']
data['volume_change'] = data['volume'].diff()
data['volume_change'] = data['volume_change'].fillna(0)

# Create additional features
data['ema_fast'] = data['close'].ewm(span=12, adjust=False).mean()
data['ema_slow'] = data['close'].ewm(span=26, adjust=False).mean()
data['change_1'] = data['close'].pct_change(1)
data['change_5'] = data['close'].pct_change(5)
data['change_10'] = data['close'].pct_change(10)
data['change_20'] = data['close'].pct_change(20)
data['change_30'] = data['close'].pct_change(30)
data['change_50'] = data['close'].pct_change(50)
data['change_100'] = data['close'].pct_change(100)
data = data.fillna(0)

# %% Bin the states based on the trailing 5 weeks
five_weeks = 25
quantiles = data['log_return'].tail(five_weeks).quantile([0.25, 0.5, 0.75])
bins = [-np.inf, quantiles[0.25], quantiles[0.5], quantiles[0.75], np.inf]
labels = ['Large Down', 'Small Down', 'Small Up', 'Large Up']
data['return_bin'] = pd.cut(data['log_return'], bins=bins, labels=labels)

# Map the bins to integers
bin_mapping = {'Large Down': 0, 'Small Down': 1, 'Small Up': 2, 'Large Up': 3}
data['return_bin'] = data['return_bin'].map(bin_mapping)

# Split the data into training and validation sets
train_data, validation_data = train_test_split(data['return_bin'], test_size=0.2, shuffle=False)

# Define the HMM model with the desired parameters
model = hmm.MultinomialHMM(n_components=4, n_iter=1000, random_state=42)

# Train the model on the training data
train_data = np.array(train_data).reshape(-1, 1)
logging.info("Training the model...")
model.fit(train_data)  # No loop needed, fit will iterate n_iter times

# Check for convergence
if not model.monitor_.converged:
    logging.warning('Model not converged. Increasing n_iter or changing initialization parameters may help.')

# Evaluate the model on the validation data
validation_data = np.array(validation_data).reshape(-1, 1)
hidden_states = model.predict(validation_data)
log_likelihood = model.score(validation_data)
logging.info(f"Log likelihood: {log_likelihood}")

# %%
