# %%
import warnings

warnings.filterwarnings("ignore")

import vectorbtpro as vbt
import numpy as np
import pandas as pd

vbt.settings.set_theme("dark")
vbt.settings.plotting["layout"]["width"] = 800
vbt.settings.plotting["layout"]["height"] = 200

import pandas_ta as ta


# %% [markdown]
# # Import the data

# %%
btc_90M_db_vbt = vbt.BinanceData.load("data/btc_90M_db_vbt.pkl")

data = btc_90M_db_vbt["2021-01-01":"2023-01-01"]
outofsample_data = btc_90M_db_vbt["2023-01-01":"2023-06-03"]
print(data.shape)
print(outofsample_data.shape)
# Wherever you saved the pickle file
data_path = "/Users/ericervin/Documents/Coding/data-repository/data/fixed_BTCUSDT.csv"
# min_data = vbt.BinanceData.from_csv(data_path)
# print(min_data.shape)

# %%
markov_df = data.get()
markov_df["pct_change"] = markov_df["Close"].pct_change()
markov_df["log_ret"] = np.log(markov_df["Close"]).diff()
markov_df["volatility"] = markov_df["log_ret"].rolling(100).std() * np.sqrt(100)
markov_df["volume_chg"] = markov_df["Volume"].pct_change()
markov_df["current_state"] = markov_df["pct_change"].apply(
    lambda x: 1 if x > 0 else 0
)  # 1 for up, 0 for down
markov_df = markov_df.dropna()

up_counts = len(markov_df[markov_df["current_state"] == 1])
down_counts = len(markov_df[markov_df["current_state"] == 0])
up_to_up = (
    len(
        markov_df[
            (markov_df["current_state"] == 1)
            & (markov_df["current_state"].shift(-1) == 1)
        ]
    )
    / up_counts
)
down_to_up = (
    len(
        markov_df[
            (markov_df["current_state"] == 0)
            & (markov_df["current_state"].shift(-1) == 1)
        ]
    )
    / down_counts
)
up_to_down = (
    len(
        markov_df[
            (markov_df["current_state"] == 1)
            & (markov_df["current_state"].shift(-1) == 0)
        ]
    )
    / up_counts
)
down_to_down = (
    len(
        markov_df[
            (markov_df["current_state"] == 0)
            & (markov_df["current_state"].shift(-1) == 0)
        ]
    )
    / down_counts
)

transition_matrix = pd.DataFrame(
    {"up": [up_to_up, up_to_down], "down": [down_to_up, down_to_down]},
    index=["up", "down"],
)

print(transition_matrix.head())




# # Now let's make this more robust

# ### Define the thresholds
# For demonstration purposes, I'll use the 25th and 75th percentiles of the positive and negative pct_change values, respectively.

# %%
positive_changes = markov_df[markov_df["pct_change"] > 0]["pct_change"]
negative_changes = markov_df[markov_df["pct_change"] < 0]["pct_change"]

small_up_threshold = positive_changes.quantile(0.75)
large_up_threshold = positive_changes.quantile(0.80)

small_down_threshold = negative_changes.quantile(0.25)
large_down_threshold = negative_changes.quantile(0.20)

print(f"Small up threshold: {small_up_threshold}")
print(f"Large up threshold: {large_up_threshold}")
print(f"Small down threshold: {small_down_threshold}")
print(f"Large down threshold: {large_down_threshold}")


def define_state(x):
    if x > large_up_threshold:
        return "large_up"
    elif x > 0 and x <= large_up_threshold:
        return "small_up"
    elif x < 0 and x >= large_down_threshold:
        return "small_down"
    elif x < large_down_threshold:
        return "large_down"


markov_df["current_state"] = markov_df["pct_change"].apply(define_state)

# %% [markdown]
# ### Modify the State Definition
# Now we'll categorize the pct_change values based on the thresholds:

# Function to create sequences of states with a given lookback period
def create_sequences(data, lookback=3):
    sequences = []
    # Ensure we have enough data to create the lookback sequence
    if len(data) > lookback:
        for i in range(len(data) - lookback):
            sequence = data['current_state'].iloc[i:i+lookback].tolist()  # Lookback states
            sequences.append(sequence)
    return sequences

# Create sequences with a lookback of 3
lookback_period = 3
sequences = create_sequences(markov_df, lookback=lookback_period)
# %% convert the sequences to integers
import pandas as pd
from collections import defaultdict

# Define a function to map states to integers
def map_states_to_integers(sequences):
    state_to_int = defaultdict(lambda: len(state_to_int))
    int_sequences = [[state_to_int[state] for state in seq] for seq in sequences]
    return int_sequences, dict(state_to_int)

# Assume sequences is a list of lists containing state strings
# Convert sequences to integers
 # Creating sequences of length 3
int_sequences, state_mapping = map_states_to_integers(sequences)

# Display the first few integer sequences and the state mapping
int_sequences[:5], state_mapping

print(" Running Markov Chain Model")
# %% Build and train a model to predict the next state
from pomegranate.markov_chain import MarkovChain
import torch

# Initialize a 3rd-order Markov chain
model = MarkovChain(k=3)

# Create a dataset with sequences of observations
# The data must be three-dimensional, with the dimensions being (n_samples, length, dimensions)
X = torch.tensor([[[1], [0], [0], [1]],
                  [[0], [1], [0], [0]],
                  [[0], [0], [0], [0]],
                  [[0], [0], [0], [1]],
                  [[0], [1], [1], [0]]])

# Fit the model to the data
model.fit(X)

# After fitting the model
distributions = model.distributions()

# After fitting the model
# Access the list of distributions directly without calling it
first_order_probs = model.distributions[0].parameters[0]
second_order_probs = model.distributions[1].parameters[0]
third_order_probs = model.distributions[2].parameters[0]




# Print the transition probabilities
print("First Order Transition Probabilities:", first_order_probs)
print("Second Order Transition Probabilities:", second_order_probs)
print("Third Order Transition Probabilities:", third_order_probs)

# Compute the probability of the sequences
probabilities = model.probability(X)
log_probabilities = model.log_probability(X)

print("Probabilities:", probabilities)
print("Log Probabilities:", log_probabilities)

# Sample new sequences from the model
sampled_sequences = model.sample(5, length=4)
print("Sampled Sequences:", sampled_sequences)

# %%
