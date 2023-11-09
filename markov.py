# %%
import warnings
warnings.filterwarnings("ignore")

import vectorbtpro as vbt
import numpy as np
import pandas as pd

vbt.settings.set_theme("dark")
vbt.settings.plotting["layout"]["width"] = 800
vbt.settings.plotting['layout']['height'] = 200

import pandas_ta as ta


# %% [markdown]
# # Import the data

# %%
btc_90M_db_vbt = vbt.BinanceData.load('data/btc_90M_db_vbt.pkl')

data = btc_90M_db_vbt['2021-01-01':'2023-01-01']
outofsample_data = btc_90M_db_vbt['2023-01-01':'2023-06-03']
print(data.shape)
print(outofsample_data.shape)
# Wherever you saved the pickle file
data_path = '/Users/ericervin/Documents/Coding/data-repository/data/fixed_BTCUSDT.csv'
min_data = vbt.BinanceData.from_csv(data_path)
print(min_data.shape)

# %%
markov_df = data.get()
markov_df['pct_change'] = markov_df['Close'].pct_change()
markov_df['log_ret'] = np.log(markov_df['Close']).diff()
markov_df['volatility'] = markov_df['log_ret'].rolling(100).std() * np.sqrt(100)
markov_df['volume_chg'] = markov_df['Volume'].pct_change()
markov_df['state'] = markov_df['pct_change'].apply(lambda x: 1 if x > 0 else 0)
markov_df = markov_df.dropna()

up_counts = len(markov_df[markov_df['state'] == 1])
down_counts = len(markov_df[markov_df['state'] == 0])
up_to_up    = len(markov_df[(markov_df['state'] == 1) & (markov_df['state'].shift(-1) == 1)])/len(markov_df.query('state == 1'))
down_to_up  = len(markov_df[(markov_df['state'] == 1) & (markov_df['state'].shift(-1) == 0)])/len(markov_df.query('state == 1'))
up_to_down  = len(markov_df[(markov_df['state'] == 0) & (markov_df['state'].shift(-1) == 1)])/len(markov_df.query('state == 0'))
down_to_down= len(markov_df[(markov_df['state'] == 0) & (markov_df['state'].shift(-1) == 0)])/len(markov_df.query('state == 0'))

transition_matrix = pd.DataFrame({
    "up": [up_to_up, up_to_down],
    "down": [down_to_up, down_to_down]
}, index=["up", "down"])

print(transition_matrix)

# %%
pf = vbt.Portfolio.from_signals(
    markov_df['Close'],
    entries=np.where(markov_df['state']==1,True,False),
    td_stop = 3,
    time_delta_format = 'rows',
    freq = '10T',
    fees = 0.0004,
)

pf.stats()

# %% [markdown]
# # Now let's make this more robust

# %% [markdown]
# ### Define the thresholds
# For demonstration purposes, I'll use the 25th and 75th percentiles of the positive and negative pct_change values, respectively.

# %%
positive_changes = markov_df[markov_df['pct_change'] > 0]['pct_change']
negative_changes = markov_df[markov_df['pct_change'] < 0]['pct_change']

small_up_threshold = positive_changes.quantile(0.25)
large_up_threshold = positive_changes.quantile(0.75)

small_down_threshold = negative_changes.quantile(0.25)
large_down_threshold = negative_changes.quantile(0.75)


# %% [markdown]
# ### Modify the State Definition
# Now we'll categorize the pct_change values based on the thresholds:

# %%
def define_state(x):
    if x > large_up_threshold:
        return "large_up"
    elif x > small_up_threshold:
        return "small_up"
    elif x > large_down_threshold:
        return "small_down"
    else:
        return "large_down"

markov_df['state'] = markov_df['pct_change'].apply(define_state)


# %% [markdown]
# ### Compute Transition Probabilities
# Finally, let's update the transition matrix calculation:

# %%
states = ["large_up", "small_up", "small_down", "large_down"]
transition_matrix = pd.DataFrame(index=states, columns=states)

for from_state in states:
    for to_state in states:
        from_count = len(markov_df[markov_df['state'] == from_state])
        
        transition_prob = len(markov_df[(markov_df['state'] == from_state) & (markov_df['state'].shift(-1) == to_state)]) / from_count
        transition_matrix.at[from_state, to_state] = transition_prob

print(transition_matrix)


# %% [markdown]
# ### Now let's move to Higher order Markov Chain Analysis

# %%
print(markov_df.tail(10))

# %%

order = 3

# Create state columns for the current state and previous states
for i in range(order):
    markov_df[f'state{i}'] = markov_df['state'].shift(-i)

# Generate sequences of states
sequences = [tuple(markov_df[[f'state{i}' for i in range(order)]].iloc[i].dropna().tolist()) for i in range(len(markov_df) - order + 1)]

# Count transitions
transition_counts = {}
for seq in sequences:
    from_states = seq[:-1]
    to_state = seq[-1]
    from_states_str = ' -> '.join(from_states)
    if from_states_str not in transition_counts:
        transition_counts[from_states_str] = {}
    if to_state not in transition_counts[from_states_str]:
        transition_counts[from_states_str][to_state] = 0
    transition_counts[from_states_str][to_state] += 1

# Calculate transition probabilities
transition_probs = {}
for from_states_str, to_states in transition_counts.items():
    total_counts = sum(to_states.values())
    transition_probs[from_states_str] = {to_state: count / total_counts for to_state, count in to_states.items()}

# Convert to DataFrame
transition_matrix = pd.DataFrame(transition_probs).T.fillna(0)

# Order the columns
ordered_columns = ['large_up', 'small_up', 'small_down', 'large_down']
transition_matrix = transition_matrix[ordered_columns]

# Sort the transition matrix
transition_matrix.sort_index(inplace=True)

# Print the transition matrix
print(transition_matrix)


# %% [markdown]
# # Plot the transition matrix as a heatmap

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_transition_matrix_heatmap(transition_matrix, title="Transition Matrix Heatmap"):
    fig, ax = plt.subplots(figsize=(15, 200))
    sns.heatmap(
        transition_matrix.apply(pd.to_numeric, errors='coerce'),  # Convert values to numeric, coercing errors to NaN
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        ax=ax
    )
    ax.set_title(title)

    # Create a second x-axis at the top of the plot with the same ticks and labels as the main x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(ax.get_xticklabels())
    ax2.tick_params(axis='x', direction='in', top=True, bottom=False)

    plt.show()

plot_transition_matrix_heatmap(transition_matrix, "Transition Matrix Heatmap")


# %%
# Count occurrences for each sequence
sequence_occurrences = {}
for seq in sequences:
    if seq not in sequence_occurrences:
        sequence_occurrences[seq] = 0
    sequence_occurrences[seq] += 1

# Print the number of occurrences for each sequence
for seq, count in sequence_occurrences.items():
    print(f'{seq}: {count}')

# Convert the sequence occurrences to a DataFrame
occurrences_df = pd.DataFrame(list(sequence_occurrences.items()), columns=['sequence', 'occurrences'])

# Display the DataFrame
print(occurrences_df)


# %%
# Drop 'occurrences_x', 'occurrences_y', and 'occurrences' columns from transition_matrix if they exist
transition_matrix = transition_matrix.drop(columns=['occurrences_x', 'occurrences_y', 'occurrences'], errors='ignore')

# Rename 'occurrences' column in occurrences_df to 'count'
occurrences_df = occurrences_df.rename(columns={'occurrences': 'count'})

# Merge transition_matrix and occurrences_df on their indexes
merged_df = transition_matrix.merge(occurrences_df, left_index=True, right_index=True, how='left')

# Fill NaN values in 'count' column with 0
merged_df['count'].fillna(0, inplace=True)

# Show the resulting dataframe
print(merged_df.columns)


# %%
# Query merged_df where sum of small_down and large_down greater than 0.6 and count greater than 100
print(merged_df.query('(small_down + large_down) > 0.6 and count > 100'))


# %% [markdown]
# # Create a buy or sell signal based on probabilities
# identify if the probability of a large down or a large up move is high and enter a -1 or 1 in the signal column of the original dataframe

# %%
import numpy as np

# Define conditions
conditions = [
    (merged_df['small_down'] + merged_df['large_down'] > 0.6) & (merged_df['count'] > 100),
    (merged_df['small_up'] + merged_df['large_up'] > 0.6) & (merged_df['count'] > 100)
]

# Define choices
choices = ['sell', 'buy']

# Create new column 'signal' with values based on conditions
merged_df['signal'] = np.select(conditions, choices, default='hold')

# Display the DataFrame
print(merged_df)

# Determine the order (number of states) dynamically
print(f"The Order previously used was {order}")

# Handle NaN values in the state columns
for i in range(order):
    markov_df[f'state{i}'] = markov_df[f'state{i}'].fillna('')

# Create the "markov_chain" column dynamically based on the order
markov_df['markov_chain'] = markov_df['state'] # Initialize the first state
for i in range(order):
    markov_df['markov_chain'] += ' -> ' + markov_df[f'state{i}']



# %%
markov_df

# %%
pf = vbt.Portfolio.from_signals(
    markov_df['Close'],
    entries=np.where(markov_df['signal']=='buy',True,False),
    short_entries=np.where(markov_df['signal']=='sell',True,False),
    td_stop = 10,
    time_delta_format = 'rows',
    freq = '10T',
    fees = 0.0004,
)

pf.stats()

# %%
pf.plot().show()



