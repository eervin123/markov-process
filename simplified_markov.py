# %%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Fetch Bitcoin historical data from Yahoo Finance
btc_data = yf.download("BTC-USD", start="2022-01-01", end="2023-01-01")


# %%
filename = "btc_daily_data.csv"
btc_data.to_csv(filename)

data = pd.read_csv(filename)


# %%

def add_transition_state_optimized(df, lookback):
    # Calculate binary column for 'up' (1) or 'down' (0) days
    df["binary"] = (df["pct_chg"] > 0).astype(int)

    # Create shifted columns for the lookback period
    # Start with an empty DataFrame to collect the shifted states
    shifted_states = pd.DataFrame(index=df.index)
    for shift in range(1, lookback + 1):
        shifted_states[f"lag_{shift}"] = df["binary"].shift(shift)

    # Use dropna to remove rows with incomplete lookback data
    shifted_states.dropna(inplace=True)

    # Convert all lagged binary columns to integer for proper formatting and then to string
    for col in shifted_states.columns:
        shifted_states[col] = shifted_states[col].astype(int).astype(str)

    # Concatenate the binary state strings to form the transition state
    df["transition_state"] = shifted_states.apply(lambda x: "".join(x), axis=1)

    # Ensure that only rows with a complete set of lookback data are returned
    return df[df["transition_state"].notna()]


# Usage example:
n = 3  # Set the lookback window to 3 periods for 'n' binary digits in 'transition_state'

# Load your data into 'data' DataFrame
# data = ...

data["pct_chg"] = data["Close"].pct_change()  # Calculate percent change
data["binary"] = data["pct_chg"].apply(
    lambda x: 1 if x > 0 else 0
)  # Calculate binary column for 'up' or 'down' days

# Add transition state column to the data efficiently
data_with_transition_state = add_transition_state_optimized(data, n)

# Print the DataFrame with the transition state
print(
    data_with_transition_state[
        ["Date", "Close", "pct_chg", "binary", "transition_state"]
    ].tail()
)


# %% [markdown]
# # Note
# `transition_state` is based on the last 4 states not the current state (this shouldn't be known until the close) so the current `pct_chg` is essentially the target variable that we are trying to predict based on the `transition_state` or the last several periods


# %%

# %%
def analyze_transition_matrix(df, transition_column, return_column):
    # Make a copy of df to avoid SettingWithCopyWarning when df is a slice of another DataFrame
    df = df.copy()

    # Calculate the return multiplier for each row
    df["return_multiplier"] = 1 + df[return_column] / 100

    # Group by transition state and calculate statistics
    stats = df.groupby(transition_column).agg(
        {
            return_column: ["mean", "min", "max", "count"],
            "return_multiplier": "prod",  # Calculate the compounded return
        }
    )

    # Flatten the MultiIndex columns
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]

    # Calculate up/down counts and probabilities
    up_counts = (
        df[df[return_column] > 0].groupby(transition_column)[return_column].count()
    )
    down_counts = (
        df[df[return_column] <= 0].groupby(transition_column)[return_column].count()
    )
    total_counts = df.groupby(transition_column)[return_column].count()

    stats["prob_up"] = up_counts / total_counts
    stats["prob_down"] = down_counts / total_counts
    stats["cum_return"] = (stats["return_multiplier_prod"] - 1) * 100

    # Fill NaN values with zero where there were no up/down periods
    stats["prob_up"] = stats["prob_up"].fillna(0)
    stats["prob_down"] = stats["prob_down"].fillna(0)

    # Rename the columns for better readability
    stats.rename(
        columns={
            return_column + "_mean": "avg_return",
            return_column + "_min": "min_return",
            return_column + "_max": "max_return",
            return_column + "_count": "count",
        },
        inplace=True,
    )

    # Convert the mean, min, and max from decimal to percentage
    stats["avg_return"] *= 100
    stats["min_return"] *= 100
    stats["max_return"] *= 100

    # Print the transition analysis matrix
    print("Transition Analysis Matrix:")
    print(
        stats[
            [
                "count",
                "prob_down",
                "prob_up",
                "avg_return",
                "min_return",
                "max_return",
                "cum_return",
            ]
        ]
    )


# Usage example:
analyze_transition_matrix(data_with_transition_state, "transition_state", "pct_chg")


# %% [markdown]
# # Example usage to run a simple backtest simulation
# We can see above that there were 49 times that the market went down -> up -> up but then the showed a 65% probability of being down in the next period. let's short the market whenever this occurs.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Download and save CSV (if not already downloaded)
# btc_data = yf.download('BTC-USD', start='2022-01-01', end='2023-01-01')
# btc_data.to_csv('btc_daily_data.csv')

# Assuming data_with_transition_state is already defined and has the required 'transition_state' column
initial_capital = 1000  # Set the initial capital
test_state = "111"  # Set the transition state to test
data_with_transition_state = data_with_transition_state.copy()  # Make a copy to avoid SettingWithCopyWarning
# Calculate short selling return multipliers using .loc for proper assignment
data_with_transition_state.loc[:, "short_return_multiplier"] = (
    1 - data_with_transition_state["pct_chg"] / 100
)

# Create mask for the test state using .loc to avoid SettingWithCopyWarning
short_investment_mask = data_with_transition_state["transition_state"] == test_state

# Apply strategy multiplier where the mask is True, otherwise keep as 1 (no change)
# Here you set the default multiplier to 1 first
data_with_transition_state.loc[:, "strategy_multiplier"] = 1
# Then apply short multiplier where the condition is True
data_with_transition_state.loc[
    short_investment_mask, "strategy_multiplier"
] = data_with_transition_state.loc[short_investment_mask, "short_return_multiplier"]

# Calculate cumulative portfolio value using .loc for proper assignment
data_with_transition_state.loc[:, "cumulative_portfolio"] = (
    initial_capital * data_with_transition_state.loc[:, "strategy_multiplier"].cumprod()
)

# Output the DataFrame to CSV if needed
# data_with_transition_state.to_csv('btc_data_with_strategy.csv', index=False)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(
    data_with_transition_state["Date"],
    data_with_transition_state["cumulative_portfolio"],
    color="orange",
    linewidth=2,
    label="Portfolio Value (Short Strategy)",
)
plt.title("Hypothetical Portfolio Value Over Time with Short Selling Strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
plt.show()


# %% [markdown]
#
