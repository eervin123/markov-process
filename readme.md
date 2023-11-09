# Working with Markov Chains

Start with the file [simplified_markov.ipynb](simplified_markov.ipynb)

This file has two primary functions:
- Create the transition states using binary encoding 000 for example would mean the market was down-down-down in the last three bars. 
- Analyze transition matrix to evaluate what the probability of the current candle being up or down is based on the previous chained states.


The final cell just plots a hypothetical backtest if you were to long or short everytime the state showed up. 