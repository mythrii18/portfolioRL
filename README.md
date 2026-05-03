# PortfolioRL — Portfolio Optimization using Deep Reinforcement Learning

> Markets Are Chaos. Your Portfolio Doesn't Have To Be.

## What This Project Does
An AI agent that learns to optimally allocate investment capital across 
S&P 500 stocks using PPO (Proximal Policy Optimization).

## Results
- Total Return: +76.95%
- Final Value: $17,695 (from $10,000)
- Sharpe Ratio: 1.050
- Max Drawdown: -12.66%

## Tech Stack
- Python, Flask
- Stable-Baselines3 (PPO)
- Gymnasium (custom environment)
- PyTorch
- Chart.js

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Dataset
S&P 500 Historical Data from Kaggle