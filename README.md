# PortfolioRL 📈
### Portfolio Optimization using Deep Reinforcement Learning (PPO)

> "Markets Are Chaos. Your Portfolio Doesn't Have To Be."

---

## 🧠 What is This Project?

PortfolioRL is an AI-powered portfolio optimization system that uses Deep Reinforcement Learning to learn the optimal way to allocate investment capital across multiple stocks.

Instead of using traditional fixed formulas, our PPO agent learns by interacting with 5 years of real S&P 500 historical data — getting rewarded for good decisions and penalised for risky ones.

This project was developed as a DLP Mini Project at [Your College Name].

---

## 🎯 The Problem We Solved

If you have $10,000 to invest across 5 stocks — how much do you put in each?

Traditional methods use fixed math formulas that:
- Assume markets follow a normal distribution
- Cannot adapt to changing market conditions
- Are affected by human emotions and biases

Our solution: A PPO-based RL agent that learns from experience — just like how a chess AI learns by playing thousands of games.

---

## 📊 Results

| Metric | PPO Strategy | Benchmark |
|--------|-------------|-----------|
| Total Return | +76.95% | +77.01% |
| Final Value | $17,695 | ~$17,701 |
| Sharpe Ratio | 1.050 | ~0.82 |
| Max Drawdown | -12.66% | -14.3% |
| Calmar Ratio | 0.955 | ~0.75 |
| Training Time | under 60 seconds | — |

Sharpe Ratio above 1.0 is the gold standard in finance.

---

## 🏗️ System Architecture


### WEB BROWSER
Upload CSV → Configure → Train → Results
↓
HTTP POST /train

### FLASK BACKEND (app.py)

Load CSV → Clean → Pivot → Weekly Resample

↓

Gymnasium Environment

↓

PPO Agent Training

↓

Backtest + Metrics

↓

Return JSON Results


### RESULTS DASHBOARD

Metric Cards | Equity Curve | Donut Chart


---

## 🤖 PPO Agent Architecture


Input: 20 features (5 stocks × 4 weeks)

↓

Hidden Layer 1 (32 neurons, ReLU)

↓

Hidden Layer 2 (32 neurons, ReLU)

↓

Actor-Critic Output

↓

Softmax

↓

Output: 5 portfolio weights (sum = 100%)


---

## 🔄 Reinforcement Learning Loop


STATE (20 features)

↓

PPO AGENT

↓

ACTION (5 weights)

↓

REWARD = return − 0.15 × volatility

↓

Repeat for training timesteps

↓

Agent improves over time


---

## 🔢 Key Formulas


Reward = Portfolio Return − 0.15 × Volatility
Softmax = e^(x_i) / Σ e^(x_j)
Portfolio R = Σ(weight_i × stock_return_i)
Sharpe Ratio = (Mean Return / Std Dev) × √52
Max Drawdown = (Value − Peak) / Peak × 100
Calmar Ratio = Annualised Return / Max Drawdown
Daily Return = (P_t − P_prev) / P_prev

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python + Flask | Backend server |
| Stable-Baselines3 | PPO implementation |
| Gymnasium | Custom RL environment |
| PyTorch | Neural network backend |
| Pandas + NumPy | Data processing |
| HTML + CSS + JS | Frontend dashboard |
| Chart.js | Interactive charts |

---

## 📁 Project Structure
portfolioRL/
├── app1.py            ← Flask backend + PPO training
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
├── .gitignore         ← Ignored files
└── frontend/
└── index.html     ← Dashboard UI

---

## 🚀 How to Run

Step 1 — Clone the repo
git clone https://github.com/mythrii18/portfolioRL.git
cd portfolioRL

Step 2 — Create virtual environment
python -m venv venv
venv\Scripts\activate

Step 3 — Install dependencies
pip install -r requirements.txt

Step 4 — Run the app
python app1.py

Step 5 — Open in browser
http://localhost:5000

---

## 📂 Dataset

Uses the S&P 500 Historical Stock Data from Kaggle.

1. Go to kaggle.com/datasets/camnugent/sandp500
2. Download all_stocks_5yr.csv
3. Upload directly in the web dashboard

Note: CSV not included in repo due to large size. Download from Kaggle.

---

## 📈 Dashboard Features

- Drag and drop CSV upload
- Animated 4-step training pipeline
- 7 metric cards (Return, Benchmark, Sharpe, Drawdown, Calmar, CAGR, Final Value)
- Interactive equity curve with PPO vs Benchmark
- Circular asset allocation donut chart
- Stock weights table with animated bars
- Model architecture information card
- Export chart as PNG
- Fully responsive design

---

## ⚙️ Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Network | 32 32 MLP ReLU |
| Learning Rate | 0.0003 |
| Timesteps | 2500 |
| Observation Window | 4 weeks |
| n_steps | 32 |
| batch_size | 16 |
| n_epochs | 4 |
| clip_range | 0.2 |
| Data Frequency | Weekly 261 rows |

---

## ⚠️ Limitations

- Trained on historical data only
- No transaction costs modelled
- Only long positions
- Weekly resampling may miss intra-week movements

---

## 🔮 Future Work

- LSTM policy for better time-series learning
- Live data via Yahoo Finance API
- Cloud deployment on AWS or Heroku
- Compare PPO vs A2C SAC DDPG
- Add transaction cost modelling

---

## 👥 Team

| Name | Role |
|------|------|
| Your Name | Backend PPO Model Environment |
| Partner Name | Frontend Data Processing Evaluation |

College: Your College Name
Department: Your Department
Subject: Deep Learning Practices DLP

---

## 📚 References

1. Schulman et al 2017 — Proximal Policy Optimization Algorithms — arxiv.org/abs/1707.06347
2. Stable-Baselines3 Documentation — stable-baselines3.readthedocs.io
3. Gymnasium Documentation — gymnasium.farama.org
4. S&P 500 Dataset — kaggle.com/datasets/camnugent/sandp500
5. Markowitz 1952 — Portfolio Selection Journal of Finance

---

Built with Deep Reinforcement Learning
