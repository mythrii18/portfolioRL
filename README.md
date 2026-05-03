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
pip install -r requirements.txtPortfolioRL 📈
Portfolio Optimization using Deep Reinforcement Learning (PPO)

"Markets Are Chaos. Your Portfolio Doesn't Have To Be."


🧠 What is This Project?
PortfolioRL is an AI-powered portfolio optimization system that uses Deep Reinforcement Learning to learn the optimal way to allocate investment capital across multiple stocks.
Instead of using traditional fixed formulas, our PPO agent learns by interacting with 5 years of real S&P 500 historical data — getting rewarded for good decisions and penalised for risky ones. Over thousands of training episodes, it figures out the best investment strategy on its own.


🎯 The Problem We Solved
If you have $10,000 to invest across 5 stocks — how much do you put in each?
Traditional methods use fixed math formulas that:

Assume markets follow a normal distribution ❌
Cannot adapt to changing market conditions ❌
Are affected by human emotions and biases ❌

Our solution: A PPO-based RL agent that learns from experience — just like how a chess AI learns by playing thousands of games.

📊 Results
MetricPPO StrategyEqual-Weight BenchmarkTotal Return+76.95%+77.01%Final Value$17,695~$17,701Sharpe Ratio1.050 ✅~0.82Max Drawdown-12.66%-14.3%Calmar Ratio0.955~0.75Training Time< 60 seconds—

📌 Sharpe Ratio above 1.0 is the gold standard in finance — returns are worth the risk taken.


🏗️ System Architecture
┌─────────────────────────────────────────────────────────┐
│                     WEB BROWSER                         │
│         Dark Dashboard (HTML + CSS + Chart.js)          │
│   Upload CSV → Configure → Click Train → View Results   │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP POST /train
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FLASK BACKEND (app1.py)                 │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Step 1    │    │   Step 2     │    │  Step 3   │  │
│  │  Load CSV   │───▶│  Preprocess  │───▶│  Resample │  │
│  │  (Pandas)   │    │  Clean+Pivot │    │  Weekly   │  │
│  └─────────────┘    └──────────────┘    └─────┬─────┘  │
│                                               │         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────▼─────┐  │
│  │   Step 6    │    │   Step 5     │    │  Step 4   │  │
│  │  Compute    │◀───│   Backtest   │◀───│  Train    │  │
│  │  Metrics    │    │  (simulate)  │    │  PPO Agent│  │
│  └──────┬──────┘    └──────────────┘    └───────────┘  │
│         │                                               │
│         │  Return JSON                                  │
└─────────┼───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                   RESULTS DASHBOARD                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Metric  │  │  Equity  │  │  Donut   │  │ Model  │  │
│  │  Cards   │  │  Curve   │  │  Chart   │  │  Info  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘

🤖 PPO Agent Architecture
┌─────────────────────────────────────────────────────────┐
│                  PPO AGENT (MlpPolicy)                   │
│                                                         │
│   STATE (Input)                                         │
│   ┌────────────────────────────────┐                   │
│   │  5 stocks × 4 week window      │                   │
│   │  = 20 input features           │                   │
│   └───────────────┬────────────────┘                   │
│                   ▼                                     │
│   ┌────────────────────────────────┐                   │
│   │  Hidden Layer 1 — 32 neurons   │                   │
│   │  Activation: ReLU              │                   │
│   └───────────────┬────────────────┘                   │
│                   ▼                                     │
│   ┌────────────────────────────────┐                   │
│   │  Hidden Layer 2 — 32 neurons   │                   │
│   │  Activation: ReLU              │                   │
│   └──────────┬─────────────┬───────┘                   │
│              ▼             ▼                            │
│   ┌──────────────┐  ┌──────────────┐                   │
│   │    ACTOR     │  │    CRITIC    │                   │
│   │  5 raw values│  │  1 scalar    │                   │
│   │  → Softmax   │  │  (value est) │                   │
│   └──────┬───────┘  └──────────────┘                   │
│          ▼                                              │
│   ACTION (Output)                                       │
│   ┌────────────────────────────────┐                   │
│   │  5 portfolio weights           │                   │
│   │  sum = 1.0 (100%)              │                   │
│   │  e.g. AEP:20% WEC:21% WFC:19% │                   │
│   └────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘

🔄 Reinforcement Learning Loop
┌──────────────────────────────────────────────────────┐
│                                                      │
│   ┌─────────┐    STATE (20 features)   ┌──────────┐ │
│   │         │ ◀────────────────────── │          │ │
│   │   PPO   │                          │  STOCK   │ │
│   │  AGENT  │  ACTION (5 weights)      │  MARKET  │ │
│   │         │ ──────────────────────▶ │   ENV    │ │
│   │         │                          │          │ │
│   │         │  REWARD                  │          │ │
│   │         │ ◀────────────────────── │          │ │
│   └─────────┘  ret − 0.15 × vol       └──────────┘ │
│                                                      │
│   Repeat for 2500 timesteps → Agent learns! ✅       │
└──────────────────────────────────────────────────────┘

🔢 Key Formulas
Reward       =  Portfolio Return − 0.15 × Volatility
Softmax      =  e^(x_i) / Σ e^(x_j)
Portfolio R  =  Σ (weight_i × stock_return_i)
Sharpe Ratio =  (Mean Return / Std Dev) × √52
Max Drawdown =  (Value − Peak) / Peak × 100
Calmar Ratio =  Annualised Return / |Max Drawdown|
Daily Return =  (P_t − P_{t-1}) / P_{t-1}

🛠️ Tech Stack
LayerTechnologyPurposeAI/MLStable-Baselines3 (PPO)RL agent trainingEnvironmentGymnasiumCustom stock market simulatorDataPandas, NumPyPreprocessing & returnsBackendPython, FlaskREST API serverNeural NetPyTorchNeural network backendFrontendHTML, CSS, JavaScriptInteractive dashboardChartsChart.jsEquity curve, donut chart

📁 Project Structure
portfolioRL/
├── app1.py                 ← Flask backend + PPO training logic
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Ignored files
└── frontend/
    └── index.html          ← Full dashboard UI

🚀 How to Run
Step 1 — Clone the repository
bashgit clone https://github.com/mythrii18/portfolioRL.git
cd portfolioRL
Step 2 — Create virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
Step 3 — Install dependencies
bashpip install -r requirements.txt
Step 4 — Run the app
bashpython app1.py
Step 5 — Open in browser
http://localhost:5000

📂 Dataset
Uses the S&P 500 Historical Stock Data from Kaggle.

Go to: kaggle.com/datasets/camnugent/sandp500
Download all_stocks_5yr.csv
Upload directly in the web dashboard


⚠️ CSV not included in this repo due to large file size (~30MB). Please download from Kaggle.


📈 Dashboard Features

✅ Drag and drop CSV upload
✅ Animated 4-step training pipeline
✅ 7 metric cards (Return, Benchmark, Sharpe, Drawdown, Calmar, CAGR, Final Value)
✅ Interactive equity curve — PPO vs Benchmark
✅ Circular asset allocation donut chart
✅ Stock weights table with animated bars
✅ Model architecture information card
✅ Export chart as PNG
✅ Fully responsive design


⚙️ Model Hyperparameters
ParameterValueAlgorithmPPONetwork[32, 32] MLP with ReLULearning Rate3 × 10⁻⁴Timesteps2500Observation Window4 weeksn_steps32batch_size16n_epochs4clip_range0.2Data FrequencyWeekly (261 rows for 5yr)

⚠️ Limitations

Trained on historical data — does not guarantee future returns
No transaction costs or taxes modelled
Only long positions — no short selling
Weekly resampling may miss intra-week movements


🔮 Future Work

 LSTM-based policy for better time-series learning
 Live stock data via Yahoo Finance API
 Cloud deployment (AWS / Heroku)
 Compare PPO vs A2C, SAC, DDPG
 Add transaction cost modelling


👥 Team
NameRole[Your Name]Backend, PPO Model, Environment[Partner Name]Frontend, Data Processing, Evaluation
College: [Your College Name]
Department: [Your Department]
Subject: Deep Learning Practices (DLP)

📚 References

Schulman et al. (2017) — Proximal Policy Optimization Algorithms
Stable-Baselines3 Documentation
Gymnasium Documentation
S&P 500 Dataset — Kaggle
Markowitz (1952) — Portfolio Selection, Journal of Finance


📄 License
This project is for educational purposes only. Not financial advice.

<p align="center">Built with ❤️ using Deep Reinforcement Learning</p>
python app.py
```
Open http://localhost:5000

## Dataset
S&P 500 Historical Data from Kaggles