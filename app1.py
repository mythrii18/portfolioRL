from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import os
import traceback

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ── Reward Tracker Callback ──────────────────────────────
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(
                float(self.model.ep_info_buffer[-1].get('r', 0))
            )
        return True


class PortfolioEnv(gym.Env):
    def __init__(self, returns, n_assets, window=4):
        super().__init__()
        self.returns      = returns.values.astype(np.float32)
        self.n_assets     = n_assets
        self.window       = window
        self.current_step = window
        self.max_steps    = len(returns) - 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_assets * window,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )

    def _obs(self):
        s = self.current_step
        return self.returns[s - self.window : s].flatten()

    def reset(self, seed=None, options=None):
        self.current_step = self.window
        return self._obs(), {}

    def step(self, action):
        a       = action - action.max()
        weights = np.exp(a) / np.exp(a).sum()
        ret              = self.returns[self.current_step]
        portfolio_return = float(np.dot(weights, ret))
        recent = self.returns[max(0, self.current_step - 10) : self.current_step]
        vol    = float(np.std(recent)) if len(recent) > 1 else 0.0
        reward = portfolio_return - 0.15 * vol
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._obs(), reward, done, False, {}


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/train', methods=['POST'])
def train():
    try:
        file          = request.files.get('file')
        n_stocks      = int(request.form.get('n_stocks', 5))
        timesteps     = int(request.form.get('timesteps', 2500))
        initial_value = float(request.form.get('initial_value', 10000))

        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # ── 1. Load CSV ──────────────────────────────────────────
        df = pd.read_csv(file, low_memory=False)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        date_col  = next((c for c in df.columns if 'date' in c), None)
        name_col  = next((c for c in df.columns
                          if c in ['name','symbol','ticker','stock','company']), None)
        close_col = next((c for c in df.columns
                          if c in ['close','close_price','price','adj_close','adj_close_price']), None)

        missing = []
        if not date_col:  missing.append('date')
        if not name_col:  missing.append('name/symbol/ticker')
        if not close_col: missing.append('close/price')
        if missing:
            return jsonify({'error': f'Missing columns: {missing}. Found: {list(df.columns)}'}), 400

        df = df[[date_col, name_col, close_col]].copy()
        df[date_col]  = pd.to_datetime(df[date_col], errors='coerce')
        df            = df.dropna(subset=[date_col])
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df            = df.dropna(subset=[close_col])
        df            = df.sort_values(date_col)

        if df.empty:
            return jsonify({'error': 'No valid rows after cleaning.'}), 400

        date_span = (df[date_col].max() - df[date_col].min()).days
        if date_span < 90:
            return jsonify({'error': f'Data spans only {date_span} days — need at least 90.'}), 400

        # ── 2. Pivot ─────────────────────────────────────────────
        pivot = df.pivot_table(
            index=date_col, columns=name_col,
            values=close_col, aggfunc='last'
        )
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.70))

        if pivot.shape[1] < 2:
            return jsonify({'error': 'Fewer than 2 stocks have sufficient data.'}), 400

        n_stocks   = min(n_stocks, pivot.shape[1])
        top_stocks = pivot.isnull().sum().sort_values().head(n_stocks).index.tolist()
        pivot      = pivot[top_stocks].ffill().bfill().dropna()

        # ── 3. Weekly resampling ──────────────────────────────────
        resampled  = pivot.resample('W').last().dropna(how='all').ffill().dropna()
        freq_label = 'Weekly'
        FREQ       = 52

        returns = resampled.pct_change().dropna().clip(-0.5, 0.5)

        if len(returns) < 30:
            returns    = pivot.pct_change().dropna().clip(-0.5, 0.5)
            freq_label = 'Daily'
            FREQ       = 252

        if len(returns) < 15:
            return jsonify({'error': 'Not enough data rows after resampling.'}), 400

        n_assets = len(top_stocks)
        WINDOW   = 4

        # ── 4. PPO settings ───────────────────────────────────────
        n_steps    = min(32, max(16, len(returns) // 6))
        batch_size = min(16, n_steps)
        eff_ts     = min(int(timesteps), 2500)

        torch.set_num_threads(os.cpu_count() or 4)

        env = DummyVecEnv([lambda: PortfolioEnv(returns, n_assets, WINDOW)])

        model = PPO(
            'MlpPolicy', env,
            verbose=0,
            device='cpu',
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=[32, 32],
                activation_fn=nn.ReLU
            ),
        )

        # ── Train with reward callback ────────────────────────────
        callback = RewardCallback()
        model.learn(total_timesteps=eff_ts, callback=callback, progress_bar=False)

        # ── 5. Backtest ───────────────────────────────────────────
        test_env = PortfolioEnv(returns, n_assets, WINDOW)
        obs, _   = test_env.reset()
        pv       = [initial_value]
        wh       = []
        ret_vals = returns.values
        dates    = returns.index.strftime('%Y-%m-%d').tolist()

        for i in range(WINDOW, len(returns) - 1):
            action, _ = model.predict(obs, deterministic=True)
            a         = action - action.max()
            w         = np.exp(a) / np.exp(a).sum()
            wh.append(w.tolist())
            obs, _, done, _, _ = test_env.step(action)
            pv.append(pv[-1] * (1 + float(np.dot(w, ret_vals[i]))))
            if done:
                break

        used_dates = dates[WINDOW : WINDOW + len(pv)]
        pv         = pv[:len(used_dates)]
        final_val  = pv[-1]
        total_ret  = ((final_val - initial_value) / initial_value) * 100

        # ── 6. Risk metrics ───────────────────────────────────────
        pv_arr     = np.array(pv)
        ra         = np.diff(pv_arr) / pv_arr[:-1]
        sharpe     = float((np.mean(ra) / (np.std(ra) + 1e-9)) * np.sqrt(FREQ))
        roll_max   = np.maximum.accumulate(pv_arr)
        max_dd     = float(np.min((pv_arr - roll_max) / roll_max) * 100)
        volatility = float(np.std(ra) * np.sqrt(FREQ) * 100)
        yrs        = max(date_span / 365.25, 0.1)
        ann_return = float(((final_val / initial_value) ** (1.0 / yrs) - 1) * 100)
        calmar     = round(ann_return / (abs(max_dd) + 1e-9), 3)

        # ── 7. Benchmark ──────────────────────────────────────────
        eq_w = np.full(n_assets, 1.0 / n_assets)
        bv   = [initial_value]
        for rv in ret_vals[WINDOW : WINDOW + len(pv) - 1]:
            bv.append(bv[-1] * (1 + float(np.dot(eq_w, rv))))
        bv           = bv[:len(used_dates)]
        bench_return = round(((bv[-1] - initial_value) / initial_value) * 100, 2)

        final_weights = wh[-1] if wh else eq_w.tolist()

        # ── 8. Reward curve — thin to max 60 points for UI ────────
        rw = callback.rewards
        if len(rw) > 60:
            step = len(rw) // 60
            rw   = rw[::step]
        rw = [round(float(x), 5) for x in rw]

        # ── 9. Returns histogram data ─────────────────────────────
        returns_hist = [round(float(x), 4) for x in ra.tolist()]

        return jsonify({
            'dates':            used_dates,
            'portfolio_values': [round(v, 2) for v in pv],
            'benchmark_values': [round(v, 2) for v in bv],
            'benchmark_return': bench_return,
            'total_return':     round(total_ret, 2),
            'final_value':      round(final_val, 2),
            'initial_value':    initial_value,
            'sharpe_ratio':     round(sharpe, 3),
            'max_drawdown':     round(max_dd, 2),
            'volatility':       round(volatility, 2),
            'calmar_ratio':     calmar,
            'ann_return':       round(ann_return, 2),
            'stocks':           top_stocks,
            'final_weights':    [round(float(w), 4) for w in final_weights],
            'n_stocks':         n_assets,
            'date_range':       f"{used_dates[0]} to {used_dates[-1]}",
            'data_points':      len(returns),
            'reward_curve':     rw,           # NEW — training reward over time
            'returns_hist':     returns_hist, # NEW — weekly returns distribution
            'model_info': {
                'algorithm':    'PPO',
                'policy':       'MlpPolicy',
                'architecture': '[32, 32] ReLU',
                'obs_window':   f'{WINDOW}-week lookback',
                'obs_space':    f'{n_assets} × {WINDOW} = {n_assets * WINDOW} features',
                'action_space': f'Continuous ({n_assets} weights)',
                'reward':       'ret − 0.15 × vol',
                'timesteps':    eff_ts,
                'data_freq':    f'{freq_label} ({len(returns)} rows)',
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('frontend', exist_ok=True)
    app.run(debug=True, port=5000)
