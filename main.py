import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from bitmex import BitmexWs  # pip install bitmex-api

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Feature Engineering ---
def compute_orderbook_features(ob):
    """Extract advanced order book features."""
    bids = np.array(ob['bids'])  # [[price, size], ...]
    asks = np.array(ob['asks'])
    best_bid, best_ask = bids[0, 0], asks[0, 0]
    spread = best_ask - best_bid
    bid_vol = bids[:5, 1].sum()
    ask_vol = asks[:5, 1].sum()
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
    microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol + 1e-8)
    features = [best_bid, best_ask, spread, bid_vol, ask_vol, imbalance, microprice]
    return np.array(features)

class RollingZScore:
    def __init__(self, window=50):
        self.window = window
        self.data = []

    def update(self, x):
        self.data.append(x)
        if len(self.data) > self.window:
            self.data.pop(0)

    def normalize(self, x):
        arr = np.array(self.data)
        if arr.shape[0] < 2:
            return x
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-8
        return (x - mean) / std

# --- Model ---
class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.fc(out)

# --- Cosine LR Decay ---
def cosine_lr_decay(optimizer, epoch, max_epochs, base_lr):
    lr = base_lr * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- Dynamic Risk Management ---
class RiskManager:
    def __init__(self, max_position=2, max_drawdown=0.15, stop_loss=0.004, take_profit=0.01):
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def check_position(self, current_position, signal):
        if abs(current_position + signal) > self.max_position:
            return 0
        return signal

    def check_drawdown(self, equity_curve):
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / (peak + 1e-8)
        if np.any(dd > self.max_drawdown):
            return False
        return True

    def stop_loss_take_profit(self, entry_price, current_price, position):
        if position == 0:
            return False
        change = (current_price - entry_price) / entry_price * np.sign(position)
        if change <= -self.stop_loss or change >= self.take_profit:
            return True
        return False

# --- Backtesting with Latency Simulation ---
class Backtester:
    def __init__(self, price_series, signals, fee_rate=0.0005, risk_manager=None, latency_ms=50):
        self.prices = price_series
        self.signals = signals
        self.fee_rate = fee_rate
        self.risk_manager = risk_manager
        self.latency_ms = latency_ms

    def run(self):
        position = 0
        entry_price = 0
        pnl = [0]
        positions = []
        trades = []
        equity = [1]
        for i in range(1, len(self.prices)):
            # Simulate latency by acting on delayed signal
            latency_ticks = max(1, int(self.latency_ms / 1000 * 2))  # 2 ticks/sec
            idx = max(0, i - latency_ticks)
            signal = self.signals[idx]
            if self.risk_manager:
                signal = self.risk_manager.check_position(position, signal)
            # Stop-loss/take-profit
            if self.risk_manager and self.risk_manager.stop_loss_take_profit(entry_price, self.prices[i], position):
                signal = -np.sign(position)
            # Trade execution
            if signal != 0:
                trade_pnl = position * (self.prices[i] - self.prices[i-1])
                fee = abs(signal) * self.fee_rate * self.prices[i]
                pnl.append(pnl[-1] + trade_pnl - fee)
                position += signal
                entry_price = self.prices[i] if position != 0 else entry_price
                trades.append(signal)
            else:
                trade_pnl = position * (self.prices[i] - self.prices[i-1])
                pnl.append(pnl[-1] + trade_pnl)
                trades.append(0)
            positions.append(position)
            equity.append(pnl[-1])
            if self.risk_manager and not self.risk_manager.check_drawdown(np.array(equity)):
                logging.warning("Max drawdown exceeded. Halting strategy.")
                break
        return np.array(equity), np.array(positions), np.array(trades)

# --- Performance KPIs ---
def performance_metrics(equity_curve, trades):
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 12)
    max_dd = np.max((np.maximum.accumulate(equity_curve) - equity_curve) / (np.maximum.accumulate(equity_curve) + 1e-8))
    win_rate = np.mean(np.array(trades) > 0)
    num_trades = np.sum(np.abs(trades) > 0)
    total_return = equity_curve[-1] / equity_curve[0] - 1
    return {
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Num Trades': num_trades,
        'Total Return': total_return
    }

def plot_pnl(equity_curve, title='P&L Curve'):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid()
    plt.show()

# --- Real-time Data Collection and Simulation ---
async def collect_bitmex_data(symbol="BTC/USDC", duration_sec=300):
    """Collects real-time order book data from BitMEX WebSocket."""
    ob_features = []
    prices = []
    normalizer = RollingZScore(window=50)
    async with BitmexWs({}) as ws:
        t0 = datetime.now()
        while (datetime.now() - t0).total_seconds() < duration_sec:
            ob = await ws.watch_order_book(symbol)
            features = compute_orderbook_features(ob)
            normalizer.update(features)
            norm_features = normalizer.normalize(features)
            ob_features.append(norm_features)
            prices.append(features[0])  # best_bid as proxy for trade price
    return np.array(ob_features), np.array(prices)

# --- Main Institutional Pipeline ---
async def main():
    # 1. Collect live BitMEX data (simulation)
    logging.info("Collecting live BitMEX order book data...")
    features, prices = await collect_bitmex_data(duration_sec=600)  # 10 min

    # 2. Prepare ML dataset
    X = torch.tensor(features[:-1].reshape(-1, 1, features.shape[1]), dtype=torch.float32)
    y = torch.tensor((prices[1:] > prices[:-1]).astype(float).reshape(-1, 1), dtype=torch.float32)

    # 3. Train model (production-grade)
    model = TradingLSTM(input_size=features.shape[1], hidden_size=32, output_size=1)
    optimizer = optim.RMSprop(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=10)
    batch_size = 64
    max_epochs = 30

    for epoch in range(max_epochs):
        idx = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
        val_loss = loss.item()
        cosine_lr_decay(optimizer, epoch, max_epochs, 0.005)
        early_stop(val_loss)
        logging.info(f"Epoch {epoch+1}, Loss: {val_loss:.6f}")
        if early_stop.early_stop:
            logging.info("Early stopping triggered.")
            break

    # 4. Generate signals
    with torch.no_grad():
        preds = model(X).numpy().flatten()
    signals = np.where(preds > 0, 1, -1)

    # 5. Backtest with advanced risk and latency
    risk_manager = RiskManager(max_position=2, max_drawdown=0.15, stop_loss=0.004, take_profit=0.01)
    backtester = Backtester(price_series=prices[:-1], signals=signals, fee_rate=0.0005, risk_manager=risk_manager, latency_ms=50)
    equity_curve, positions, trades = backtester.run()

    # 6. KPIs and P&L plot
    kpis = performance_metrics(equity_curve, trades)
    for k, v in kpis.items():
        print(f"{k}: {v:.4f}")
    plot_pnl(equity_curve, title='BTCUSDT Perps HFT Strategy P&L Curve (Live BitMEX Data)')

if __name__ == "__main__":
    asyncio.run(main())

