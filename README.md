# Copyright @ ST Technologies

# HFT_BitMEX_Perps_Futures

HFT Trading Strategy for BitMEX BTC-USDT Perps Futures 

This is Insitutional Level Production Grade Python framework for HFT on BitMEX BTC-USDT Perps, integrating:

Real BitMEX data via the official SDK and WebSocket API for live L2 order book features and trade ticks.

Advanced order book features (bid/ask depth, imbalance, spread, microprice, last trade direction).

Latency simulation (order queueing, execution delay).

Dynamic risk management (position limits, drawdown, stop-loss/take-profit).

Full ML pipeline: model training with gradient clipping, RMSprop, cosine LR decay, batch/rolling normalization, early stopping.

Comprehensive KPIs: P&L, Sharpe, max drawdown, win rate, trade stats, and institutional logging.

Note: This code uses the bitmex-api SDK (install with pip install bitmex-api). For real trading, use API keys and consider deploying in a secure, low-latency environment.

Key institutional features:

Live BitMEX WebSocket L2 data (via BitmexWs): advanced order book features, real-time simulation.

Latency model: all signals executed with realistic delay.

Advanced order book features: bid/ask depth, spread, imbalance, microprice.

Dynamic risk: position/drawdown/SL/TP enforcement.

ML pipeline: robust LSTM, normalization, optimizer, early stopping.

Full performance analytics: KPIs, P&L plotting, logging.

To deploy:

Use real API keys for authenticated trading.

Integrate with BitMEX’s REST/WebSocket for order execution in production.

For longer backtests, collect and store more historical data.

References:

BitMEX official SDK and API

