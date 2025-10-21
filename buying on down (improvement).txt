#buying on down (improvement)
#trying to improve earnings on 21/9/25
# ===========================
# SECTION 1: Imports & Configs
# ===========================
# ===========================
# SECTION 1: Imports & Configs
# ===========================
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
# ---- Replace duplicate/fragmented datetime imports with this single line near the top ----
from datetime import datetime, date, timedelta
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
import alpaca_trade_api as tradeapi
from tqdm import tqdm
import json
from jinja2 import Template
from dateutil import parser
import decimal
from decimal import Decimal, ROUND_DOWN
from ta.momentum import StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import cvxpy as cp
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
WEIGHTS_FILE = "learned_weights.json"

# Load weights from file if exists
default_weights = {
    'C_SMA3_gt_SMA20': 1.0,
    'C_SMA20_gt_SMA50': 0.8,
    'C_Ret5d_in_range': 0.6,
    'C_ATR_not_too_high': 0.5,
    'C_Volume_gt_250k': 0.4,
    'C_Price_in_range': 0.4,
    'C_RSI_neutral': 0.6,
    'MACD_Bullish': 0.7,
    'BB_Breakout': 0.6,
    'StochRSI_Bull': 0.5,
    'OBV_Positive': 0.4,
    'Gap_Up': 0.3,
    'C_5d_Positive_Momentum': 0.5,
    'C_3d_Volume_Spike': 0.3
}


if os.path.exists(WEIGHTS_FILE):
    with open(WEIGHTS_FILE, 'r') as f:
        weights = json.load(f)
else:
    weights = default_weights.copy()
# === Reinforce Top 3 Conditions (Weekly Boost)
try:
    perf_df = pd.read_csv("condition_performance.csv")
    top_conditions = perf_df.sort_values(by="Win_Rate_%", ascending=False).head(3)['Condition'].tolist()

    for cond in top_conditions:
        if cond in weights:
            weights[cond] *= 1.2  # Boost by 20%
            weights[cond] = round(weights[cond], 4)

    print(f"🔥 Boosted weights for top performers: {top_conditions}")
    # Penalize bottom 2 poorly performing conditions
    bottom_conditions = perf_df.sort_values(by="Win_Rate_%", ascending=True).head(2)['Condition'].tolist()

    for cond in bottom_conditions:
        if cond in weights:
            weights[cond] *= 0.9  # Decrease by 10%
            weights[cond] = round(max(weights[cond], 0.05), 4)
            print(f"📉 Penalized {cond} (low win rate)")
except Exception as e:
    print(f"⚠️ Reinforcement failed: {e}")

# --- Alpaca API ---
API_KEY = 'PK0MW7WFYFJVXSY3IBM2'
API_SECRET = 'KYiN25E6xyS2gzoJMnW7hfXZ5ReXuqltbe7poIbl'
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# --- User Config ---
STOCK_CSV = r'C:\Users\DELL\Desktop\Hybrid stock model\Only volume\buying on the down\improving\top_300_stocks_by_volume.csv'
MAX_BUDGET = 500.0
MIN_HOLD_DAYS = 1
MAX_HOLD_DAYS = 7
# Banking settings
hold_buffer_days = 1  # Require holding at least 1 full day before banking
EARLY_EXIT_PCT = 7.0
PROFIT_TARGET = 20.0
STOP_LOSS = 4.0
DIP_MIN = -2.5
SLEEP_HOURS = 1
MAX_PER_TRADE = 50
MAX_SHARES_PER_STOCK = 5
DAILY_PROFIT_TARGET_DOLLARS = 5.0
DAILY_PROFIT_TARGET_PCT = 5.0
DAILY_PROFIT_TARGET = 5.0
SLIPPAGE_BPS = 0.001   # default slippage assumption (0.1%) used by multiple places
MAX_CANDIDATES = 10
# --- File Paths ---
IND_CSV = 'stock_indicators_volume.csv'
OHLCV_CSV = 'stock_prices_volume.csv'
PORTFOLIO = 'portfolio_volume.csv'
TRADES_LOG = 'Trade_Logs.csv'
BANKING_STATUS_FILE = "banking_status.txt"
## ===========================
# SECTION 2: HISTORICAL SETUP WITH SCORING
# ===========================
# Constants
RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_DELAY = 0.5  # seconds
MIN_DATA_POINTS = 25

# === Load symbols ===
df_stocks = pd.read_csv(STOCK_CSV)
tickers = df_stocks['Symbol'].dropna().unique().tolist()
ind_rows, price_rows, failed_symbols = [], [], []

print("📥 Downloading 60-day data from Yahoo Finance...")

for sym in tqdm(tickers):
    success = False

    for attempt in range(RETRIES):
        try:
            data = yf.download(
                tickers=sym,
                period='60d',
                interval='1d',
                progress=False,
                auto_adjust=True,
                timeout=15
            )

            if data.empty or len(data) < MIN_DATA_POINTS:
                raise ValueError("Data too short or empty")

            data.dropna(inplace=True)

            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs(sym, axis=1, level=1)

            if 'Close' not in data.columns:
                raise ValueError("Missing 'Close' column")

            close = data['Close'].astype(float)
            volume = data['Volume'].astype(float)
            high = data['High'].astype(float)
            low = data['Low'].astype(float)

            # === Indicators ===
            sma3 = SMAIndicator(close=close, window=3).sma_indicator()
            sma20 = SMAIndicator(close=close, window=20).sma_indicator()
            sma50 = SMAIndicator(close=close, window=50).sma_indicator()
            atr14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            rsi = RSIIndicator(close=close, window=14).rsi()
            ret5d = close.pct_change(5) * 100
            ret3d = close.pct_change(3) * 100
            # Calculate volume spike
            vol_avg_5d = volume.rolling(5).mean()
            vol_spike = (volume.iloc[-1] > vol_avg_5d.iloc[-1] * 1.5)
            avg_vol = volume.rolling(window=20).mean()
            price_change = close.pct_change(1) * 100
            # MACD
            macd = MACD(close=close)
            macd_diff = macd.macd_diff()

            # Bollinger Bands
            bb = BollingerBands(close=close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()

            # Stochastic RSI (using close since your RSI is already 14)
            stoch = StochasticOscillator(close=close, high=high, low=low, window=14)
            stoch_rsi = stoch.stoch()

            # OBV
            obv = OnBalanceVolumeIndicator(close=close, volume=volume)
            obv_val = obv.on_balance_volume()

            # Gap Up Detection (last open vs previous close)
            gap_up = int((data['Open'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] > 0.02)

            last = data.iloc[-1]
            last_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            # --- 21-day daily return moments (mu, sigma)
            returns_daily = close.pct_change().dropna()
            # fallback to 0 if not enough data
            mu_21d = float(returns_daily.tail(21).mean()) if not returns_daily.empty else 0.0
            sigma_21d = float(returns_daily.tail(21).std()) if not returns_daily.empty else 0.0
            # === Scoring Conditions ===
            conditions = {
                'C_SMA3_gt_SMA20': int(last['Close'] > sma20.iloc[-1]),
                'C_SMA20_gt_SMA50': int(sma20.iloc[-1] > sma50.iloc[-1]),
                'C_Ret5d_in_range': int(-2.5 < ret5d.iloc[-1] < 0),
                'C_ATR_not_too_high': int(atr14.iloc[-1] / last['Close'] < 0.04),
                'C_Volume_gt_250k': int(avg_vol.iloc[-1] > 250000),
                'C_Price_in_range': int(5 <= last['Close'] <= 100),
                'C_RSI_neutral': int(45 < last_rsi < 55),
                'MACD_Bullish': int(macd_diff.iloc[-1] > 0 and macd_diff.iloc[-2] <= 0),
                'BB_Breakout': int(close.iloc[-1] > bb_upper.iloc[-1]),
                'StochRSI_Bull': int(stoch_rsi.iloc[-1] > 0.8),
                'OBV_Positive': int(obv_val.iloc[-1] > obv_val.iloc[-5]),  # OBV rising
                'Gap_Up': gap_up,
                'C_5d_Positive_Momentum': int(ret5d.iloc[-1] > 2.0),  # >2% gain
                'C_3d_Volume_Spike': int(vol_spike)
            }

            weights = {
                'C_SMA3_gt_SMA20': 1.0,
                'C_SMA20_gt_SMA50': 0.8,
                'C_Ret5d_in_range': 0.6,
                'C_ATR_not_too_high': 0.5,
                'C_Volume_gt_250k': 0.4,
                'C_Price_in_range': 0.4,
                'C_RSI_neutral': 0.6,
                'MACD_Bullish': 0.7,
                'BB_Breakout': 0.6,
                'StochRSI_Bull': 0.5,
                'OBV_Positive': 0.4,
                'Gap_Up': 0.3,
                'C_5d_Positive_Momentum': 0.5,
                'C_3d_Volume_Spike': 0.3

            }

            weighted_score = sum(conditions[k] * weights[k] for k in conditions)
            max_score = sum(weights.values())
            total_score = round(weighted_score / max_score, 3)

            # === Add to indicator rows ===
            ind_rows.append({
                'Symbol': sym,
                'Date': data.index[-1].strftime('%Y-%m-%d'),
                'Price': round(float(last['Close']), 2),
                'SMA_3': round(float(sma3.iloc[-1]), 2),
                'SMA_20': round(float(sma20.iloc[-1]), 2),
                'SMA_50': round(float(sma50.iloc[-1]), 2),
                'ATR_14': round(float(atr14.iloc[-1]), 2),
                'Ret_5d': round(float(ret5d.iloc[-1]), 2),
                'Avg_Vol': int(avg_vol.iloc[-1]),
                'RSI_14': round(float(last_rsi), 2),
                'MACD_Diff': round(float(macd_diff.iloc[-1]), 4),
                'StochRSI': round(float(stoch_rsi.iloc[-1]), 2),
                'OBV': round(float(obv_val.iloc[-1]), 2),
                'BB_Upper': round(float(bb_upper.iloc[-1]), 2),
                'BB_Lower': round(float(bb_lower.iloc[-1]), 2),
                'Gap_Up': gap_up,
                'MACD_Bullish': conditions['MACD_Bullish'],
                'BB_Breakout': conditions['BB_Breakout'],
                'StochRSI_Bull': conditions['StochRSI_Bull'],
                'StochRSI_Bull': conditions['StochRSI_Bull'],
                'OBV_Positive': conditions['OBV_Positive'],
                'Ret_3d': round(float(ret3d.iloc[-1]), 2),
                'Vol_Spike_Ratio': round(float(volume.iloc[-1] / vol_avg_5d.iloc[-1]), 2),
                'Mu_21d': round(mu_21d, 6),
                'Sigma_21d': round(sigma_21d, 6),

                **conditions,
            })

            # === Add OHLCV rows ===
            for dt, r in data.iterrows():
                price_rows.append({
                    'Symbol': sym,
                    'Date': dt.strftime('%Y-%m-%d'),
                    'Open': r['Open'],
                    'High': r['High'],
                    'Low': r['Low'],
                    'Close': r['Close'],
                    'Volume': r['Volume']
                })

            success = True
            break

        except Exception as e:
            print(f"⚠️ Retry {attempt + 1}/3 for {sym} — {e}")
            time.sleep(RETRY_DELAY)

    if not success:
        print(f"❌ Failed to load data for {sym} after {RETRIES} attempts.")
        failed_symbols.append(sym)

    time.sleep(REQUEST_DELAY)

# Save failed symbols
if failed_symbols:
    pd.DataFrame(failed_symbols, columns=["Failed Symbols"]).to_csv("failed_downloads.csv", index=False)
    print(f"⚠️ {len(failed_symbols)} symbols failed. Saved to failed_downloads.csv.")

# Save outputs
# === Append instead of overwrite so we keep history ===
today_str = datetime.today().strftime('%Y-%m-%d')

# === Save outputs (with mu/sigma z-score integration into Total_Score) ===
today_str = datetime.today().strftime('%Y-%m-%d')

# Build combined indicators DataFrame (append history, avoid duplicate today's rows)
if os.path.exists(IND_CSV):
    old_ind = pd.read_csv(IND_CSV)
    old_ind = old_ind[old_ind['Date'] != today_str]  # remove today's if exists
    combined_ind = pd.concat([old_ind, pd.DataFrame(ind_rows)], ignore_index=True)
else:
    combined_ind = pd.DataFrame(ind_rows)

# --- Compute Mu/Sigma z-scores & scaled [0,1] features using only today's rows (universe normalization)
# Filter today's rows (the last date we just processed)
try:
    today_df = combined_ind[combined_ind['Date'] == today_str].copy()
    # if today's rows empty (edge case), pick the latest date present
    if today_df.empty:
        last_date = combined_ind['Date'].max()
        today_df = combined_ind[combined_ind['Date'] == last_date].copy()

    # compute mu/sigma mean/std across universe (use robust fallbacks)
    mu_mean = today_df['Mu_21d'].mean() if 'Mu_21d' in today_df.columns else 0.0
    mu_std = today_df['Mu_21d'].std(ddof=0) if 'Mu_21d' in today_df.columns else 0.0
    sigma_mean = today_df['Sigma_21d'].mean() if 'Sigma_21d' in today_df.columns else 0.0
    sigma_std = today_df['Sigma_21d'].std(ddof=0) if 'Sigma_21d' in today_df.columns else 0.0

    # avoid zero-division
    mu_std = mu_std if mu_std > 0 else 1e-9
    sigma_std = sigma_std if sigma_std > 0 else 1e-9

    # z-score -> clipped [-3,3] -> scaled to [0,1]
    def scaled_from_z(x, mean, std):
        z = (x - mean) / std
        z = max(-3.0, min(3.0, z))
        return (z + 3.0) / 6.0

    # add continuous scores to today's frame
    today_df['mu_score'] = today_df['Mu_21d'].apply(lambda x: scaled_from_z(x, mu_mean, mu_std))
    # for volatility, lower sigma is better -> invert scaled sigma
    today_df['sigma_scaled'] = today_df['Sigma_21d'].apply(lambda x: scaled_from_z(x, sigma_mean, sigma_std))
    today_df['sigma_score'] = 1.0 - today_df['sigma_scaled']

    # Choose continuous weights (small so they augment, not dominate)
    w_mu = 0.30
    w_sigma = 0.20

    # ===== Replace hard-coded binary_weights with learned 'weights' (fallback to default_weights) =====
    try:
        # 'weights' is loaded/updated earlier; use a copy so we don't accidentally mutate it later
        binary_weights = weights.copy()
    except Exception:
        binary_weights = default_weights.copy()


    # compute binary weighted score (per row)
    # ensure missing columns default to 0
    def binary_weighted_score(row):
        s = 0.0
        for cond, w in binary_weights.items():
            try:
                val = float(row.get(cond, 0))
            except:
                val = 0.0
            s += val * w
        return s

    today_df['binary_weighted'] = today_df.apply(binary_weighted_score, axis=1)

    # normalization denominator
    denom = sum(binary_weights.values()) + w_mu + w_sigma

    # Final total score: combine binary + continuous, then normalize
    today_df['Total_Score'] = today_df.apply(
        lambda r: round((r['binary_weighted'] + w_mu * r['mu_score'] + w_sigma * r['sigma_score']) / denom, 3),
        axis=1
    )

    # --- Clean any prior leftover score columns to avoid duplicate-column merge collisions ---
    for c in ['Total_Score', 'mu_score', 'sigma_score']:
        if c in combined_ind.columns:
            combined_ind = combined_ind.drop(columns=[c])
    # Now safe to merge today's computed scores
    combined_ind = combined_ind.merge(
        today_df[['Symbol', 'Date', 'Total_Score', 'mu_score', 'sigma_score']],
        on=['Symbol', 'Date'],
        how='left'
    )

    # Fill any remaining NaNs (older rows) with previous Total_Score if present, else 0
    if 'Total_Score_x' in combined_ind.columns and 'Total_Score_y' in combined_ind.columns:
        # handle merge name collisions (if any)
        combined_ind['Total_Score'] = combined_ind['Total_Score_y'].fillna(combined_ind['Total_Score_x'])
        combined_ind.drop(['Total_Score_x', 'Total_Score_y'], axis=1, inplace=True)

    if 'Total_Score' not in combined_ind.columns:
        combined_ind['Total_Score'] = combined_ind.get('Total_Score', 0.0)

except Exception as e:
    print(f"⚠️ Could not compute mu/sigma z-scores and Total_Score: {e}")
    # fallback: compute Total_Score from binary only if something failed
    try:
        bw = weights if 'weights' in globals() else default_weights
        combined_ind['binary_weighted'] = combined_ind.apply(lambda r: sum(float(r.get(c,0))*bw.get(c,0) for c in bw), axis=1)
        combined_ind['Total_Score'] = combined_ind['binary_weighted'] / sum(bw.values())
    except:
        combined_ind['Total_Score'] = 0.0

# Finally save indicators to CSV
combined_ind.to_csv(IND_CSV, index=False)

# OHLCV CSV
if os.path.exists(OHLCV_CSV):
    old_price = pd.read_csv(OHLCV_CSV)
    old_price = old_price[old_price['Date'] != today_str]
    combined_price = pd.concat([old_price, pd.DataFrame(price_rows)], ignore_index=True)
else:
    combined_price = pd.DataFrame(price_rows)
combined_price.to_csv(OHLCV_CSV, index=False)

# ===========================
# SHOW STOCKS WITH HIGH SCORE (>= 0.75)
# ===========================
# ===== Replace existing top_picks creation with this (uses combined_ind which has Total_Score) =====
# Use today's scored rows (fall back to IND_CSV if needed)
try:
    top_picks = combined_ind[combined_ind['Date'] == today_str].copy()
    if top_picks.empty:
        # fallback: read IND_CSV in case combined_ind is not current
        top_picks = pd.read_csv(IND_CSV)
        top_picks = top_picks[top_picks['Date'] == today_str]
except Exception:
    # final fallback: build DataFrame and ensure Total_Score exists
    top_picks = pd.DataFrame(ind_rows)
    if 'Total_Score' not in top_picks.columns:
        top_picks['Total_Score'] = 0.0

# === Dynamic threshold filter ===
if not top_picks.empty:
    dynamic_thresh = max(0.65, top_picks['Total_Score'].quantile(0.85))  # top 15% or at least 0.65
    top_picks = top_picks[top_picks['Total_Score'] >= dynamic_thresh]
else:
    top_picks = pd.DataFrame()  # keep consistent type if empty



if not top_picks.empty:
    print("\n⭐ Stocks with Total_Score >= 0.70:")
    print(top_picks[['Symbol', 'Price', 'Total_Score']].sort_values(by='Total_Score', ascending=False).to_string(index=False))
else:
    print("\n⚠️ No stocks found with a score >= 0.70.")

print("✅ Historical indicators and scoring saved.")


# ===========================
# SECTION 3: Portfolio Setup
# ===========================
if os.path.exists(PORTFOLIO):
    port_df = pd.read_csv(PORTFOLIO, parse_dates=['Buy_Date'])
else:
    port_df = pd.DataFrame(columns=['Symbol', 'Buy_Date', 'Buy_Price', 'Qty'])

if not os.path.exists(TRADES_LOG):
    with open(TRADES_LOG, 'w', encoding='utf-8') as f:
        f.write('Action,Symbol,Time,Price,Qty,Reason,Spread,Mu_21d,Sigma_21d,F_star,Alloc,Kelly_f\n')


def days_held(buy_date):
    if isinstance(buy_date, str):
        buy_date = parser.parse(buy_date)
    return (date.today() - buy_date.date()).days
# ===========================
# Dynamic Hold Duration Function
# ===========================
def get_dynamic_hold_days(rsi, atr, sma_short, sma_long):
    hold = 2/24
    if rsi >= 45 and rsi <= 55 and atr < 0.04:
        if sma_short > sma_long:
            hold = 5
        else:
            hold = 3
    if sma_short > sma_long and rsi >= 48 and atr < 0.03:
        hold = 6
    return min(MAX_HOLD_DAYS, hold)

if datetime.today().weekday() == 0:  # Monday = 0
    with open("weekly_profit_log.csv", "a") as f:
        f.write(f"\n--- New Week: {datetime.today().strftime('%Y-%m-%d')} ---\n")
    print("🔁 Weekly profit log started for new week.")
# ===========================
# SECTION 3.5: Learn from Past Trades
# ===========================
try:
    trades = pd.read_csv(TRADES_LOG)
    indicators = pd.read_csv(IND_CSV)

    sell_trades = trades[trades['Action'] == 'SELL']
    deltas = {k: 0 for k in default_weights}

    for _, row in sell_trades.iterrows():
        sym, time_str, pnl = row['Symbol'], row['Time'], float(row['Spread'])
        trade_date = pd.to_datetime(time_str).strftime('%Y-%m-%d')

        # Look up indicator row
        match = indicators[(indicators['Symbol'] == sym) & (indicators['Date'] == trade_date)]
        if match.empty:
            continue

        match = match.iloc[0]
        for cond in deltas:
            if cond in match and match[cond] == 1:
                if pnl > 0:
                    deltas[cond] += 0.05
                else:
                    deltas[cond] -= 0.05

    # Update weights
    # === Volatile indicators (learn faster)
    volatile_indicators = ['MACD_Bullish', 'BB_Breakout', 'StochRSI_Bull', 'Gap_Up']

    for k in weights:
        change = deltas[k]
        # Faster learning for volatile indicators
        if k in volatile_indicators:
            change *= 1.5  # 50% faster adjustment

        new_weight = weights[k] + change
        weights[k] = round(min(2.0, max(0.05, new_weight)), 3)


    # Save
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=2)
    print("🧠 Weights updated based on trade performance.")

except Exception as e:
    print(f"❌ Learning error: {e}")
# ===========================
# SECTION 3.6: Strategy Leaderboard
# ===========================
try:
    condition_stats = []
    condition_cols = [c for c in default_weights]

    for cond in condition_cols:
        wins, losses, total_pnl = 0, 0, []

        for _, row in sell_trades.iterrows():
            sym, time_str, pnl = row['Symbol'], row['Time'], float(row['Spread'])
            trade_date = pd.to_datetime(time_str).strftime('%Y-%m-%d')
            match = indicators[(indicators['Symbol'] == sym) & (indicators['Date'] == trade_date)]

            if match.empty:
                continue

            match_row = match.iloc[0]
            if cond in match_row and match_row[cond] == 1:
                total_pnl.append(pnl)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        if wins + losses > 0:
            avg_pnl = round(np.mean(total_pnl), 2)
            winrate = round((wins / (wins + losses)) * 100, 2)
        else:
            avg_pnl = 0
            winrate = 0

        condition_stats.append({
            "Condition": cond,
            "Wins": wins,
            "Losses": losses,
            "Win_Rate_%": winrate,
            "Avg_PnL": avg_pnl
        })

    perf_df = pd.DataFrame(condition_stats)
    perf_df = perf_df.sort_values(by="Win_Rate_%", ascending=False)
    perf_df.to_csv("condition_performance.csv", index=False)
    print("📊 Saved condition performance leaderboard → condition_performance.csv")

except Exception as e:
    print(f"❌ Leaderboard error: {e}")

def approve_sell(sym, buy_date, reason="Banking", hold_buffer_days=1):
    """
    Checks if a stock can be sold based on holding rules & strategy.
    - Before Wednesday: must meet dynamic hold days
    - Wednesday onward: must meet hold_buffer_days
    Applies normal selling logic in both cases.
    """
    dh = days_held(buy_date)
    weekday = datetime.today().weekday()  # Monday=0, Sunday=6

    # Before Wednesday — dynamic hold rules
    if weekday < 2:  # Mon(0) or Tue(1)
        # Get stock's dynamic hold requirement
        try:
            port_df = pd.read_csv(PORTFOLIO, parse_dates=['Buy_Date'])
            dyn_hold = port_df.loc[port_df['Symbol'] == sym, 'Dynamic_Hold_Days']
            dyn_hold_days = int(dyn_hold.iloc[0]) if not dyn_hold.empty else 2
        except Exception as e:
            print(f"⚠️ Could not get dynamic hold days for {sym}: {e}")
            dyn_hold_days = 2

        # Enforce min hold before selling
        if dh < dyn_hold_days:
            print(f"⏳ Skipping {sym} {reason}: Only held {dh} days (< dynamic {dyn_hold_days})")
            return False

    # From Wednesday onward — simpler hold buffer rule
    else:
        if dh < hold_buffer_days:
            print(f"⏳ Skipping {sym} {reason}: Only held {dh} days (< {hold_buffer_days})")
            return False

    # === Strategy approval (score check) ===
    if os.path.exists(IND_CSV):
        try:
            ind_df = pd.read_csv(IND_CSV)
            score = ind_df.loc[ind_df['Symbol'] == sym, 'Total_Score']
            if not score.empty and score.iloc[0] >= 0.65:
                print(f"⚠️ Skipping {sym} {reason}: Still high scoring ({score.iloc[0]:.3f})")
                return False
        except Exception as e:
            print(f"⚠️ Could not check strategy approval for {sym}: {e}")

    return True


def mine_weekly_patterns(indicator_file=IND_CSV, output_file="weekly_success_patterns.csv"):
    try:
        df = pd.read_csv(indicator_file, parse_dates=['Date'])
        condition_cols = list(default_weights.keys())
        pattern_rows = []

        # Calculate last full week dates (Monday-Friday)
        today = datetime.today()
        last_monday = today - timedelta(days=today.weekday() + 7)
        last_friday = last_monday + timedelta(days=4)

        print(f"🔍 Analyzing last week: {last_monday.strftime('%Y-%m-%d')} to {last_friday.strftime('%Y-%m-%d')}")

        # Filter to last week's data
        weekly_data = df[
            (df['Date'] >= last_monday.strftime('%Y-%m-%d')) &
            (df['Date'] <= last_friday.strftime('%Y-%m-%d'))
        ]

        if weekly_data.empty:
            print("⚠️ No data for last week in indicators file. Pattern mining skipped.")
            return

        # Get top 15 performers of the week
        top_performers = []
        for symbol in weekly_data['Symbol'].unique():
            stock_data = weekly_data[weekly_data['Symbol'] == symbol].sort_values('Date')

            if len(stock_data) < 2:
                continue

            monday_data = stock_data[stock_data['Date'] == last_monday.strftime('%Y-%m-%d')]
            friday_data = stock_data[stock_data['Date'] == last_friday.strftime('%Y-%m-%d')]

            if monday_data.empty or friday_data.empty:
                continue

            monday_row = monday_data.iloc[0]
            friday_row = friday_data.iloc[0]
            weekly_return = (friday_row['Price'] - monday_row['Price']) / monday_row['Price'] * 100

            top_performers.append({
                'Symbol': symbol,
                'Return': weekly_return,
                'Monday_Conditions': tuple(int(monday_row[col]) for col in condition_cols)
            })

        # Sort by best performance
        top_performers.sort(key=lambda x: x['Return'], reverse=True)
        top_15 = top_performers[:15]

        # Save patterns
        for stock in top_15:
            pattern_rows.append({
                "Symbol": stock['Symbol'],
                "Return": stock['Return'],
                "Combo": str(stock['Monday_Conditions'])
            })

        if pattern_rows:
            pattern_df = pd.DataFrame(pattern_rows)
            pattern_df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(pattern_df)} top patterns from last week")

            # Indicator frequency boost
            indicator_counts = {col: 0 for col in condition_cols}
            for stock in top_15:
                for i, col in enumerate(condition_cols):
                    if stock['Monday_Conditions'][i] == 1:
                        indicator_counts[col] += 1

            for col, count in indicator_counts.items():
                if count >= 5:  # appeared in >33% of top performers
                    boost_factor = 1 + (count / 15) * 0.3
                    weights[col] = min(2.0, weights[col] * boost_factor)
                    weights[col] = round(weights[col], 4)
                    print(f"🚀 Boosted {col} by {((boost_factor - 1) * 100):.1f}% (appeared {count}/15 times)")

            with open(WEIGHTS_FILE, 'w') as f:
                json.dump(weights, f, indent=2)
        else:
            print("⚠️ No top patterns found for last week")

    except Exception as e:
        print(f"❌ Weekly pattern miner error: {e}")
def load_banking_status():
    """Load today's banking status and last_banked_equity."""
    if os.path.exists(BANKING_STATUS_FILE):
        with open(BANKING_STATUS_FILE, "r") as f:
            data = f.read().strip().split(",")
            if len(data) == 3:
                saved_date, done_today, last_equity = data
                if saved_date == datetime.now().strftime("%Y-%m-%d"):
                    return done_today.lower() == "true", float(last_equity)
    return False, None  # No banking done today

def save_banking_status(done_today, last_equity):
    """Save today's banking status and last equity."""
    with open(BANKING_STATUS_FILE, "w") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d')},{done_today},{last_equity:.2f}")

def trigger_banking_all(top_buy_candidates):
    banked_file = "banked_cash.txt"
    if os.path.exists(banked_file):
        try:
            with open(banked_file, "r") as f:
                banked_cash = float(f.read().strip() or 0.0)
        except:
            banked_cash = 0.0
    else:
        banked_cash = 0.0

    if os.path.exists(PORTFOLIO):
        port_df = pd.read_csv(PORTFOLIO, parse_dates=['Buy_Date'])
        if not port_df.empty:
            port_df['Qty'] = pd.to_numeric(port_df['Qty'], errors='coerce').fillna(0.0)
            port_df['Buy_Price'] = pd.to_numeric(port_df['Buy_Price'], errors='coerce').fillna(0.0)
    else:
        port_df = pd.DataFrame(columns=['Symbol', 'Buy_Date', 'Buy_Price', 'Qty'])

    try:
        positions = api.list_positions()
    except Exception as e:
        print(f"⚠️ Could not fetch Alpaca positions: {e}")
        positions = []

    top_symbols = set(top_buy_candidates['Symbol'].tolist()) if (
        top_buy_candidates is not None and not top_buy_candidates.empty) else set()

    for pos in positions:
        sym = pos.symbol
        try:
            available_qty = float(pos.qty)
            live_price = float(pos.current_price)
            avg_entry = float(getattr(pos, 'avg_entry_price', 0.0) or 0.0)
        except Exception:
            print(f"⚠️ Skipping {sym}: could not read position values")
            continue

        if sym in top_symbols:
            print(f"⏭ Skipping {sym} (in top buy candidates)")
            continue

        if avg_entry == 0.0 and sym in port_df['Symbol'].values:
            try:
                avg_entry = float(port_df.loc[port_df['Symbol'] == sym, 'Buy_Price'].iloc[0])
            except:
                avg_entry = 0.0

        total_pnl = round((live_price - avg_entry) * available_qty, 2) if avg_entry != 0 else 0.0
        gain_pct = ((live_price - avg_entry) / avg_entry * 100) if avg_entry else 0.0

        if total_pnl <= 0:
            print(f"⏭ Skipping {sym}: not profitable (PnL {total_pnl:.2f})")
            continue

        buy_date = None
        if sym in port_df['Symbol'].values:
            buy_date = port_df.loc[port_df['Symbol'] == sym, 'Buy_Date'].iloc[0]
        if not approve_sell(sym, buy_date, reason="Banking"):
            continue

        try:
            # ✅ Safe clamping and rounding
            safe_qty = Decimal(str(available_qty)).quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
            if safe_qty <= 0:
                print(f"⚠️ Skipping {sym} — no sellable qty for banking")
                continue

            api.submit_order(
                symbol=sym,
                side='sell',
                type='market',
                qty=float(safe_qty),
                time_in_force='day'
            )
            print(f"✅ Sold {safe_qty} shares of {sym} for banking (Gain: {gain_pct:.2f}%)")

            banked_cash = round(banked_cash + total_pnl, 2)
            with open(banked_file, "w") as f:
                f.write(str(banked_cash))

            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(TRADES_LOG, 'a') as f:
                f.write(f"SELL,{sym},{now_str},{live_price},{safe_qty},Banking,{total_pnl}\n")

            if sym in port_df['Symbol'].values:
                port_df = port_df[port_df['Symbol'] != sym]
                port_df.to_csv(PORTFOLIO, index=False)
                print(f"🗑 Removed {sym} from portfolio.csv")

            try:
                recalc_used_avail()
            except Exception:
                pass

        except Exception as e:
            print(f"❌ Could not bank {sym}: {e}")

    print(f"💰 Banking run complete. Total banked cash: ${banked_cash:.2f}")


##Dashboard##
# === Create Weekly Top Strategies CSV
top_df = perf_df.sort_values(by="Win_Rate_%", ascending=False).head(3)
top_df.to_csv("top_strategies_weekly.csv", index=False)
print("📤 Exported weekly top strategies → top_strategies_weekly.csv")

# === Generate Dashboard HTML
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Strategy Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; }
        table { border-collapse: collapse; width: 80%; margin: auto; background: #fff; }
        th, td { padding: 12px; border: 1px solid #ccc; text-align: center; }
        th { background-color: #0074D9; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .top { background-color: #2ECC40; color: #fff; font-weight: bold; }
    </style>
</head>
<body>
    <h2 style="text-align:center;">📊 Strategy Condition Dashboard</h2>
    <table>
        <tr>
            <th>Condition</th>
            <th>Wins</th>
            <th>Losses</th>
            <th>Win Rate (%)</th>
            <th>Avg PnL</th>
        </tr>
        {% for row in data %}
        <tr class="{{ 'top' if row['Condition'] in top_conditions else '' }}">
            <td>{{ row['Condition'] }}</td>
            <td>{{ row['Wins'] }}</td>
            <td>{{ row['Losses'] }}</td>
            <td>{{ row['Win_Rate_%'] }}</td>
            <td>{{ row['Avg_PnL'] }}</td>
        </tr>
        {% endfor %}
    </table>
    <p style="text-align:center;">Updated on {{ now }}</p>
</body>
</html>
"""

# Generate the HTML dashboard
dashboard_data = perf_df.to_dict(orient='records')
template = Template(html_template)
html = template.render(data=dashboard_data, top_conditions=top_df['Condition'].tolist(), now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with open("dashboard.html", "w", encoding='utf-8') as f:
    f.write(html)

print("📈 Dashboard saved as dashboard.html")

# ===========================
# STRATEGY DASHBOARD EXPORTER
# ===========================
def export_dashboard(trade_log_path=TRADES_LOG, port_path=PORTFOLIO, out_path="dashboard_summary.csv"):
    if not os.path.exists(trade_log_path):
        print("📭 No trades to summarize.")
        return

    df = pd.read_csv(trade_log_path)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Date'] = df['Time'].dt.date

    summary = {}

    # --- 1. Weekly PnL ---
    weekly = df[df['Action'] == 'SELL'].copy()
    weekly['PnL'] = weekly['Spread']
    weekly['Week'] = weekly['Time'].dt.strftime('%Y-%U')
    weekly_pnl = weekly.groupby('Week')['PnL'].sum().reset_index()
    summary['Weekly_PnL'] = weekly_pnl

    # --- 2. Top Reasons (Strategies) ---
    df['PatternMatch'] = df['Reason'].str.contains("PatternMatch", case=False)
    reasons = df[df['Action'] == 'BUY']['Reason'].value_counts().reset_index()
    reasons.columns = ['Strategy_Reason', 'Count']

    # Save PatternMatch count separately (optional)
    pattern_count = df[df['PatternMatch'] == True].shape[0]
    with open(out_path, 'a') as f:
        f.write(f"\n\nPatternMatch Trades,{pattern_count}\n")

    reasons.columns = ['Strategy_Reason', 'Count']
    summary['Top_Strategies'] = reasons

    # --- 3. Portfolio Stats ---
    if os.path.exists(port_path):
        pf = pd.read_csv(port_path)
        if not pf.empty:
            pf_stats = {
                "Total Positions": len(pf),
                "Invested Amount": round((pf['Buy_Price'] * pf['Qty']).sum(), 2),
                "Avg Buy Price": round(pf['Buy_Price'].mean(), 2),
                "Oldest Buy Date": pf['Buy_Date'].min()
            }
            summary['Portfolio_Stats'] = pf_stats

    # --- 4. Save to CSV ---
    with open(out_path, 'w') as f:
        f.write("=== Weekly PnL ===\n")
        summary['Weekly_PnL'].to_csv(f, index=False)
        f.write("\n\n=== Top Strategies ===\n")
        summary['Top_Strategies'].to_csv(f, index=False)
        f.write("\n\n=== Portfolio Stats ===\n")
        for k, v in summary.get('Portfolio_Stats', {}).items():
            f.write(f"{k},{v}\n")

    print(f"📊 Dashboard summary exported to {out_path}")

def is_tradable_and_fractionable(symbol, api):
    try:
        asset = api.get_asset(symbol)
        return asset.tradable and asset.fractionable
    except:
        return False
def load_and_clean_portfolio():
    """Load portfolio, clean NaN/NaT, recalc used & avail."""
    global used, avail

    if os.path.exists(PORTFOLIO):
        port_df = pd.read_csv(PORTFOLIO)

        # Drop rows without a valid symbol
        port_df = port_df.dropna(subset=['Symbol'])
        port_df = port_df[port_df['Symbol'].astype(str).str.strip() != ""]

        # Fix date column (NaT if invalid)
        if 'Buy_Date' in port_df.columns:
            port_df['Buy_Date'] = pd.to_datetime(port_df['Buy_Date'], errors='coerce')

        # Ensure numeric values for Qty and Buy_Price
        if 'Qty' in port_df.columns:
            port_df['Qty'] = pd.to_numeric(port_df['Qty'], errors='coerce').fillna(0.0)
        if 'Buy_Price' in port_df.columns:
            port_df['Buy_Price'] = pd.to_numeric(port_df['Buy_Price'], errors='coerce').fillna(0.0)

        # Remove rows where Qty <= 0
        port_df = port_df[port_df['Qty'] > 0]

        # Save cleaned file
        port_df.to_csv(PORTFOLIO, index=False)

        # Calculate used and available budget
        used = (port_df['Buy_Price'] * port_df['Qty']).sum()
        avail = max(0.0, MAX_BUDGET - used)

    else:
        port_df = pd.DataFrame(columns=['Symbol', 'Buy_Date', 'Buy_Price', 'Qty'])
        used = 0.0
        avail = MAX_BUDGET

    return port_df

# ===========================
# SECTION 4: Main Loop with Weighted Scoring
# ===========================
ind_df = pd.read_csv(IND_CSV)
ind_df = ind_df.sort_values(by='Total_Score', ascending=False)
print("\U0001F501 Starting hourly trading loop. Press Ctrl+C to stop.")

## section 4.1
# ===========================
# NEW SECTION: Weekly Profit Target Tracker
# ===========================
def check_weekly_progress():
    try:
        if not os.path.exists(TRADES_LOG):
            return  # No trades yet

        df = pd.read_csv(TRADES_LOG)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time'])

        df['Week'] = df['Time'].dt.isocalendar().week
        df['Year'] = df['Time'].dt.year
        df['Day'] = df['Time'].dt.day_name()

        # Only use SELL trades with valid Spread
        sell_df = df[df['Action'] == 'SELL'].copy()
        sell_df['PnL'] = pd.to_numeric(sell_df['Spread'], errors='coerce')
        sell_df = sell_df.dropna(subset=['PnL'])

        week_now = datetime.today().isocalendar().week
        year_now = datetime.today().year

        week_df = sell_df[(sell_df['Week'] == week_now) & (sell_df['Year'] == year_now)]

        total_pnl = week_df['PnL'].sum()
        target = round(MAX_BUDGET * 0.04, 2)
        pct = round((total_pnl / MAX_BUDGET) * 100, 2)

        status = "✅ On Track" if pct >= ((datetime.today().weekday() + 1) / 5) * 4 else "⚠️ Behind"
        log = f"{date.today()},{total_pnl:.2f},{pct:.2f},{status}"

        print(f"📊 Weekly PnL: ${total_pnl:.2f} | {pct}% of Target ({target}) → {status}")

        # Avoid duplicate logs for today
        log_path = "weekly_pnl_log.csv"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding='utf-8') as f:
                lines = f.readlines()
            lines = [line for line in lines if not line.startswith(str(date.today()))]
        else:
            lines = []

        lines.append(log + "\n")

        with open(log_path, "w", encoding='utf-8') as f:
            f.writelines(lines)

    except Exception as e:
        print(f"❗ Weekly PnL Check Error: {e}")


# Load indicators before main loop
if os.path.exists(IND_CSV):
    ind_df = pd.read_csv(IND_CSV)
else:
    # Run your indicator calculation if needed
    print("⚠️ Indicator file missing, running indicator calculation...")
    # [Add your indicator calculation code here]
    ind_df = pd.read_csv(IND_CSV)
def recalc_used_avail():
    global used, avail
    # Clean portfolio first
    if os.path.exists(PORTFOLIO):
        tmp = pd.read_csv(PORTFOLIO)

        # === CLEAN STEP ===
        tmp = tmp.dropna(subset=['Symbol'])
        tmp = tmp[tmp['Symbol'].astype(str).str.strip() != ""]
        if 'Buy_Date' in tmp.columns:
            tmp['Buy_Date'] = pd.to_datetime(tmp['Buy_Date'], errors='coerce')
        if 'Qty' in tmp.columns:
            tmp['Qty'] = pd.to_numeric(tmp['Qty'], errors='coerce').fillna(0.0)
        if 'Buy_Price' in tmp.columns:
            tmp['Buy_Price'] = pd.to_numeric(tmp['Buy_Price'], errors='coerce').fillna(0.0)
        tmp = tmp[tmp['Qty'] > 0]  # remove zero qty

        # Save cleaned portfolio
        tmp.to_csv(PORTFOLIO, index=False)

        used = float((tmp['Buy_Price'] * tmp['Qty']).sum()) if not tmp.empty else 0.0
    else:
        used = 0.0

    avail = max(0.0, MAX_BUDGET - used)
# --- Drawdown guard helpers ---
PEAK_FILE = "equity_peak.txt"
DRAW_THRESHOLD = 0.12   # 12% drawdown triggers defensive mode

def read_equity_peak():
    if os.path.exists(PEAK_FILE):
        try:
            with open(PEAK_FILE, "r") as f:
                return float(f.read().strip() or 0.0)
        except:
            return 0.0
    return 0.0

def update_equity_peak(current_equity):
    peak = read_equity_peak()
    if current_equity > peak:
        with open(PEAK_FILE, "w") as f:
            f.write(str(round(current_equity, 2)))
        return current_equity
    return peak

def check_drawdown_mode(current_equity):
    peak = read_equity_peak() or current_equity
    draw = (peak - current_equity) / peak if peak > 0 else 0.0
    return draw >= DRAW_THRESHOLD
# ----------------- MVO daily-run helper (paste BEFORE the hourly loop) -----------------
import subprocess, os, pandas as pd, traceback

MVO_SCRIPT = "mvo_allocator.py"        # adjust if in different folder
MVO_OUTPUT = "mvo_weights.csv"
MVO_LAST_RUN = "mvo_last_run.txt"

def run_mvo_if_needed(verbose=True):
    today = date.today().isoformat()
    last = None
    if os.path.exists(MVO_LAST_RUN):
        try:
            last = open(MVO_LAST_RUN, "r").read().strip()
        except:
            last = None

    if last == today:
        if verbose: print(f"🔁 MVO already run today ({today}), loading weights...")
    else:
        if verbose: print(f"⚙️ Running daily MVO allocator ({MVO_SCRIPT}) for {today} ...")
        try:
            # prefer importing if you put mvo_allocator as a module, else call subprocess
            # run mvo and let errors print to console so failures are visible (do not mute stdout/stderr)
            subprocess.run([sys.executable, MVO_SCRIPT], check=True)
            open(MVO_LAST_RUN, "w").write(today)
            if verbose: print("✅ MVO completed and saved.")
        except Exception as e:
            print("❌ MVO run failed:", e)
            traceback.print_exc()
            # touch last-run only if you explicitly want; here we don't so it retries next run

    # load mvo_weights.csv into dict map
    mvo_map = {}
    try:
        if os.path.exists(MVO_OUTPUT):
            mvo_df = pd.read_csv(MVO_OUTPUT)
            # choose column to use: use 'MVO_Weight_ScaledVol' if present, else 'MVO_Weight'
            if 'MVO_Weight_ScaledVol' in mvo_df.columns:
                col = 'MVO_Weight_ScaledVol'
            elif 'MVO_Weight' in mvo_df.columns:
                col = 'MVO_Weight'
            else:
                # fallback to RP_Weight or uniform
                col = 'RP_Weight' if 'RP_Weight' in mvo_df.columns else None

            if col:
                mvo_map = dict(zip(mvo_df['Symbol'].astype(str), mvo_df[col].astype(float)))
            else:
                # fallback uniform
                syms = mvo_df['Symbol'].astype(str).tolist()
                mvo_map = {s: 1.0/len(syms) for s in syms}
            if verbose: print(f"📥 Loaded {len(mvo_map)} MVO weights from {MVO_OUTPUT}")
        else:
            if verbose: print(f"⚠️ {MVO_OUTPUT} not found; continuing without MVO.")
    except Exception as e:
        print("❌ Error loading mvo_weights.csv:", e)
        traceback.print_exc()
    return mvo_map

# run it once here and keep the map for the loop
mvo_map = run_mvo_if_needed()

# Call the function once before the loop starts
check_weekly_progress()
# 🧠 Run pattern miner before loop
mine_weekly_patterns()
pattern_miner_run = False
## section 4.2
while True:
    try:
        # ==================== ADD THIS ====================
        # Reload indicators at start of each loop iteration
        print("\n🔄 Refreshing indicator data...")
        if os.path.exists(IND_CSV):
            ind_df = pd.read_csv(IND_CSV)
            ind_df = ind_df.drop_duplicates(subset=['Symbol'], keep='last')

            # Sort by Total_Score descending
            ind_df = ind_df.sort_values(by='Total_Score', ascending=False)
            print(f"✅ Loaded {len(ind_df)} indicators")
        else:
            print(f"⚠️ Indicator file {IND_CSV} not found! Using empty DataFrame")
            ind_df = pd.DataFrame(columns=['Symbol', 'Price', 'Total_Score'])

        port_df = pd.read_csv(PORTFOLIO, parse_dates=['Buy_Date']) if os.path.exists(PORTFOLIO) else pd.DataFrame(
            columns=['Symbol', 'Buy_Date', 'Buy_Price', 'Qty']
        )

        # Load mined patterns (once per loop) — SAFE: creates `pattern_df` and `winning_combos`
        try:
            pattern_df = pd.read_csv("weekly_success_patterns.csv")

            # Ensure Combo column exists and convert safely from string to tuple
            if 'Combo' in pattern_df.columns:
                def safe_parse_combo(x):
                    try:
                        if isinstance(x, str):
                            # eval string representation like "(1,0,1,...)"
                            parsed = eval(x)
                            # ensure tuple
                            return tuple(parsed) if isinstance(parsed, (list, tuple)) else tuple()
                        elif isinstance(x, (list, tuple)):
                            return tuple(x)
                        else:
                            return tuple()
                    except Exception:
                        return tuple()


                pattern_df['Combo'] = pattern_df['Combo'].apply(safe_parse_combo)
            else:
                pattern_df['Combo'] = [tuple()] * len(pattern_df)

            # Build a set of winning combos for quick membership testing
            winning_combos = set(pattern_df['Combo'].dropna().tolist())

            # print a notice if no combos found (respect optional global 'verbose' flag)
            if not winning_combos:
                if globals().get('verbose', True):
                    print("⚠️ weekly_success_patterns.csv loaded but no combos found.")

        except Exception as e:
            print(f"❗ Could not load pattern file: {e}")
            pattern_df = pd.DataFrame(columns=['Symbol', 'Combo'])
            winning_combos = set()

        # ✅ Recalculate used & available budget
        recalc_used_avail()

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n🕒 {now} | Budget Left: ${avail:.2f}")
        # Before doing buys/sells each loop
        port_df = load_and_clean_portfolio()

        ## section 4.3:
        # --- Adjust strategy if behind weekly target after Wednesday ---
        strategy_boost = False
        try:
            df = pd.read_csv(TRADES_LOG)
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df['Week'] = df['Time'].dt.isocalendar().week
            df['Year'] = df['Time'].dt.year
            df['Day'] = df['Time'].dt.day_name()

            sell_df = df[df['Action'] == 'SELL'].copy()
            sell_df['PnL'] = pd.to_numeric(sell_df['Spread'], errors='coerce')
            week_df = sell_df[
                (sell_df['Week'] == datetime.today().isocalendar().week) &
                (sell_df['Year'] == datetime.today().year)
                ]
            total_pnl = week_df['PnL'].sum()
            pct = round((total_pnl / MAX_BUDGET) * 100, 2)

            if datetime.today().weekday() >= 2 and pct < ((datetime.today().weekday() + 1) / 5) * 4:
                if not globals().get("strategy_already_adjusted", False):
                    strategy_boost = True
                    strategy_already_adjusted = True
                    print("⚠️ Behind weekly target — boosting strategy AND lowering threshold...")
            else:
                strategy_boost = False
                strategy_already_adjusted = False

            # If it's Wednesday (2) or later and we're behind, trigger adjustment
            if datetime.today().weekday() >= 2 and pct < ((datetime.today().weekday() + 1) / 5) * 4:
                strategy_boost = True
                print(f"⚠️ Behind weekly target! Boosting strategy...")

                # Sell all deadweight
                for i, row in port_df.iterrows():
                    try:
                        entry = port_df[port_df['Symbol'] == row['Symbol']].iloc[0]
                        sym = entry['Symbol']
                        bp = float(entry['Buy_Price'])
                        qty = float(entry['Qty'])
                        dh = days_held(entry['Buy_Date'])  # Ensure days_held returns int

                        # Skip if held less than hold_buffer_days
                        if not approve_sell(sym, entry['Buy_Date'], reason="ForceSell",
                                            hold_buffer_days=hold_buffer_days):
                            continue

                        time.sleep(0.3)
                        price = float(api.get_latest_trade(sym)._raw['p'])
                        gain_pct = (price - bp) / bp * 100

                        if gain_pct < 0.1: # Breakeven or losing
                            print(f"💣 Trimming weak position: {sym} ({gain_pct:.2f}%)")
                            # Get accurate available qty from API first
                            try:
                                position = api.get_position(sym)
                                available_qty = float(position.qty_available)
                            except:
                                available_qty = qty  # fallback if position not fetched

                            # Ensure qty doesn't exceed available by mistake
                            safe_qty = min(qty, available_qty)
                            safe_qty = Decimal(str(safe_qty)).quantize(Decimal('0.000000001'), rounding=ROUND_DOWN)

                            try:
                                api.submit_order(
                                    symbol=sym,
                                    side='sell',
                                    type='market',
                                    qty=float(safe_qty),
                                    time_in_force='day'
                                )
                                # Only remove from portfolio and log if successful
                                port_df = port_df[port_df.Symbol != sym]
                                port_df.to_csv(PORTFOLIO, index=False)
                                # Only remove from portfolio and log if successful
                                bp = float(entry['Buy_Price']) if 'Buy_Price' in entry else 0.0
                                spread = round((price * (1 - SLIPPAGE_BPS) - bp) * qty, 2)
                                pnl = round(spread * qty, 2)  # total PnL for trade
                                with open(TRADES_LOG, 'a') as f:
                                    f.write(f"SELL,{sym},{now},{price},{qty},ForceSell,{spread}\n")

                                port_df = port_df[port_df.Symbol != sym]
                                port_df.to_csv(PORTFOLIO, index=False)

                            except Exception as e:
                                print(f"❌ Force-sell failed for {sym}: {e}")
                    except Exception as e:
                        print(f"❗ Error during force-sell loop: {e}")

        except Exception as e:
            print(f"❗ Strategy Boost Error: {e}")

        ## section 4.4
        # === BUY CANDIDATE FILTER ===
        # Condition columns must match the order used when saving Combos -> use default_weights keys for consistency
        condition_cols = list(default_weights.keys())
        nconds = len(condition_cols)

        # Start with universe excluding holdings
        candidates = ind_df[~ind_df['Symbol'].isin(port_df['Symbol'])].copy()

        # If empty short-circuit
        if candidates.empty:
            top_buy_candidates = pd.DataFrame(columns=ind_df.columns)
        else:
            # Dynamic quantile threshold (use looser cutoff if strategy_boost)
            quantile_level = 0.85 if not strategy_boost else 0.75
            dynamic_thresh = max(0.60, candidates['Total_Score'].quantile(quantile_level))  # floor at 0.60
            # Debug print so you can see the threshold each loop
            print(f"🔎 Dynamic threshold (quantile {quantile_level}): {dynamic_thresh:.3f} (floor 0.60)")

            # Filter by dynamic threshold
            candidates = candidates[candidates['Total_Score'] >= dynamic_thresh].copy()

            # --- Safe Pattern Matching & Boosting ---
            try:
                if os.path.exists("weekly_success_patterns.csv"):
                    pattern_df = pd.read_csv("weekly_success_patterns.csv")
                    # safe parse Combo -> tuple
                    if 'Combo' in pattern_df.columns:
                        import ast


                        def _safe_parse(x):
                            try:
                                if isinstance(x, str):
                                    # prefer ast.literal_eval for safety
                                    parsed = ast.literal_eval(x)
                                    return tuple(parsed) if isinstance(parsed, (list, tuple)) else tuple()
                                elif isinstance(x, (list, tuple)):
                                    return tuple(x)
                                else:
                                    return tuple()
                            except Exception:
                                return tuple()


                        pattern_df['Combo'] = pattern_df['Combo'].apply(_safe_parse)
                    else:
                        pattern_df['Combo'] = [tuple()] * len(pattern_df)

                    pattern_combos = [pc for pc in pattern_df['Combo'].tolist()
                                      if isinstance(pc, (list, tuple)) and len(pc) == nconds]

                    if pattern_combos and not candidates.empty:
                        match_scores = []
                        for _, crow in candidates.iterrows():
                            try:
                                current_combo = tuple(int(crow[col]) for col in condition_cols)
                            except Exception:
                                match_scores.append(0.0)
                                continue
                            best_match = 0.0
                            for pcombo in pattern_combos:
                                # compare only when lengths match
                                matches = sum(1 for i in range(nconds) if current_combo[i] == pcombo[i])
                                score = matches / nconds
                                if score > best_match:
                                    best_match = score
                            match_scores.append(float(best_match))
                        candidates['Pattern_Match'] = match_scores
                        # linear boost: up to +10% to Total_Score for perfect match
                        candidates['Total_Score'] = candidates['Total_Score'] * (
                                    1.0 + candidates['Pattern_Match'] * 0.1)
                    else:
                        candidates['Pattern_Match'] = 0.0
            except Exception as e:
                print(f"⚠️ Pattern matching failed: {e}")
                if 'Pattern_Match' not in candidates.columns:
                    candidates['Pattern_Match'] = 0.0

            # Final sorting and limit to MAX_CANDIDATES
            candidates = candidates.sort_values(by='Total_Score', ascending=False).head(MAX_CANDIDATES)
            top_buy_candidates = candidates.sort_values(by=['Total_Score', 'Price'],
                                                        ascending=[False, True]).reset_index(drop=True)

        # Print top picks
        print("\n🔍 Top Buy Candidates:")
        if not top_buy_candidates.empty:
            print(top_buy_candidates[['Symbol', 'Price', 'Total_Score']])
        else:
            print("⚠️ No buy candidates found after filtering")
            print(f" - Portfolio symbols: {port_df['Symbol'].tolist()}")
            print(f" - Strategy boost: {strategy_boost}")
            print(f" - Available budget: ${avail:.2f}")

        # ✅ Check if we need to bank profits
        trigger_banking_all(top_buy_candidates)

        # 🔄 Reload portfolio after banking so we don’t try to sell same stock again
        port_df = pd.read_csv(PORTFOLIO, parse_dates=['Buy_Date']) if os.path.exists(PORTFOLIO) else pd.DataFrame(
            columns=['Symbol', 'Buy_Date', 'Buy_Price', 'Qty'])
        trigger_banking_all(top_buy_candidates)

        # Reload portfolio after banking
        port_df = pd.read_csv(PORTFOLIO, parse_dates=['Buy_Date']) if os.path.exists(PORTFOLIO) else pd.DataFrame(
            columns=['Symbol', 'Buy_Date', 'Buy_Price', 'Qty'])

        # ===========================
        # === SECTION 4.5: SELL LOOP (Held Stocks) — replaced with partial TP + trailing stop logic
        # ===========================
        for _, entry in port_df.iterrows():
            sym = entry['Symbol']
            try:
                price = float(api.get_latest_trade(sym)._raw['p'])
            except Exception as e:
                print(f"⚠️ Could not fetch price for {sym}: {e}")
                continue

            if sym not in ind_df['Symbol'].values:
                print(f"⚠️ {sym} missing from indicators, skipping.")
                continue

            row = ind_df[ind_df['Symbol'] == sym].iloc[0]

            bp = float(entry.get('Buy_Price', 0.0))
            q_owned = float(entry.get('Qty', 0.0))
            remaining = float(entry.get('Remaining_Qty', q_owned)) if 'Remaining_Qty' in entry else q_owned
            trailing_stop = float(entry.get('Trailing_Stop', bp * (1 - 0.03)))
            take_target1 = float(entry.get('Take_Target1', bp * (1 + 0.07)))
            take_ratio1 = float(entry.get('Take_Ratio1', 0.5))

            dh = days_held(entry['Buy_Date'])
            try:
                dynamic_days = get_dynamic_hold_days(
                    rsi=row.get('RSI_14', 50),
                    atr=row.get('ATR_14', 0.0),
                    sma_short=row.get('SMA_3', 0.0),
                    sma_long=row.get('SMA_20', 0.0)
                )
            except:
                dynamic_days = MIN_HOLD_DAYS

            gain_pct = (price - bp) / bp * 100

            print(f"\n🔍 Checking SELL for {sym} | Price ${price:.2f} | BP ${bp:.2f} | Remaining {remaining}")

            # --- 1) Partial Take-Profit (Tier 1)
            if remaining > 0 and price >= take_target1:
                sell_qty = round(remaining * take_ratio1, 6)
                try:
                    api.submit_order(symbol=sym, side='sell', type='market', qty=float(sell_qty), time_in_force='day')
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # slippage-aware realized spread per share
                    SLIPPAGE_BPS = globals().get('SLIPPAGE_BPS', 0.001)
                    spread = round((price * (1 - SLIPPAGE_BPS) - bp) * sell_qty, 2)
                    with open(TRADES_LOG, 'a', encoding='utf-8') as f:
                        f.write(f"SELL,{sym},{now_str},{price},{sell_qty},TakeProfit1,{spread}\n")
                    print(f"✅ Partial TP executed: {sym} sold {sell_qty} @ {price:.2f} (Take1)")

                    # update remaining qty and trailing stop
                    remaining = max(0.0, remaining - sell_qty)
                    new_trailing = price * (1 - 0.03)  # set trailing at 3% below current price (tunable)
                    port_df.loc[port_df['Symbol'] == sym, 'Remaining_Qty'] = remaining
                    port_df.loc[port_df['Symbol'] == sym, 'Trailing_Stop'] = round(new_trailing, 4)

                    if remaining == 0:
                        port_df = port_df[port_df['Symbol'] != sym]
                    port_df.to_csv(PORTFOLIO, index=False)

                except Exception as e:
                    print(f"❌ Partial TP failed for {sym}: {e}")

                # after partial TP move on
                continue

            # --- 2) Trailing stop: sell remaining if price <= trailing_stop
            try:
                if remaining > 0 and price <= trailing_stop:
                    sell_qty = remaining
                    api.submit_order(symbol=sym, side='sell', type='market', qty=float(sell_qty), time_in_force='day')
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    SLIPPAGE_BPS = globals().get('SLIPPAGE_BPS', 0.001)
                    spread = round((price * (1 - SLIPPAGE_BPS) - bp) * sell_qty, 2)
                    with open(TRADES_LOG, 'a', encoding='utf-8') as f:
                        f.write(f"SELL,{sym},{now_str},{price},{sell_qty},TrailingStop,{spread}\n")
                    print(f"💥 Trailing stop fired for {sym}, sold remaining {sell_qty} @ {price:.2f}")
                    port_df = port_df[port_df['Symbol'] != sym]
                    port_df.to_csv(PORTFOLIO, index=False)
                    continue
            except Exception as e:
                print(f"❌ Trailing stop sell failed for {sym}: {e}")

            # --- 3) Fallback / original hybrid logic (early exit, hybrid target, SMA break, stop loss)
            decision, reason, spread = 'HOLD', '', 0.0
            try:
                if dh < dynamic_days:
                    if gain_pct >= EARLY_EXIT_PCT:
                        decision, reason, spread = 'SELL', "Early +7%", round(price - bp, 2)
                    elif gain_pct <= -EARLY_EXIT_PCT:
                        decision, reason, spread = 'SELL', "Early -7%", round(price - bp, 2)
                else:
                    hybrid_target = max(0.35, bp * 0.025)
                    if price >= (bp + hybrid_target):
                        decision, reason, spread = 'SELL', f"Hybrid Target Hit (+{round(hybrid_target, 2)})", round(
                            price - bp, 2)
                    elif gain_pct <= -STOP_LOSS:
                        decision, reason, spread = 'SELL', "Stop -4%", round(price - bp, 2)
                    elif row.get('C_SMA3_gt_SMA20', 1) == 0:
                        decision, reason, spread = 'SELL', "SMA3<20", round(price - bp, 2)

                if decision == 'SELL':
                    try:
                        alpaca_position = api.get_position(sym)
                        alpaca_qty = float(alpaca_position.qty)
                        safe_qty = min(remaining, alpaca_qty)
                        qty = float(Decimal(str(safe_qty)).quantize(Decimal('0.000000001'), rounding=ROUND_DOWN))

                        api.submit_order(symbol=sym, side='sell', type='market', qty=qty, time_in_force='day')
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        SLIPPAGE_BPS = globals().get('SLIPPAGE_BPS', 0.001)
                        spread = round((price * (1 - SLIPPAGE_BPS) - bp) * qty, 2)
                        with open(TRADES_LOG, 'a', encoding='utf-8') as f:
                            f.write(f"SELL,{sym},{now_str},{price},{qty},{reason},{spread}\n")

                        print(f"💸 Selling {sym} → Reason: {reason} | Qty: {qty}")

                        port_df = port_df[port_df['Symbol'] != sym]
                        port_df.to_csv(PORTFOLIO, index=False)

                    except Exception as e:
                        print(f"❌ Sell error for {sym}: {e}")

            except Exception as e:
                print(f"❗ Sell loop error for {sym}: {e}")

        ## === MICRO POSITION TRIMMER + BANKING ===
        try:
            positions = api.list_positions()
            port_df = pd.read_csv(PORTFOLIO)  # ✅ Sync latest portfolio

            for pos in positions:
                qty = float(pos.qty)
                value = float(pos.market_value)
                sym = pos.symbol

                if value <= 0.01 or qty <= 0.005:
                    qty = float(Decimal(str(min(pos.qty, pos.qty_available)))
                                .quantize(Decimal('0.000000001'), rounding=ROUND_DOWN))

                    try:
                        print(f"💣 Trimming micro position: {sym} (${value:.4f})")
                        api.submit_order(
                            symbol=sym,
                            side='sell',
                            type='market',
                            qty=qty,
                            time_in_force='day'
                        )

                        # ✅ Safely get Buy Price from portfolio
                        bp = float(port_df[port_df['Symbol'] == sym]['Buy_Price'].values[0]) if sym in port_df[
                            'Symbol'].values else 0.0
                        price = float(pos.current_price)

                        # Correct spread = total PnL
                        spread = round((price * (1 - SLIPPAGE_BPS) - bp) * qty, 2)
                        pnl = round(spread * qty, 2)

                        # Log the trim
                        with open(TRADES_LOG, 'a') as f:
                            f.write(
                                f"TRIM,{sym},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{price},{qty},Auto-Trim,{spread}\n")

                        # Weekly profit log (total profit, not per share)
                        date_str = datetime.today().strftime('%Y-%m-%d')
                        with open("weekly_profit_log.csv", "a") as f:
                            f.write(f"{date_str},{sym},{spread}\n")

                        # Update banked cash only if profit is positive
                        if spread > 0:
                            if not os.path.exists("banked_cash.txt"):
                                with open("banked_cash.txt", "w") as b:
                                    b.write("0.0")
                            with open("banked_cash.txt", "r") as b:
                                current_banked = float(b.read().strip())
                            current_banked += spread
                            with open("banked_cash.txt", "w") as b:
                                b.write(str(round(current_banked, 2)))
                            print(f"💰 Banked ${spread:.2f} from {sym}. Total banked: ${current_banked:.2f}")
                        else:
                            print(f"📉 Logged loss of ${spread:.2f} for {sym}")

                        # Remove from portfolio
                        port_df = port_df[port_df['Symbol'] != sym]
                        port_df.to_csv(PORTFOLIO, index=False)

                    except Exception as e:
                        print(f"⚠️ Could not trim {sym}: {e}")

        except Exception as e:
            print(f"❌ Could not fetch positions for trimming: {e}")


        # === BOT-CENTRIC Daily Profit Banking (robust) ===
        try:
            # load banked cash safely
            banked_file = "banked_cash.txt"
            if os.path.exists(banked_file):
                with open(banked_file, "r") as f:
                    try:
                        banked_cash = float(f.read().strip())
                    except:
                        banked_cash = 0.0
            else:
                banked_cash = 0.0

            # get current positions from Alpaca (map symbol -> position) for fallback/live prices
            try:
                positions = api.list_positions()
                positions_map = {p.symbol: p for p in positions}
            except Exception:
                positions_map = {}

            # compute current market value and used (money currently invested)
            current_market_value = 0.0
            used_calc = 0.0

            if not port_df.empty:
                # primary source: PORTFOLIO file (trusted)
                for _, r in port_df.iterrows():
                    sym = r['Symbol']
                    try:
                        qty = float(r['Qty'])
                    except:
                        qty = 0.0
                    try:
                        bp = float(r['Buy_Price'])
                    except:
                        bp = 0.0

                    # prefer live price from Alpaca position if available, else query last trade
                    if sym in positions_map:
                        live_price = float(positions_map[sym].current_price)
                    else:
                        try:
                            live_price = float(api.get_latest_trade(sym)._raw['p'])
                        except:
                            live_price = 0.0

                    current_market_value += live_price * qty
                    used_calc += bp * qty

            else:
                # fallback: rebuild market value from Alpaca positions (if PORTFOLIO got out-of-sync)
                for sym, p in positions_map.items():
                    try:
                        qty = float(p.qty)
                        live_price = float(p.current_price)
                        # use avg_entry_price if available to compute used/cost basis
                        try:
                            entry = float(p.avg_entry_price)
                        except:
                            entry = live_price
                        current_market_value += live_price * qty
                        used_calc += entry * qty
                    except Exception:
                        continue

            used = float(used_calc)
            avail = max(0.0, MAX_BUDGET - used)

            # Bot equity defined as realized banked cash + unrealized market value
            bot_equity = banked_cash + current_market_value
            profit_today = bot_equity - MAX_BUDGET

            print(
                f"💹 Bot Equity: ${bot_equity:.2f} | MarketValue: ${current_market_value:.2f} | Banked: ${banked_cash:.2f} | Profit: ${profit_today:.2f} | Used: ${used:.2f} | Avail: ${avail:.2f}")

            # Update peak and check drawdown
            try:
                update_equity_peak(bot_equity)
                defensive_mode = check_drawdown_mode(bot_equity)
            except:
                defensive_mode = False

            if defensive_mode:
                print(f"⚠️ Defensive mode ON — drawdown exceeded {DRAW_THRESHOLD * 100:.1f}%")
                # reduce Kelly aggressiveness OR pause buys
                KELLY_ALPHA = 0.10  # override when defensive
            else:
                KELLY_ALPHA = 0.25  # normal

            # run banking only if profit target met
            if profit_today >= DAILY_PROFIT_TARGET:
                print(
                    f"🎯 Bot daily profit target hit (${profit_today:.2f} ≥ ${DAILY_PROFIT_TARGET:.2f}) → Banking gains")

                # Build sell candidates from port_df (prefer profitable ones)
                # if port_df empty fall back to positions_map
                candidates = []
                if not port_df.empty:
                    for _, r in port_df.iterrows():
                        sym = r['Symbol']
                        qty = float(r['Qty'])
                        bp = float(r['Buy_Price'])
                        live_price = float(positions_map[sym].current_price) if sym in positions_map else (
                            float(api.get_latest_trade(sym)._raw['p']) if positions_map or True else 0.0)
                        unrealized = (live_price - bp) * qty
                        candidates.append((sym, qty, live_price, bp, unrealized))
                else:
                    for sym, p in positions_map.items():
                        qty = float(p.qty)
                        live_price = float(p.current_price)
                        bp = float(p.avg_entry_price) if getattr(p, 'avg_entry_price', None) else live_price
                        unrealized = (live_price - bp) * qty
                        candidates.append((sym, qty, live_price, bp, unrealized))

                # sort profitable candidates by unrealized profit (desc)
                profitable = sorted([c for c in candidates if c[4] > 0], key=lambda x: x[4], reverse=True)

                banked_now = 0.0
                for sym, qty, live_price, bp, unrealized in profitable:
                    if banked_now >= DAILY_PROFIT_TARGET:
                        break
                    try:
                        api.submit_order(symbol=sym, side='sell', type='market', qty=float(qty), time_in_force='day')
                    except Exception as e:
                        print(f"❌ Failed to submit bank sell for {sym}: {e}")
                        continue

                    realized = round((live_price - bp) * qty, 2)
                    banked_now += max(0.0, realized)

                    # update port_df and used/avail
                    if sym in port_df['Symbol'].values:
                        port_df = port_df[port_df['Symbol'] != sym]
                        port_df.to_csv(PORTFOLIO, index=False)
                    recalc_used_avail()

                    # update banked file
                    banked_cash += realized
                    with open(banked_file, "w") as f:
                        f.write(str(round(banked_cash, 2)))

                    # log
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(TRADES_LOG, 'a') as f:
                        f.write(f"SELL,{sym},{now_str},{live_price},{qty},DailyBanking,{realized}\n")
                    print(
                        f"✅ Banked ${realized:.2f} from {sym}. Banked this run: ${banked_now:.2f} (Total banked: ${banked_cash:.2f})")

                if banked_now <= 0:
                    print("⚠️ No profitable positions available to bank right now.")
        except Exception as e:
            print(f"❗ BOT Daily Banking Check Failed: {e}")

            # === Weekly Profit Banking on Friday Late Night ===
        now = datetime.now()
        if now.strftime('%A') == 'Saturday' and now.time() >= datetime.strptime("00:30", "%H:%M").time():
                print("🏦 Weekly profit banking window triggered")
                try:
                    for _, entry in port_df.iterrows():
                        sym = entry['Symbol']
                        try:
                            # Get all current positions only once
                            all_positions = {p.symbol: p for p in api.list_positions()}

                            if sym not in all_positions:
                                print(f"⚠️ {sym} not found in current positions, skipping banking.")
                                continue

                            price = float(api.get_latest_trade(sym)._raw['p'])
                            bp = float(entry['Buy_Price'])
                            live_qty = float(all_positions[sym].qty)
                            qty = min(entry['Qty'], live_qty)
                            qty = round(qty, 9)

                            gain = price - bp
                            if gain <= 0:
                                continue  # Skip non-profitable positions

                            sale_value = qty * price
                            principal = qty * bp
                            net_gain = sale_value - principal

                            # Sell the position
                            api.submit_order(
                                symbol=sym,
                                side='sell',
                                type='market',
                                qty=qty,
                                time_in_force='day'
                            )

                            print(
                                f"✅ Sold {sym} at ${price:.2f} | Principal: ${principal:.2f} | Profit: ${net_gain:.2f}")

                            # Update portfolio and bank
                            used -= principal
                            avail = max(0.0, MAX_BUDGET - used)
                            port_df = port_df[port_df['Symbol'] != sym]
                            port_df.to_csv(PORTFOLIO, index=False)

                            # Load current banked
                            if not os.path.exists("banked_cash.txt"):
                                with open("banked_cash.txt", "w") as f:
                                    f.write("0.0")

                            with open("banked_cash.txt", "r") as f:
                                current_banked = float(f.read().strip())

                            current_banked += net_gain
                            with open("banked_cash.txt", "w") as f:
                                f.write(str(round(current_banked, 2)))

                            print(f"💰 Banked ${net_gain:.2f}. Total banked: ${current_banked:.2f}")

                            # === NEW: Update banking_status.txt ===
                            bot_equity = current_banked + float(sum(
                                float(p.market_value) for p in api.list_positions()
                            ))
                            with open("banking_status.txt", "w") as f:
                                f.write(f"{now.strftime('%Y-%m-%d')},True,{bot_equity:.2f}")

                        except Exception as e:
                            print(f"⚠️ Could not bank {sym}: {e}")

                except Exception as e:
                    print(f"❗ Weekly realized banking failed: {e}")

                ## === SECTION 4.6: BUY LOOP ===
        if not top_buy_candidates.empty:
            dynamic_cap = min(MAX_PER_TRADE, avail * 0.15)
            per_stock_limit = min(MAX_PER_TRADE, avail * 0.20)
            MIN_BUY_AMOUNT = 25

            # Risk / sizing / Kelly / vol-scaling params (tune in backtest)
            BASE_KELLY_ALPHA = 0.25  # normal fractional Kelly multiplier
            DEFENSIVE_KELLY_ALPHA = 0.10  # reduced when drawdown guard fires
            RISK_PER_TRADE_PCT = 0.005  # risk per trade as fraction of equity (0.5%)
            STOP_ATR_MULT = 1.5  # stop = STOP_ATR_MULT * ATR_14 / price
            TARGET_WEEKLY_VOL = 0.06  # target weekly vol (6% example)
            SLIPPAGE_BPS = 0.001  # 0.1% slippage assumption for cost checks
            DRAW_THRESHOLD = 0.12  # 12% drawdown to enter defensive mode
            PEAK_FILE = "equity_peak.txt"


            # Helper: read/update peak equity for drawdown guard
            def read_equity_peak():
                try:
                    if os.path.exists(PEAK_FILE):
                        with open(PEAK_FILE, "r") as f:
                            return float(f.read().strip() or 0.0)
                except:
                    pass
                return 0.0


            def write_equity_peak(val):
                try:
                    with open(PEAK_FILE, "w") as f:
                        f.write(str(round(val, 2)))
                except:
                    pass


            # compute conservative "equity": banked + market value + avail
            try:
                with open("banked_cash.txt", "r") as b:
                    banked = float(b.read().strip() or 0.0)
            except:
                banked = 0.0

            current_market_value = 0.0
            try:
                if os.path.exists(PORTFOLIO):
                    tmp_pf = pd.read_csv(PORTFOLIO)
                    for _, r in tmp_pf.iterrows():
                        s = r['Symbol']
                        q = float(r.get('Qty', 0.0))
                        try:
                            p = float(ind_df.loc[ind_df['Symbol'] == s, 'Price'].iloc[0])
                        except:
                            try:
                                p = float(api.get_latest_trade(s)._raw['p'])
                            except:
                                p = 0.0
                        current_market_value += q * p
            except Exception as e:
                print(f"⚠️ Could not compute current_market_value: {e}")

            equity = banked + current_market_value + avail

            # drawdown guard: update peak and set defensive flag
            try:
                peak = read_equity_peak() or equity
                if equity > peak:
                    write_equity_peak(equity)
                    peak = equity
                draw = (peak - equity) / peak if peak > 0 else 0.0
                defensive_mode = draw >= DRAW_THRESHOLD
                if defensive_mode:
                    print(f"⚠️ Defensive mode ON. Drawdown = {draw * 100:.2f}% (threshold {DRAW_THRESHOLD * 100:.0f}%)")
                    KELLY_ALPHA = DEFENSIVE_KELLY_ALPHA
                else:
                    KELLY_ALPHA = BASE_KELLY_ALPHA
            except Exception as e:
                print(f"⚠️ Drawdown guard error: {e}")
                KELLY_ALPHA = BASE_KELLY_ALPHA

            # baseline score-based allocation
            total_score = top_buy_candidates['Total_Score'].sum()
            if total_score == 0:
                print("⚠️ Total score is 0. Skipping allocation.")
            else:
                top_buy_candidates['Budget_Weight'] = top_buy_candidates['Total_Score'] / total_score
                top_buy_candidates['Alloc'] = top_buy_candidates['Budget_Weight'] * avail
                top_buy_candidates['Alloc'] = top_buy_candidates['Alloc'].apply(
                    lambda x: max(MIN_BUY_AMOUNT, min(x, per_stock_limit))
                )
                # ---------- PREP: renormalize MVO weights across the current buy candidate subset ----------
                # Use the mvo_map loaded earlier (mvo_map = {symbol: weight})
                # If mvo_map missing some symbols, fallback to uniform weights across the subset.

                try:
                    cand_symbols = top_buy_candidates['Symbol'].astype(str).tolist()
                except Exception:
                    cand_symbols = []

                # Use available cash for the cap calculation (use 'avail' or choose another metric)
                equity_for_caps = float(globals().get('avail', MAX_BUDGET))  # prefer 'avail' (cash left)

                if cand_symbols:
                    # Collect raw MVO weights for this subset
                    mvo_raw = np.array([mvo_map.get(s, 0.0) for s in cand_symbols], dtype=float)

                    # If all zeros (MVO didn't include these symbols), fallback to uniform
                    if mvo_raw.sum() <= 1e-9:
                        mvo_subset_ws = np.ones(len(mvo_raw)) / len(mvo_raw)
                        mvo_note = "uniform/fallback"
                    else:
                        # normalize weights across subset
                        mvo_subset_ws = mvo_raw / mvo_raw.sum()
                        mvo_note = "normalized-from-mvo_map"

                    # map symbol -> renormalized weight
                    mvo_subset_map = dict(zip(cand_symbols, mvo_subset_ws))

                    # diagnostic print (optional)
                    print(
                        f"[MVO-RENORM] mode={mvo_note} | sum={(mvo_subset_ws.sum()):.6f} | equity_for_caps=${equity_for_caps:.2f}")
                else:
                    mvo_subset_map = {}
                # --------------------------------------------------------------------

                # iterate candidates and apply Kelly cap, vol scaling, ATR risk sizing, slippage buffer
                for _, row in top_buy_candidates.iterrows():
                    sym = row['Symbol']
                    print(f"🔎 Evaluating Buy: {sym}")

                    try:
                        time.sleep(0.25)
                        price = float(api.get_latest_trade(sym)._raw['p'])
                    except Exception as e:
                        print(f"⚠️ Skipping {sym}: {e}")
                        continue

                    if not is_tradable_and_fractionable(sym, api):
                        print(f"🚫 Skipping {sym}: Not tradable or not fractionable.")
                        continue

                    # read mu/sigma from ind_df (daily) — fallbacks to zero if missing
                    try:
                        mu_daily = float(ind_df.loc[ind_df['Symbol'] == sym, 'Mu_21d'].iloc[0])
                    except Exception:
                        mu_daily = 0.0
                    try:
                        sigma_daily = float(ind_df.loc[ind_df['Symbol'] == sym, 'Sigma_21d'].iloc[0])
                    except Exception:
                        sigma_daily = 0.0

                    # convert to weekly units (5 trading days)
                    mu_week = mu_daily * 5.0
                    sigma_week = sigma_daily * (5.0 ** 0.5) if sigma_daily > 0 else 0.0

                    # continuous Kelly (weekly) — conservative fractional Kelly
                    f_star = 0.0
                    if sigma_week > 0:
                        f_star = (mu_week) / (sigma_week ** 2)  # risk-free omitted for conservative estimate
                    kelly_f = max(0.0, KELLY_ALPHA * f_star)

                    # baseline allocation from score
                    alloc_score = float(row['Alloc'])

                    # Kelly dollar cap (only if positive); otherwise use score alloc
                    alloc_kelly = kelly_f * equity if kelly_f > 0 else alloc_score
                    alloc = min(alloc_score, alloc_kelly)

                    # volatility-scaling: scale allocation so that expected vol ~ target
                    realized_week_vol = sigma_week if sigma_week > 0 else 0.0001
                    vol_scale = TARGET_WEEKLY_VOL / realized_week_vol
                    vol_scale = max(0.5, min(vol_scale, 2.0))
                    alloc *= vol_scale

                    # ---------- SAFE MVO-CAP (use renormalized subset weights / skip tiny caps) ----------
                    MAX_PORTFOLIO_EXPOSURE = globals().get('MAX_PORTFOLIO_EXPOSURE', 0.60)

                    # Prefer renormalized subset map if available; fall back sensibly
                    mvo_w = None
                    if 'mvo_subset_map' in locals() and isinstance(mvo_subset_map, dict):
                        mvo_w = mvo_subset_map.get(sym, None)

                    if mvo_w is None:
                        # try global map (mvo_map) if present
                        if isinstance(mvo_map, dict):
                            mvo_w = mvo_map.get(sym, 0.0)
                        else:
                            mvo_w = 0.0

                    # Ensure we have a positive fallback (uniform) if all else failed
                    if mvo_w <= 0:
                        # uniform fallback across current candidates avoids micro-caps
                        try:
                            mvo_w = 1.0 / max(1, len(top_buy_candidates))
                        except Exception:
                            mvo_w = 1.0 / max(1, len(cand_symbols) if 'cand_symbols' in locals() else 1)

                    # Use equity_for_caps (computed earlier) for dollar cap calculations
                    mvo_cap = mvo_w * MAX_PORTFOLIO_EXPOSURE * float(
                        globals().get('equity_for_caps', equity_for_caps if 'equity_for_caps' in locals() else equity))

                    alloc_before_mvo = alloc

                    # If the computed MVO cap would be smaller than MIN_BUY_AMOUNT, skip enforcing the cap
                    if mvo_cap >= MIN_BUY_AMOUNT:
                        alloc = min(alloc, mvo_cap)
                        print(
                            f"[MVO-CAP] {sym} | mvo_w={mvo_w:.6f} | mvo_cap=${mvo_cap:.2f} | alloc_before=${alloc_before_mvo:.2f} -> final_alloc=${alloc:.2f}")
                    else:
                        # Skip cap to avoid micro-caps that kill trades
                        print(
                            f"[MVO-CAP-SKIP] {sym} | mvo_w={mvo_w:.6f} | mvo_cap=${mvo_cap:.2f} < MIN_BUY_AMOUNT ${MIN_BUY_AMOUNT:.2f}; using alloc=${alloc_before_mvo:.2f}")
                        alloc = alloc_before_mvo
                    # -------------------------------------------------------------------------------

                    # compute qty allowed by allocation
                    qty_by_alloc = Decimal(str(float(alloc) / price)).quantize(Decimal('0.000001'), rounding=ROUND_DOWN)

                    # ATR-based stop percentage (fallback to 5% if ATR missing)
                    try:
                        atr = float(ind_df.loc[ind_df['Symbol'] == sym, 'ATR_14'].iloc[0])
                        stop_pct = max(0.01, (STOP_ATR_MULT * atr) / price)
                    except Exception:
                        stop_pct = 0.05

                    # risk-based shares to keep risk per trade bounded
                    risk_dollar = RISK_PER_TRADE_PCT * equity
                    shares_by_risk = (risk_dollar / (stop_pct * price)) if stop_pct > 0 else float(qty_by_alloc)

                    # final qty: min of allocation, risk sizing, and max share cap
                    try:
                        final_qty = float(
                            min(qty_by_alloc, Decimal(str(shares_by_risk)), Decimal(str(MAX_SHARES_PER_STOCK))))
                    except Exception:
                        final_qty = float(min(qty_by_alloc, Decimal(str(MAX_SHARES_PER_STOCK))))

                    if final_qty < 0.01 or final_qty <= 0:
                        print(f"❌ Skipping {sym}: final_qty too small ({final_qty}) after sizing checks.")
                        continue

                    # effective cost including slippage buffer (buys are a little more expensive)
                    effective_cost = (final_qty * price) * (1.0 + SLIPPAGE_BPS)

                    try:
                        actual_cash = float(api.get_account().cash)
                    except Exception as e:
                        print(f"❌ Could not fetch Alpaca cash: {e}")
                        actual_cash = 0.0

                    # check minimum & budget
                    if round(effective_cost, 2) < MIN_BUY_AMOUNT:
                        print(
                            f"⚠️ Skipping {sym}: Effective cost ${effective_cost:.2f} below minimum ${MIN_BUY_AMOUNT:.2f}")
                        continue

                    if effective_cost > avail or effective_cost > actual_cash:
                        print(
                            f"🚫 Skipping {sym}: Not enough budget. Needed: ${effective_cost:.2f}, Avail (bot): ${avail:.2f}, Alpaca: ${actual_cash:.2f}")
                        continue

                    # attempt to restore banked cash if Alpaca account is low (existing logic)
                    try:
                        if actual_cash < 500:
                            if os.path.exists("banked_cash.txt"):
                                with open("banked_cash.txt", "r") as f:
                                    banked = float(f.read().strip())
                            else:
                                banked = 0.0

                            transfer = min(500 - actual_cash, banked)
                            if transfer > 0:
                                actual_cash += transfer
                                banked -= transfer
                                with open("banked_cash.txt", "w") as f:
                                    f.write(str(round(banked, 2)))
                                print(
                                    f"🔁 Restored ${transfer:.2f} from banked cash. New available cash: ${actual_cash:.2f}")
                    except Exception as e:
                        print(f"⚠️ Error accessing banked funds: {e}")

                    # submit buy (whole shares if >=1, else fractional notional)
                    try:
                        if final_qty >= 1:
                            qty_int = int(final_qty)
                            api.submit_order(
                                symbol=sym,
                                side='buy',
                                type='market',
                                qty=qty_int,
                                time_in_force='gtc'
                            )
                            used_cost = qty_int * price
                            reason = f"Top10 Score={row['Total_Score']} (whole)"
                            qty_submitted = float(qty_int)
                        else:
                            notional = round(final_qty * price, 2)
                            api.submit_order(
                                symbol=sym,
                                side='buy',
                                type='market',
                                notional=notional,
                                time_in_force='day'
                            )
                            used_cost = final_qty * price
                            reason = f"Top10 Score={row['Total_Score']} (fractional)"
                            qty_submitted = float(final_qty)

                        # Update portfolio with trailing & partial-take metadata for SELL logic
                        new = {
                            'Symbol': sym,
                            'Buy_Date': datetime.today().strftime('%Y-%m-%d'),
                            'Buy_Price': price,
                            'Qty': float(qty_submitted),
                            'Trailing_Stop': round(price * (1 - 0.03), 4),  # initial 3% trailing
                            'Take_Target1': round(price * (1 + 0.07), 4),  # 7% first take
                            'Take_Ratio1': 0.5,
                            'Remaining_Qty': float(qty_submitted)
                        }
                        port_df = pd.concat([port_df, pd.DataFrame([new])], ignore_index=True)
                        port_df.to_csv(PORTFOLIO, index=False)

                        # log with mu/sigma/kelly details
                        with open(TRADES_LOG, 'a', encoding='utf-8') as f:
                            f.write(
                                f"BUY,{sym},{datetime.now().strftime('%Y-%m-%d %H:%M')},{price},{qty_submitted},{reason},{''},{mu_daily:.6f},{sigma_daily:.6f},{f_star:.6f},{alloc:.2f},{kelly_f:.6f}\n")

                        print(
                            f"✅ BUY submitted {sym} qty={qty_submitted} cost=${used_cost:.2f} | alloc=${alloc:.2f} | kelly_f={kelly_f:.6f}")

                        # update bookkeeping
                        used += used_cost
                        avail = max(0.0, MAX_BUDGET - used)
                        if avail < 1.0:
                            break

                    except Exception as e:
                        print(f"❌ Buy error {sym}: {e}")
                        continue

                        # === BUY loop above this...

        print(f"\U0001F6CC Sleeping for {SLEEP_HOURS} hour(s)...")
        time.sleep(SLEEP_HOURS * 3600)

    except KeyboardInterrupt:
        print("\U0001F6D1 Interrupted by user. Exiting loop.")
        break

    except Exception as e:
        print(f"❗ Error in loop: {e}")
        time.sleep(60)



