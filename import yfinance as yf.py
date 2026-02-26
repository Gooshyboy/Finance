import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors


tickers = ['NVDA', 'GOOGL', 'LLY', 'XOM', 'WMT', 'CAT']
bench_ticker = 'SPY'
initial_balance = 100000.0
LEVERAGE = 1 
BORROW_RATE = 0.05  

SMA_S, SMA_L = 50, 200
STOP_LOSS = 0.07 

print(f"Backtesting {len(tickers)} stocks with {LEVERAGE}x Leverage...")
combined_data = pd.DataFrame()
for t in tickers + [bench_ticker]:
    df = yf.download(t, period="10y", progress=False, multi_level_index=False)
    if not df.empty:
        combined_data[t] = df['Close']

prices = combined_data[tickers].dropna()
spy_prices = combined_data[bench_ticker].reindex(prices.index)

sma_s = prices.rolling(window=SMA_S).mean()
sma_l = prices.rolling(window=SMA_L).mean()


cash = initial_balance
portfolio = {t: {'shares': 0, 'entry_price': 0} for t in tickers}
history = []


for date, row in prices.iterrows():
    current_val = cash
    for t in tickers:
        current_val += portfolio[t]['shares'] * row[t]
    
    borrow_cost = ((LEVERAGE - 1) * current_val * (BORROW_RATE / 252))
    cash -= borrow_cost

    for t in tickers:
        curr_p = row[t]
        s_ma, l_ma = sma_s.at[date, t], sma_l.at[date, t]
        if pd.isna(l_ma): continue
        
        # STOP LOSS
        if portfolio[t]['shares'] != 0:
            dist = (curr_p - portfolio[t]['entry_price']) / portfolio[t]['entry_price']
            if (portfolio[t]['shares'] > 0 and dist < -STOP_LOSS) or \
               (portfolio[t]['shares'] < 0 and dist > STOP_LOSS):
                cash += portfolio[t]['shares'] * curr_p
                portfolio[t] = {'shares': 0, 'entry_price': 0}
                continue

        # LEVERAGED ALLOCATION: 
        alloc_per_stock = (current_val * LEVERAGE) / len(tickers)
        
        # Golden Cross -> Long
        if s_ma > l_ma and portfolio[t]['shares'] <= 0:
            cash += portfolio[t]['shares'] * curr_p 
            portfolio[t]['shares'] = alloc_per_stock / curr_p
            portfolio[t]['entry_price'] = curr_p
            cash -= alloc_per_stock
            
        # Death Cross -> Short
        elif s_ma < l_ma and portfolio[t]['shares'] >= 0:
            cash += portfolio[t]['shares'] * curr_p 
            portfolio[t]['shares'] = -(alloc_per_stock / curr_p)
            portfolio[t]['entry_price'] = curr_p
            cash += alloc_per_stock 

    history.append(current_val)

# CHARTING
res_pct = (pd.Series(history, index=prices.index) / initial_balance - 1) * 100
spy_pct = (spy_prices / spy_prices.iloc[0] - 1) * 100

fig, ax = plt.subplots(figsize=(12,6))
l1, = ax.plot(res_pct, label=f'Long-Short ({LEVERAGE}x)', color='#d62728', lw=2)
l2, = ax.plot(spy_pct, label='S&P 500 (SPY)', color='green', ls='--', alpha=0.5)

ax.set_title(f"Leveraged Strategy Performance (2016-2026)", fontsize=14)
ax.set_ylabel("Total Return (%)")
ax.legend()
ax.grid(True, alpha=0.3)

ax.ticklabel_format(style='plain', axis='y')


cursor = mplcursors.cursor([l1, l2], hover=True)

@cursor.connect("add")
def on_add(sel):
    label = sel.artist.get_label()
 
    sel.annotation.set_text(f"{label}\nReturn: {sel.target[1]:.2f}%")
    
   
    sel.annotation.get_bbox_patch().set(fc="black", alpha=1.0, edgecolor="white")
    sel.annotation.set_color("#00FF00")  
    sel.annotation.set_fontsize(10)
    sel.annotation.set_fontweight("bold")

import numpy as np


def run_visible_monte_carlo(history, n_sims=1000):
   
    hist_series = pd.Series(history)
    daily_rets = hist_series.pct_change().dropna().values
    n_days = len(daily_rets)
    
    # Run simulations and store in a matrix
    sim_matrix = np.zeros((n_sims, n_days + 1))
    sim_matrix[:, 0] = initial_balance
    
    for i in range(n_sims):
        sim_rets = np.random.choice(daily_rets, size=n_days, replace=True)
        sim_matrix[i, 1:] = initial_balance * np.cumprod(1 + sim_rets)

    # Calculate Percentiles for the "Fan"
    percentiles = [5, 25, 50, 75, 95]
    perc_data = {p: np.percentile(sim_matrix, p, axis=0) for p in percentiles}
    
  
    plt.style.use('dark_background') 
    fig, ax = plt.subplots(figsize=(12, 7))
    
    
    ax.fill_between(range(n_days + 1), perc_data[5], perc_data[95], color='#3d3d3d', alpha=0.3, label='95% Range')
    ax.fill_between(range(n_days + 1), perc_data[25], perc_data[75], color='#00aaff', alpha=0.4, label='Middle 50%')
    
    l1, = ax.plot(perc_data[50], color='#00ffcc', lw=2, label='Median Projection', ls='--')
    l2, = ax.plot(history, color='#ff0055', lw=3, label='ACTUAL BACKTEST PATH')
    
    # Styling
    ax.set_title("MONTE CARLO: 1,000 ALTERNATIVE FUTURES", fontsize=16, fontweight='bold', color='white')
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.grid(color='gray', linestyle='--', alpha=0.2)
    ax.legend(loc='upper left', frameon=True, facecolor='black')
    
    cursor = mplcursors.cursor([l1, l2], hover=True)
    @cursor.connect("add")
    def on_add(sel):
        label = sel.artist.get_label()
        sel.annotation.get_text_content = lambda: f"{label}\nValue: ${sel.target[1]:,.2f}"
        sel.annotation.get_bbox_patch().set(fc="black", alpha=0.8)
        sel.annotation.set_fontsize(10)

    plt.tight_layout()
    plt.show()

# Run it
run_visible_monte_carlo(history)
plt.show()
