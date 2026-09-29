"""
Volatility-Managed Long Exposure -- rule-based baseline (no ML).

Directional prediction was disproven out-of-sample (see project notes). This
pivots to using the assets' forecastable VOLATILITY rather than their direction.

Hypothesis (Moreira & Muir, 2017): volatility is persistent and forecastable,
while high-vol periods do NOT pay proportionally higher returns. So scaling a
LONG position inversely to forecast vol improves risk-adjusted return (Sharpe)
and cuts drawdown versus static buy-and-hold.

Rule (deterministic, ONE risk parameter -> near-zero overfitting surface):
    w_t = clip( target_vol / forecast_vol_t , 0, vol_cap )
    forecast_vol_t = EWMA realised vol using returns up to t-1 (no look-ahead)
    target_vol     = median EWMA vol on the TRAIN split (no test leakage)
    strategy_return_t = w_t * r_t  -  |w_t - w_{t-1}| * fee

Capital protection: vol_cap = 1.0 (no leverage).

This is a BETA + RISK-MANAGEMENT product, not directional alpha: it captures the
asset's drift with a better Sharpe / shallower drawdown. Success is measured on
Sharpe and max drawdown vs buy-and-hold, NOT on raw ROI.

No new dependencies (numpy / pandas only).
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"


def ewma_vol(returns, halflife):
    """EWMA volatility (std of returns). Backward-looking by construction."""
    var = returns.pow(2).ewm(halflife=halflife, min_periods=halflife).mean()
    return np.sqrt(var)


def max_drawdown(equity):
    """Max peak-to-trough decline of an equity curve (returns a negative fraction)."""
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def annualized_sharpe(returns, periods_per_year):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


def periods_per_year(test_df):
    """Estimate bars/year from the Date span (dollar bars are not uniform in time)."""
    if 'Date' not in test_df.columns:
        return 2520.0  # ~10 bars/day * 252
    dates = pd.to_datetime(test_df['Date'])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return (len(test_df) / years) if years > 0 else 2520.0


def backtest_ticker(ticker, fee, halflife=20, vol_cap=1.0, borrow_rate=0.05):
    train_path = os.path.join(TRAIN_DIR, f"{ticker}_data.csv")
    test_path = os.path.join(TEST_DIR, f"{ticker}_data.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if 'Close' not in train.columns or 'Close' not in test.columns or len(test) < halflife + 5:
        return None

    # Target vol from TRAIN only (no test leakage): typical vol -> avg exposure ~1.0
    target_vol = float(ewma_vol(train['Close'].pct_change(), halflife).median())

    r = test['Close'].pct_change().fillna(0.0)
    # Forecast for bar t uses returns up to t-1 (shift) -> strictly no look-ahead
    fvol = ewma_vol(r, halflife).shift(1)
    w = (target_vol / fvol).clip(lower=0.0, upper=vol_cap).fillna(0.0)

    ppy = periods_per_year(test)
    strat_r = w * r
    turnover = w.diff().abs().fillna(w.abs())
    # Financing cost on the BORROWED (levered) portion only -- 0 when vol_cap<=1.0
    financing = (w - 1.0).clip(lower=0.0) * (borrow_rate / ppy)
    strat_r_net = strat_r - turnover * fee - financing

    eq = (1.0 + strat_r_net).cumprod()
    bh_eq = (1.0 + r).cumprod()

    return {
        "Ticker": ticker,
        "Strat ROI %": (eq.iloc[-1] - 1) * 100,
        "B&H ROI %": (bh_eq.iloc[-1] - 1) * 100,
        "Strat Sharpe": annualized_sharpe(strat_r_net, ppy),
        "B&H Sharpe": annualized_sharpe(r, ppy),
        "Strat MaxDD %": max_drawdown(eq.values) * 100,
        "B&H MaxDD %": max_drawdown(bh_eq.values) * 100,
        "Avg Expo": float(w.mean()),
        "strat_returns": strat_r_net,  # kept for downstream PBO; dropped from the printed table
    }


def backtest_ticker_full(ticker, fee, halflife=20, vol_cap=1.0, borrow_rate=0.05, warmup=250):
    """
    Evaluate over the FULL price history (train+test concatenated), across all
    regimes -- not just the recent ~20% test window, which can be a single
    favourable/unfavourable regime. Uses ALL available data (the opposite of
    cherry-picking a window).

    Fully causal -> no look-ahead, no leakage:
      - returns computed per-segment then concatenated (no spurious return across
        the train/test embargo gap),
      - forecast vol = EWMA shifted by one bar (uses returns up to t-1),
      - target vol = EXPANDING median of the (already-causal) vol forecast, so it
        only ever sees the past; the position is flat during the warmup window.
    """
    train_path = os.path.join(TRAIN_DIR, f"{ticker}_data.csv")
    test_path = os.path.join(TEST_DIR, f"{ticker}_data.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if 'Close' not in train.columns or 'Close' not in test.columns:
        return None

    r = pd.concat([train['Close'].pct_change(), test['Close'].pct_change()],
                  ignore_index=True).fillna(0.0)
    dates = None
    if 'Date' in train.columns and 'Date' in test.columns:
        dates = pd.concat([pd.to_datetime(train['Date']), pd.to_datetime(test['Date'])],
                          ignore_index=True)
    if len(r) < warmup + 50:
        return None

    # Bars/year from the full date span (used for both financing and annualization)
    if dates is not None:
        yrs = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        ppy = (len(r) / yrs) if yrs > 0 else 2520.0
    else:
        ppy = 2520.0

    fvol = ewma_vol(r, halflife).shift(1)
    target = fvol.expanding(min_periods=warmup).median()  # causal self-calibration
    w = (target / fvol).clip(lower=0.0, upper=vol_cap).fillna(0.0)  # flat during warmup

    strat_r = w * r
    turnover = w.diff().abs().fillna(w.abs())
    # Financing cost on the BORROWED (levered) portion only -- 0 when vol_cap<=1.0
    financing = (w - 1.0).clip(lower=0.0) * (borrow_rate / ppy)
    strat_r_net = strat_r - turnover * fee - financing

    # Metrics over the traded span only (post-warmup)
    sr = strat_r_net.iloc[warmup:]
    rr = r.iloc[warmup:]
    eq = (1.0 + sr).cumprod()
    bh_eq = (1.0 + rr).cumprod()

    return {
        "Ticker": ticker,
        "Strat ROI %": (eq.iloc[-1] - 1) * 100,
        "B&H ROI %": (bh_eq.iloc[-1] - 1) * 100,
        "Strat Sharpe": annualized_sharpe(sr, ppy),
        "B&H Sharpe": annualized_sharpe(rr, ppy),
        "Strat MaxDD %": max_drawdown(eq.values) * 100,
        "B&H MaxDD %": max_drawdown(bh_eq.values) * 100,
        "Avg Expo": float(w.iloc[warmup:].mean()),
        "Bars": int(len(rr)),
        "strat_returns": sr,
    }


def run_sweep(tickers, fee, halflife=20, borrow_rate=0.05, caps=(1.0, 1.25, 1.5, 1.75, 2.0)):
    """Leverage tradeoff curve: how mean Sharpe and mean drawdown move as the
    leverage cap rises. Always full-history (leverage must be judged across
    regimes). Financing is charged on the borrowed portion."""
    print("=" * 110)
    print("LEVERAGE SWEEP  (full history, all regimes, causal)  -- does levering up in calm recover a Sharpe edge?")
    print(f"halflife={halflife}  fee={fee}  borrow_rate={borrow_rate*100:.0f}%/yr")
    print("=" * 110)

    bh_printed = False
    print(f"{'vol_cap':>8} {'mean Strat Sharpe':>18} {'vs B&H':>8} {'mean Strat MaxDD%':>18} {'mean Avg Expo':>14}")
    print("-" * 110)
    for cap in caps:
        rows = [backtest_ticker_full(tk, fee, halflife, cap, borrow_rate) for tk in tickers]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if not bh_printed:
            bh_s, bh_dd = df["B&H Sharpe"].mean(), df["B&H MaxDD %"].mean()
        ms, mdd, mexp = df["Strat Sharpe"].mean(), df["Strat MaxDD %"].mean(), df["Avg Expo"].mean()
        print(f"{cap:>8.2f} {ms:>18.2f} {ms - df['B&H Sharpe'].mean():>+8.2f} {mdd:>18.1f} {mexp:>14.2f}")
        bh_printed = True
    print("-" * 110)
    print(f"Buy & Hold reference:  mean Sharpe {bh_s:+.2f}   mean MaxDD {bh_dd:.1f}%")
    print("=" * 110)
    print("Read: leverage should LIFT mean Strat Sharpe above B&H -- but watch MaxDD deepen as the cap rises.")
    print("Capital protection was the reason for the 1.0 cap; this quantifies the Sharpe gain vs the drawdown cost.")


def main(tickers, fee, halflife=20, vol_cap=1.0, full=False, borrow_rate=0.05):
    span = "FULL HISTORY (all regimes, causal)" if full else "TEST WINDOW (recent ~20% OOS)"
    print("=" * 110)
    print(f"VOLATILITY-MANAGED LONG EXPOSURE  (rule-based baseline, no ML)  |  span = {span}")
    lev = f"  borrow_rate={borrow_rate*100:.0f}%/yr" if vol_cap > 1.0 else ""
    print(f"halflife={halflife}  vol_cap={vol_cap}  fee={fee}{lev}  |  success metric = Sharpe & MaxDD vs Buy & Hold")
    print("=" * 110)

    rows = []
    for tk in tickers:
        res = (backtest_ticker_full(tk, fee, halflife, vol_cap, borrow_rate) if full
               else backtest_ticker(tk, fee, halflife, vol_cap, borrow_rate))
        if res is None:
            print(f"  (skipped {tk}: data missing or too short)")
            continue
        rows.append(res)

    if not rows:
        print("No results -- check that data/test and data/train exist for the configured tickers.")
        return

    df = pd.DataFrame(rows)
    show = df.drop(columns=["strat_returns"]).copy()
    for c in show.columns:
        if c != "Ticker":
            show[c] = show[c].astype(float).round(2)
    print(show.to_string(index=False))
    print("-" * 110)

    wins = int((df["Strat Sharpe"] > df["B&H Sharpe"]).sum())
    dd_better = int((df["Strat MaxDD %"] > df["B&H MaxDD %"]).sum())  # less negative = shallower
    print(f"Sharpe improved vs B&H on {wins}/{len(df)} assets.")
    print(f"Max drawdown shallower vs B&H on {dd_better}/{len(df)} assets.")
    print(f"Mean Strat Sharpe: {df['Strat Sharpe'].mean():+.2f}   |   Mean B&H Sharpe: {df['B&H Sharpe'].mean():+.2f}")
    print("=" * 110)
    print("Read: vol targeting should LIFT Sharpe and SHALLOW drawdown; raw ROI may trail B&H in")
    print("bull runs (it is de-risked on average). If Sharpe doesn't beat B&H here, the premise fails.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based volatility-managed long exposure backtest")
    parser.add_argument('--config', type=str, default='config/config_phase1.json')
    parser.add_argument('--halflife', type=int, default=20, help="EWMA halflife (bars) for the vol forecast.")
    parser.add_argument('--vol_cap', type=float, default=1.0, help="Max position weight (1.0 = no leverage; >1 = leverage).")
    parser.add_argument('--full', action='store_true',
                        help="Evaluate over the full train+test history (all regimes), not just the test window.")
    parser.add_argument('--borrow_rate', type=float, default=0.05,
                        help="Annualized financing cost charged on the borrowed (levered) portion. Default 5%%.")
    parser.add_argument('--sweep', action='store_true',
                        help="Sweep vol_cap 1.0->2.0 (full history) to show the Sharpe gain vs drawdown cost of leverage.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)
    tickers = cfg.get("tickers", [])
    fee = cfg.get("transaction_fee_percent", 0.0001)

    if args.sweep:
        run_sweep(tickers, fee, halflife=args.halflife, borrow_rate=args.borrow_rate)
    else:
        main(tickers, fee, halflife=args.halflife, vol_cap=args.vol_cap, full=args.full, borrow_rate=args.borrow_rate)
