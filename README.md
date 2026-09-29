# Gordian Vol-Targeting: volatility-managed long exposure

A follow-up to my earlier Gordian project. Gordian tried to predict short-horizon
market direction and failed out of sample. This repo tests the other thing the same
features seemed to capture, volatility, by sizing a long position inversely to
forecast volatility. The result is a second negative result: vol targeting reliably
cuts drawdown but gives no Sharpe edge. The project is closed.

## The problem

Gordian (XGBoost direction signal, PPO sizing, hourly dollar bars) produced an
out-of-sample win rate of about 50% across three experiments, and its PBO/CSCV
overfitting test came out at 0.65, a fail. Direction was not learnable on those
assets. Volatility, by contrast, is persistent and forecastable, and the literature
(Moreira & Muir, 2017) says high-volatility periods do not pay proportionally higher
returns. If that holds, de-risking in high volatility should improve risk-adjusted
return against buy-and-hold.

I treated this as a beta plus risk-management question, not an alpha one. The
success metric was Sharpe and max drawdown against buy-and-hold. Raw ROI was not
the test, because a strategy that is de-risked on average will trail buy-and-hold in
a bull run by construction.

## The approach

A rule-based baseline with no ML:

```
w_t = clip( target_vol / forecast_vol_t , 0, vol_cap )
forecast_vol_t    = EWMA realised vol using returns up to t-1   (no look-ahead)
target_vol        = median EWMA vol on the train split          (no test leakage)
strategy_return_t = w_t * r_t  -  |w_t - w_{t-1}| * fee
```

## Key design choices

- **No ML in the baseline.** The rule has one risk parameter, so the overfitting
  surface is close to zero. I wanted the simplest version to prove the premise before
  adding an ML volatility forecast, and an ML layer was only allowed if the baseline
  cleared PBO first.
- **No leverage by default (`vol_cap = 1.0`).** Capital protection came first. I only
  relaxed it in a separate sweep, with a financing cost charged, to test whether
  leverage recovers the Sharpe gain.
- **Causal everywhere.** The forecast uses returns up to t-1 and the target comes from
  the train split only, so neither side sees the test window.
- **Positive-drift universe only.** A vol-managed long needs positive expected drift,
  so I dropped VXX and other structurally decaying or inverse products.
- **PBO via CSCV as the gate** (`src/core/pbo_validator.py`), with a target of
  PBO <= 0.20 before anything would count as real.
- **Full-history mode.** The first test-window run fell in a strong equity bull
  market, the worst regime for vol targeting, so I added a causal full-history
  evaluation (`--full`) to judge it across regimes.

## Results

These numbers come from my Colab run of `src/vol_targeting.py` (halflife 20,
fee 0.0001). That run covered six tickers: XLK, XLF, XLE, BTC-USD, ETH-USD and TQQQ.
I added SPY and QQQ to `config/config_phase1.json` afterwards; their numbers are not
reproduced here.

**Full history, no leverage (`--full`, vol_cap 1.0):**

| Ticker | Strat Sharpe | B&H Sharpe | Strat MaxDD % | B&H MaxDD % | Avg exposure |
|---|---|---|---|---|---|
| XLK | 1.22 | 1.31 | -19.16 | -26.15 | 0.81 |
| XLF | 0.78 | 0.85 | -13.23 | -17.66 | 0.83 |
| XLE | 0.80 | 0.65 | -17.39 | -23.74 | 0.87 |
| BTC-USD | 0.34 | 0.53 | -46.90 | -50.90 | 0.91 |
| ETH-USD | -0.02 | 0.04 | -63.73 | -66.66 | 0.90 |
| TQQQ | 0.94 | 1.09 | -46.30 | -58.73 | 0.85 |

Max drawdown was shallower on 6 of 6 assets. Sharpe improved on only 1 of 6 (XLE).

**Leverage sweep (`--sweep`, full history, 5%/yr borrow cost):**

| vol_cap | Mean strat Sharpe | vs B&H | Mean strat MaxDD % |
|---|---|---|---|
| 1.00 | 0.68 | -0.07 | -34.5 |
| 1.25 | 0.64 | -0.10 | -36.7 |
| 1.50 | 0.62 | -0.13 | -37.5 |
| 1.75 | 0.61 | -0.13 | -37.7 |
| 2.00 | 0.61 | -0.13 | -37.7 |

Buy-and-hold reference: mean Sharpe 0.74, mean MaxDD -40.6%. Levering up in calm
periods, which is where the literature's Sharpe gain comes from, made Sharpe worse and
drawdown deeper.

**Conclusion.** Vol targeting works as a drawdown-reduction overlay and has no Sharpe
edge on the assets I could access. Neither thesis produced a source of return:
direction had no edge, and vol targeting for Sharpe had no edge. The part that held up
was risk control, so I stopped here rather than build the PBO gate and ML layer on a
premise the baseline had already failed.

## Repo layout

- `src/vol_targeting.py`: the vol-managed backtest against buy-and-hold (Sharpe,
  MaxDD, ROI, exposure), with test-window, `--full` and `--sweep` modes.
- `src/data_factory.py`: data engine carried over from Gordian: information-driven
  dollar bars, fractional differentiation, microstructure and price features,
  point-in-time PCA. The vol baseline only needs `Close`.
- `src/core/pbo_validator.py`: probability of backtest overfitting via CSCV.
- `src/core/optimize_barriers.py`, `src/core/utils.py`: utilities carried over from Gordian.
- `config/config_phase1.json`: universe, fee and data window.

## How to run

```bash
pip install -r requirements.txt

# 1. Build data (dollar bars + Close) from yfinance (equities) and CCXT (crypto)
python src/data_factory.py --config config/config_phase1.json

# 2. Backtest vs buy-and-hold on the test window
python src/vol_targeting.py --config config/config_phase1.json

# Full history, all regimes
python src/vol_targeting.py --config config/config_phase1.json --full

# Leverage sweep (vol_cap 1.0 to 2.0 with a borrow cost)
python src/vol_targeting.py --config config/config_phase1.json --sweep
```

`--halflife`, `--vol_cap` and `--borrow_rate` adjust the forecast and risk settings.
Data lands in `data/train/` and `data/test/`, which are gitignored. The repo has no
automated test suite.

## Disclaimer

Research and educational code only, not financial advice.
