# Portfolio Optimisation — Markowitz Mean-Variance Framework

Applies Harry Markowitz's Modern Portfolio Theory (1952) to 10 NSE-listed Indian assets. Formulates the minimum-variance portfolio as a convex quadratic program, solves for optimal weights across two target return levels, and traces the full efficient frontier using CVXPY.

---

## Overview

Modern Portfolio Theory establishes that portfolio risk depends not only on individual asset volatility, but on how assets move relative to each other. By combining low- or negatively-correlated assets, total portfolio variance can be reduced without sacrificing expected return — the core principle of diversification.

This study applies that framework to a cross-sector universe of Indian equities and one gold ETF, optimised using 3-month daily closing price data from the NSE.

---

## Asset Universe

10 assets selected across sectors to maximise diversification benefit:

| Ticker | Company | Sector |
|---|---|---|
| ASIANPAINT.NS | Asian Paints Ltd. | Chemicals |
| BEL.NS | Bharat Electronics Ltd. | Electronics |
| GOLDBEES.NS | Nippon India Gold BeES ETF | Commodity (Gold) |
| HINDUNILVR.NS | Hindustan Unilever Ltd. | FMCG |
| ICICIBANK.NS | ICICI Bank Ltd. | Banking |
| INDIGO.NS | IndiGo (InterGlobe Aviation) | Aviation |
| RELIANCE.NS | Reliance Industries Ltd. | Energy |
| SBILIFE.NS | SBI Life Insurance Co. | Insurance |
| SUNPHARMA.NS | Sun Pharmaceutical Industries | Pharmaceuticals |
| TATASTEEL.NS | Tata Steel Ltd. | Metals & Mining |

Data sourced via `yfinance` over a 3-month window of daily closing prices.

---

## Mathematical Formulation

### Covariance Matrix

$$\Sigma_{ij} = \frac{1}{T-1} \sum_{t=1}^{T} (r_i^{(t)} - \mu_i)(r_j^{(t)} - \mu_j)$$

Diagonal entries $\Sigma_{ii}$ represent variance of asset $i$. Correlation is derived by normalising:

$$\rho_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}$$

### Optimisation Problem

$$\min_w \quad w^\top \Sigma w$$

$$\text{subject to:} \quad w^\top \mu \geq R_{\text{target}}, \quad \sum w_i = 1, \quad w_i \geq 0 \; \forall i$$

- **Risk objective:** minimise portfolio variance $w^\top \Sigma w$
- **Return constraint:** portfolio must achieve at least target annualised return
- **Budget constraint:** weights sum to 1 (full capital investment)
- **Long-only constraint:** no short selling

This is a **convex quadratic program** — the objective is a convex quadratic form in $w$, all constraints are linear. Any locally optimal solution is globally optimal.

---

## Implementation

### CVXPY Constructs

```python
w    = cp.Variable(n)                   # weight vector
risk = cp.quad_form(w, cov_matrix)      # w^T Σ w
portfolio_return = w @ mean_returns     # expected return

constraints = [
    cp.sum(w) == 1,                     # budget constraint
    w >= 0,                             # long-only
    portfolio_return >= target          # return constraint
]

problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()
```

If the problem is infeasible (target return exceeds what any combination of assets can achieve under long-only constraints), the solver returns `None` for `w.value`.

---

## Results

### Return Statistics

| Ticker | Avg Daily Return | Avg 3-Month Return | Avg Annualised Return |
|---|---|---|---|
| TATASTEEL.NS | 0.001763 | 0.1075 | 0.4407 |
| BEL.NS | 0.000517 | 0.0316 | 0.1294 |
| GOLDBEES.NS | 0.000216 | 0.0131 | 0.0539 |
| SUNPHARMA.NS | 0.000210 | 0.0128 | 0.0525 |
| SBILIFE.NS | -0.001518 | -0.0926 | -0.3795 |
| ICICIBANK.NS | -0.001688 | -0.1030 | -0.4220 |
| RELIANCE.NS | -0.001693 | -0.1033 | -0.4233 |
| HINDUNILVR.NS | -0.001804 | -0.1100 | -0.4509 |
| INDIGO.NS | -0.004153 | -0.2533 | -1.0382 |
| ASIANPAINT.NS | -0.004465 | -0.2724 | -1.1163 |

### Optimal Portfolio Weights

| Ticker | 10% Target | 15% Target |
|---|---|---|
| TATASTEEL.NS | 0.0477 | 0.0971 |
| SUNPHARMA.NS | 0.0333 | 0.0305 |
| BEL.NS | 0.0216 | 0.0200 |
| GOLDBEES.NS | 0.0041 | 0.0024 |
| All others | 0.0000 | 0.0000 |

> The 15% target is **infeasible** on 3-month return data — no combination of assets meets this threshold under long-only constraints. Annualised returns are used for the 15% comparison.

### Key Findings

- **GOLDBEES** is the most effective diversifier — it carries low or negative correlation with every equity in the universe, consistent with gold's role as a hedge asset
- At a **10% target**, the minimum-variance portfolio concentrates in TATASTEEL, SUNPHARMA, BEL, and GOLDBEES — the four assets with positive annualised returns
- At a **15% target**, weight shifts heavily toward TATASTEEL (9.7%) as the highest-return asset; diversification benefit diminishes
- The **efficient frontier** confirms diminishing returns to diversification at higher target returns — beyond ~50% annualised return, the portfolio concentrates in a single asset and risk rises steeply

---

## Project Structure

```
portfolio-optimisation/
│
├── portfolio_optimization.py   # Full analysis script
├── report.pdf                  # Written analysis and findings
├── README.md
│
└── plots/
    ├── correlation_heatmap.png
    └── efficient_frontier.png
```

---

## Setup & Usage

```bash
git clone https://github.com/your-username/portfolio-optimisation.git
cd portfolio-optimisation
pip install -r requirements.txt
python portfolio_optimization.py
```

### Dependencies

```
numpy==1.26.4
pandas==2.2.1
yfinance==0.2.38
cvxpy==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
```

---

## References

- Markowitz, H. (1952). Portfolio selection. *Journal of Finance, 7*(1), 77–91.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press.
- CVXPY Documentation: [cvxpy.org](https://www.cvxpy.org)

---

## Author

**Jalli Raja Nandini**
B.Tech. Industrial and Systems Engineering — IIT Kharagpur (2028)
