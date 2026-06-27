import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.6f}'.format
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ["TATASTEEL.NS","SBILIFE.NS","SUNPHARMA.NS", "BEL.NS", "ASIANPAINT.NS","ICICIBANK.NS", "INDIGO.NS", "HINDUNILVR.NS", "RELIANCE.NS", "GOLDBEES.NS"]
data = yf.download(tickers, period="3mo")["Close"]
returns = data.pct_change().dropna()
returns.head()

# Average daily returns
avg_returns = returns.mean()
avg_returns
# Calculating avg return across 3 months
avg_3m_returns = avg_returns * len(returns)
avg_3m_returns
# Calculating the annnualized returns
mean_returns = avg_returns * 250
mean_returns
result = pd.DataFrame({"Average Daily Return": avg_returns, "Average 3 Month Return": avg_3m_returns, "Average annualized Return": mean_returns})
result

# Correlation matrix
corr_df = returns.corr()
corr_df
# Correlation heatmap - combining low or negatively correlated assets reduces overall portfolio variance
plt.figure(figsize=(8,6))
sns.heatmap(data=corr_df, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.xticks(rotation=45)
plt.show()

# Covariance matrix
cov_df = returns.cov()
cov_df
# Converting covariance df to a matrix
np.set_printoptions(suppress=True, precision=6)
cov_matrix = cov_df.to_numpy(dtype = float)
print(cov_matrix)

#Finding Optimal Risk for 3 Months of Data
import cvxpy as cp
# Defining weights
w = cp.Variable(len(avg_3m_returns))
risk = cp.quad_form(w, cov_matrix)
portfolio_return = w @ avg_3m_returns

#3 months return data is being analysed
#10% target
# Defining constraints
constraints = [cp.sum(w) == 1, w >= 0, portfolio_return >=0.10]
# Optimization problem
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()
optimal_weights = w.value
print(optimal_weights)
target10 = avg_3m_returns * optimal_weights
target10
risk10 = avg_3m_returns @ optimal_weights
print('Portfolio return =', risk10)

#15% target
# Defining constraints
constraints = [cp.sum(w) == 1, w>=0, portfolio_return >= 0.15]
# Optimization problem
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()
optimal_weights = w.value
print(optimal_weights)

''We observe that if we take only 3 months of data, we cannot obtain any optimal weights for a total return of 15%. This is because the individual average 3 month return for each asset is very low, and with any combiantion of assets we cannot find an optimal solution. Therefore, this problem is infeasible for a return of 15%

We can take average annualized return for each asset instead to compare the risk for targets of 10% and 15%'''

#finding optimal risk for annualised return data
# Defining weights
w = cp.Variable(len(mean_returns))
risk = cp.quad_form(w, cov_matrix)
portfolio_return = w @ mean_returns

#10% target - annualised returns
# Defining constraints
constraints = [cp.sum(w) == 1, w >= 0, portfolio_return >=0.10]
# Optimization problem
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()
optimal_weights = w.value
print(optimal_weights)
target10 = mean_returns * optimal_weights
target10
risk10 = mean_returns @ optimal_weights
print('Portfolio return =', risk10)

#15% target - annualised returns
# Defining constraints
constraints = [cp.sum(w) == 1, w>=0, portfolio_return >= 0.15]
# Optimization problem
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()
optimal_weights = w.value
print(optimal_weights)
target15 = mean_returns * optimal_weights
target15
risk15 = mean_returns @ optimal_weights
print('Portfolio return =', risk15)

#comparison
comparision = pd.DataFrame({'10% Target': target10, '15% Target': target15})
comparision

#efficient frontier
returns = []
risks = []
weights_list = []

targets = np.arange(0.1, 1.1, 0.1)
targets
for target in targets:
  w = cp.Variable(len(mean_returns))
  risk = cp.quad_form(w, cov_matrix)
  portfolio_return = w @ mean_returns

  constraints = [cp.sum(w) == 1, w >= 0, portfolio_return >= target]

  problem = cp.Problem(cp.Minimize(risk), constraints)
  problem.solve()

  if w.value is not None:
    optimal_w = w.value

    port_return = np.dot(mean_returns, optimal_w)
    port_risk = (optimal_w.T @ cov_matrix @ optimal_w)

    returns.append(port_return)
    risks.append(port_risk)
    weights_list.append(optimal_w)
plt.plot(risks, returns, label="Efficient Frontier")

plt.xlabel("Risk (variance)")
plt.ylabel("Return")
plt.title("Efficient Frontier")

plt.legend()
plt.grid()

plt.show()
