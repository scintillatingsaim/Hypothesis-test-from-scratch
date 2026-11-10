import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import ttest_ind, t

# Step 1: State the hypothesis
print("RESEARCH QUESTION")
print("Do Monday returns differ from returns on Tuesday-Friday?")
print("\nNULL HYPOTHESIS (H0)")
print("Mean Monday return = Mean Tuesday-Friday return")
print("\nALTERNATIVE HYPOTHESIS (H1)")
print("Mean Monday return != Mean Tuesday-Friday return")

# Step 2: Set significance level
alpha = 0.05

# Step 3: Download market data
prices = yf.download("^NSEI", start="2015-01-01", end="2026-01-01", auto_adjust=False)["Close"]
prices = prices.squeeze()
prices = prices.dropna()

# Step 4: Calculate daily returns
returns = prices.pct_change().dropna()

# Step 5: Identify the weekday
data = pd.DataFrame({"return": returns})
data["weekday"] = data.index.day_name()

# Step 6: Create the two samples
monday = data.loc[data["weekday"] == "Monday", "return"]
other_days = data.loc[data["weekday"].isin(["Tuesday", "Wednesday", "Thursday", "Friday"]), "return"]

# Step 7: Descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print("\nMonday:")
print("Number of observations:", len(monday))
print("Mean:", monday.mean())
print("Standard deviation:", monday.std())
print("\nTuesday-Friday:")
print("Number of observations:", len(other_days))
print("Mean:", other_days.mean())
print("Standard deviation:", other_days.std())

# Step 8: Observed difference in means
mean_monday = monday.mean()
mean_other = other_days.mean()
difference = mean_monday - mean_other
print("\nOBSERVED DIFFERENCE")
print("Monday mean - Other-day mean =", difference)

# Step 9: Welch t-test by hand
n_monday = len(monday)
n_other = len(other_days)
var_monday = monday.var(ddof=1)
var_other = other_days.var(ddof=1)
standard_error = np.sqrt(var_monday / n_monday + var_other / n_other)
t_stat_hand = difference / standard_error
df = (var_monday / n_monday + var_other / n_other) ** 2 / (((var_monday / n_monday) ** 2) / (n_monday - 1) + ((var_other / n_other) ** 2) / (n_other - 1))
p_value_hand = 2 * t.sf(abs(t_stat_hand), df)

print("\nWELCH T-TEST — HAND CALCULATION")
print("Monday variance:", var_monday)
print("Other-day variance:", var_other)
print("Standard error:", standard_error)
print("t-statistic:", t_stat_hand)
print("Degrees of freedom:", df)
print("p-value:", p_value_hand)

# Step 10: Welch t-test using Python
t_stat_code, p_value_code = ttest_ind(monday, other_days, equal_var=False)
print("\nWELCH T-TEST — PYTHON")
print("t-statistic:", t_stat_code)
print("p-value:", p_value_code)

# Step 11: Compare hand calculation and Python
print("\nCOMPARISON")
print("Difference between t-statistics:", abs(t_stat_hand - t_stat_code))
print("Difference between p-values:", abs(p_value_hand - p_value_code))

# Step 12: Statistical decision
print("\nSTATISTICAL DECISION")
if p_value_code < alpha:
    print("p-value < 0.05")
    print("Reject the null hypothesis.")
    print("There is statistically significant evidence that Monday returns differ from Tuesday-Friday returns.")
else:
    print("p-value >= 0.05")
    print("Fail to reject the null hypothesis.")
    print("There is not sufficient statistical evidence that Monday returns differ from Tuesday-Friday returns.")

# Step 13: Honest interpretation
print("\nHONEST INTERPRETATION")
if p_value_code < alpha:
    if difference > 0:
        print("Monday had a higher average return than Tuesday-Friday, and the difference was statistically significant at the 5% level.")
    else:
        print("Monday had a lower average return than Tuesday-Friday, and the difference was statistically significant at the 5% level.")
else:
    print("Although the sample means may differ, the difference is not statistically significant at the 5% level.")
    print("Therefore, this test does not provide sufficient evidence for a Monday effect.")
