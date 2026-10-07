"""
Medical Cost Optimization
Linear Regression + Convex Optimization (PPDAC Framework)
"""

import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import r2_score # pyright: ignore[reportMissingModuleSource]
from scipy.optimize import linprog, minimize # pyright: ignore[reportMissingImports]

# ── 1. LOAD & ENCODE DATA ─────────────────────────────────────────────────────
df = pd.read_csv("medical_cost.csv")

df["sex"]    = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({"northeast": 0, "northwest": 1,
                                  "southeast": 2, "southwest": 3})

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
X = df[FEATURES].values
y = df["charges"].values

# ── 2. LINEAR REGRESSION ──────────────────────────────────────────────────────
model = LinearRegression().fit(X, y)
c = model.coef_        # regression coefficients
d = model.intercept_   # intercept

print(f"R² Score : {r2_score(y, model.predict(X)):.4f}")
print("\nCoefficients:")
for name, coef in zip(FEATURES, c):
    print(f"  {name:<10} : {coef:>10.2f}")
print(f"  {'intercept':<10} : {d:>10.2f}")

# ── 3. VISUALIZATION ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Fig 1 — Feature Impact
colors = ["#C00000" if v > 0 else "#70AD47" for v in c]
axes[0].barh(FEATURES, c, color=colors)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_title("Feature Impact on Charges")
axes[0].set_xlabel("Coefficient")

# Fig 2 — Charge Distribution
axes[1].hist(df["charges"], bins=30, color="#4472C4", edgecolor="white")
axes[1].set_title("Distribution of Medical Charges")
axes[1].set_xlabel("Charges (USD)")
axes[1].set_ylabel("Frequency")

# Fig 3 — Smoker vs Non-Smoker
groups = [df[df["smoker"] == 0]["charges"], df[df["smoker"] == 1]["charges"]]
bp = axes[2].boxplot(groups, patch_artist=True, tick_labels=["Non-Smoker", "Smoker"])
bp["boxes"][0].set_facecolor("#4472C4")
bp["boxes"][1].set_facecolor("#C00000")
axes[2].set_title("Charges: Smoker vs Non-Smoker")
axes[2].set_ylabel("Charges (USD)")

plt.tight_layout()
plt.savefig("medical_cost_plots.png", dpi=150)
plt.show()

# ── 4. VARIABLE BOUNDS ────────────────────────────────────────────────────────
# [age, sex, bmi, children, smoker(fixed=0), region]
BOUNDS = [(18, 64), (0, 1), (18.5, 30), (0, 5), (0, 0), (0, 3)]

# ── 5. PROBLEM 1 — MINIMIZE PREDICTED CHARGES (LP) ───────────────────────────
result1 = linprog(c=c, bounds=BOUNDS, method="highs")
x1      = result1.x
charges1 = c @ x1 + d

print("\n── Problem 1: Minimize Charges ──")
for name, val in zip(FEATURES, x1):
    print(f"  {name:<10} = {val:.2f}")
print(f"  Predicted Charges = ${charges1:,.2f}")

# ── 6. PROBLEM 2 — MAXIMIZE EXPECTED PROFIT (L-BFGS-B) ───────────────────────
# Premium = 0.15 × charges + 600
# Profit  = Premium − charges = −0.85 × charges + 600
# Maximize Profit → Minimize −Profit

def neg_profit(x):
    charges = c @ x + d
    premium = 0.15 * charges + 600
    return -(premium - charges)

x0      = [np.mean(b) if b[0] != b[1] else b[0] for b in BOUNDS]
result2 = minimize(neg_profit, x0, method="L-BFGS-B", bounds=BOUNDS)
x2      = result2.x
charges2 = c @ x2 + d
premium2 = 0.15 * charges2 + 600
profit2  = premium2 - charges2

print("\n── Problem 2: Maximize Profit ──")
for name, val in zip(FEATURES, x2):
    print(f"  {name:<10} = {val:.2f}")
print(f"  Predicted Charges = ${charges2:,.2f}")
print(f"  Premium           = ${premium2:,.2f}")
print(f"  Expected Profit   = ${profit2:,.2f}")
