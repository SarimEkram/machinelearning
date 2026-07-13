"""
Linear Regression Visualization (ALL PLOTS IN ONE WINDOW)
Shows normalization's effect on convergence + how well the model fits.

SETUP: imports from course1_review.py, so that file needs:
  X_lin, y_lin, predict_single, compute_cost_linear,
  compute_gradient_linear, gradient_descent_linear, zscore_normalize
  (with its test/training code wrapped in `if __name__ == "__main__":`)
"""

import numpy as np
import matplotlib.pyplot as plt

from course1_review import (
    X_lin,
    y_lin,
    predict_single,
    compute_cost_linear,
    compute_gradient_linear,
    zscore_normalize,
)


def train_with_history(X, y, alpha, num_iters):
    w = np.zeros(X.shape[1])
    b = 0.0
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_linear(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost_linear(X, y, w, b))
    return w, b, cost_history


# ---- train both versions ----
w_raw, b_raw, hist_raw = train_with_history(X_lin, y_lin, alpha=1e-5, num_iters=2000)

X_norm, mu, sigma = zscore_normalize(X_lin)
w_norm, b_norm, hist_norm = train_with_history(X_norm, y_lin, alpha=0.1, num_iters=2000)

# ---- predictions from the good (normalized) model ----
m = X_norm.shape[0]
predictions = np.zeros(m)
for i in range(m):
    predictions[i] = predict_single(X_norm[i], w_norm, b_norm)


# ============================================================
# ONE FIGURE, everything inside as subplots
# Layout: 2 rows x 3 columns
#   Row 1: [cost before] [cost after] [predicted vs actual]
#   Row 2: [size]        [bedrooms]   [age]
# ============================================================
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Linear Regression - full picture", fontsize=16)

# --- Row 1, col 0: cost BEFORE normalization ---
ax[0, 0].plot(hist_raw, color='red')
ax[0, 0].set_title('Cost BEFORE norm (alpha=1e-5)')
ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('cost')

# --- Row 1, col 1: cost AFTER normalization ---
ax[0, 1].plot(hist_norm, color='green')
ax[0, 1].set_title('Cost AFTER norm (alpha=0.1)')
ax[0, 1].set_xlabel('iteration')
ax[0, 1].set_ylabel('cost')

# --- Row 1, col 2: predicted vs actual ---
ax[0, 2].scatter(y_lin, predictions, c='blue', s=80, label='predictions')
lo = min(y_lin.min(), predictions.min())
hi = max(y_lin.max(), predictions.max())
ax[0, 2].plot([lo, hi], [lo, hi], 'r--', label='perfect')
ax[0, 2].set_title('Predicted vs Actual')
ax[0, 2].set_xlabel('actual price')
ax[0, 2].set_ylabel('predicted price')
ax[0, 2].legend()

# --- Row 2: each feature vs price ---
feature_names = ['size (1000 sqft)', 'bedrooms', 'age (years)']
for j in range(3):
    ax[1, j].scatter(X_lin[:, j], y_lin, c='red', marker='x', s=80, label='actual')
    ax[1, j].scatter(X_lin[:, j], predictions, c='blue', label='predicted')
    ax[1, j].set_xlabel(feature_names[j])
    ax[1, j].set_ylabel('price ($1000s)')
    ax[1, j].legend()

plt.tight_layout()
plt.show()


# ---- print summary ----
print(f"Cost before norm (2000 iters): {hist_raw[-1]:.2f}  (barely converged)")
print(f"Cost after  norm (2000 iters): {hist_norm[-1]:.2f}  (converged!)")
print(f"Weights: {w_norm},  bias: {b_norm:.2f}")