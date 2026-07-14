"""
Logistic Regression Visualization
Trains the classifier and shows:
  1. Cost dropping during training (proof it learned)
  2. The data + decision boundary (the line separating pass/fail)
  3. Predicted vs actual labels

SETUP: imports from course1_review.py, which must define:
  X_log, y_log, sigmoid, predict_single,
  compute_cost_logistic, compute_gradient_logistic,
  gradient_descent_logistic, predict_logistic
  (with its own test/training code wrapped in `if __name__ == "__main__":`)
"""

import numpy as np
import matplotlib.pyplot as plt

from course1_review import (
    X_log,
    y_log,
    sigmoid,
    predict_single,
    compute_cost_logistic,
    compute_gradient_logistic,
    predict_logistic,
)


# ============================================================
# Train, but record cost each iteration for the cost curve
# ============================================================
def train_logistic_with_history(X, y, alpha, num_iters):
    w = np.zeros(X.shape[1])
    b = 0.0
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost_logistic(X, y, w, b))
    return w, b, cost_history


w, b, cost_history = train_logistic_with_history(X_log, y_log, alpha=0.1, num_iters=10000)

# get hard 0/1 predictions
preds = predict_logistic(X_log, w, b)
accuracy = np.mean(preds == y_log) * 100

print("Learned w:", w, " b:", b)
print("Predictions:", preds)
print("Actual:     ", y_log.astype(float))
print(f"Accuracy: {accuracy:.0f}%")


# ============================================================
# ONE window, 3 panels
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Logistic Regression - student pass/fail classifier", fontsize=15)

# ---- Panel 1: cost vs iteration ----
ax[0].plot(cost_history, color='purple')
ax[0].set_title('Cost during training')
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('cost')

# ---- Panel 2: data + decision boundary ----
# separate the two classes for plotting
pass_mask = (y_log == 1)
fail_mask = (y_log == 0)

ax[1].scatter(X_log[pass_mask, 0], X_log[pass_mask, 1],
              c='green', marker='o', s=90, label='pass (y=1)')
ax[1].scatter(X_log[fail_mask, 0], X_log[fail_mask, 1],
              c='red', marker='x', s=90, label='fail (y=0)')

# decision boundary: where w0*x0 + w1*x1 + b = 0
#   -> x1 = -(w0*x0 + b) / w1
x0_line = np.linspace(X_log[:, 0].min() - 0.5, X_log[:, 0].max() + 0.5, 100)
x1_line = -(w[0] * x0_line + b) / w[1]
ax[1].plot(x0_line, x1_line, 'b--', label='decision boundary')

ax[1].set_title('Data + decision boundary')
ax[1].set_xlabel('hours studied')
ax[1].set_ylabel('prior GPA')
ax[1].legend()

# ---- Panel 3: predicted vs actual ----
idx = np.arange(len(y_log))
ax[2].scatter(idx, y_log, c='red', marker='x', s=120, label='actual')
ax[2].scatter(idx, preds, c='blue', marker='o', s=40, label='predicted')
ax[2].set_title(f'Predicted vs Actual  (accuracy {accuracy:.0f}%)')
ax[2].set_xlabel('student index')
ax[2].set_ylabel('class (0=fail, 1=pass)')
ax[2].set_yticks([0, 1])
ax[2].legend()

plt.tight_layout()
plt.show()