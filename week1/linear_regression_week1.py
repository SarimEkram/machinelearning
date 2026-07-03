import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# 1. TRAINING DATA
# ============================================================
x_train = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0])
y_train = np.array([300.0, 320.0, 340.0, 360.0, 400.0, 420.0, 460.0, 470.0, 500.0])

m = x_train.shape[0]
print(f"Number of training examples: {m}")


# ============================================================
# 2. MODEL: f_wb(x) = w*x + b
# ============================================================
def predict(x, w, b):
    """
    Single-value prediction.
    Args:
      x (scalar): one input (e.g. one house size)
      w, b (scalar): model parameters
    Returns:
      prediction (scalar)
    """
    return w * x + b


def compute_model_output(x, w, b):
    """
    Computes predictions for an entire array of inputs.
    Args:
      x (ndarray (m,)): m examples
      w, b (scalar): model parameters
    Returns:
      f_wb (ndarray (m,)): predictions for each example
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


# ============================================================
# 3. COST FUNCTION: J(w,b) = (1/2m) * sum((f_wb - y)^2)
# ============================================================
def compute_cost(x, y, w, b):
    """
    Computes the cost (how bad w,b currently are).
    Args:
      x, y (ndarray (m,)): data and targets
      w, b (scalar): model parameters
    Returns:
      total_cost (float): average squared error
    """
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost          # accumulate, not overwrite
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


# ============================================================
# 4. GRADIENT: dJ/dw and dJ/db
# ============================================================
def compute_gradient(x, y, w, b):
    """
    Computes the gradient (slope) of the cost function
    with respect to w and b, at the current w,b.
    Args:
      x, y (ndarray (m,)): data and targets
      w, b (scalar): model parameters
    Returns:
      dj_dw, dj_db (scalar): slopes in the w and b directions
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]   # note the extra x[i] here
        dj_db_i = f_wb - y[i]            # no x[i] multiplier for b
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


# ============================================================
# 5. GRADIENT DESCENT: repeatedly step downhill
# ============================================================
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b.
    Args:
      x, y (ndarray (m,)): data and targets
      w_in, b_in (scalar): initial parameter guesses
      alpha (float): learning rate (step size)
      num_iters (int): number of steps to take
      cost_function: function used to compute J (for logging only)
      gradient_function: function used to compute dj_dw, dj_db
    Returns:
      w, b (scalar): final trained parameters
      J_history (list): cost at each iteration
      p_history (list): [w,b] at each iteration
    """
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # 1. Feel the slope
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # 2. Step downhill
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # 3. Log progress (diagnostic only, doesn't affect training)
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # 4. Print occasional status updates
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


# ============================================================
# 6. RUN IT
# ============================================================
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha,
    iterations, compute_cost, compute_gradient
)

print(f"\n(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# ============================================================
# 7. PREDICT WITH THE TRAINED MODEL
# ============================================================
x_i = 1.2  # 1200 sqft
predicted_price = predict(x_i, w_final, b_final)
print(f"Predicted price for a 1200 sqft house: ${predicted_price:.0f} thousand dollars")

# ============================================================
# 8. PLOT: data points + fitted line
# ============================================================
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
tmp_f_wb = compute_model_output(x_train, w_final, b_final)
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# ============================================================
# 9. PLOT: cost vs iteration (confirm gradient descent converged)
# ============================================================
plt.plot(J_hist)
plt.title("Cost vs. Iteration")
plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.show()