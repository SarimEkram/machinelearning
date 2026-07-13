import numpy as np

# ============ PART A DATA: multi-feature linear regression ============
# Predicting house price ($1000s) from [size(1000sqft), bedrooms, age(years)]
X_lin = np.array([
    [1.0, 2, 20],
    [1.5, 3, 15],
    [2.0, 3, 10],
    [2.5, 4, 8],
    [3.0, 4, 5],
    [1.2, 2, 30],
])
y_lin = np.array([300.0, 360.0, 480.0, 560.0, 640.0, 290.0])

# ============ PART B DATA: logistic regression ============
# Classifying whether a student passes (1) or fails (0) from [hours_studied, prior_gpa]
X_log = np.array([
    [1.0, 2.0],
    [2.0, 2.5],
    [1.5, 2.2],
    [4.0, 3.5],
    [5.0, 3.8],
    [4.5, 3.2],
])
y_log = np.array([0, 0, 0, 1, 1, 1])


def predict_single(x, w, b):
    return np.dot(x, w) + b


def compute_cost_linear(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = predict_single(X[i], w, b)
        cost += (f_wb - y[i])**2
    cost = cost / (2*m)
    return cost


print(compute_cost_linear(X_lin, y_lin, np.zeros(3), 0.0))