# ML Practice Assignment: Build Two Models From Scratch

A consolidation exercise covering everything from Course 1. Write every function
yourself. Formulas and guidance are provided; the code is not. Fill in each
`# YOUR CODE HERE` block.

Rule for yourself: do NOT look back at the course labs until you've attempted a
function. Struggle first, check second. That's how it sticks.

---

## Setup (given to you, just run it)

```python
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
```

---

# PART A — Multivariable Linear Regression

## A1. `predict_single(x, w, b)`

Predict for ONE example (a single row of features).

**Formula:**
$$f_{w,b}(x) = w \cdot x + b = \sum_{j} w_j x_j + b$$

**Guidance:**
- `x` is a 1-D array of features (e.g. `[1.0, 2, 20]`), `w` is a 1-D array of weights, `b` is a scalar.
- Use `np.dot(x, w) + b`. One line.
- **Do NOT** loop manually — use `np.dot`. You've earned the right to vectorize.

```python
def predict_single(x, w, b):
    # YOUR CODE HERE
    pass
```

**Test:** `predict_single(np.array([1.0,2,20]), np.array([100,20,-1]), 50)` should give
`100*1 + 20*2 + (-1)*20 + 50 = 170`.

---

## A2. `compute_cost_linear(X, y, w, b)`

**Formula:**
$$J(w,b) = \frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2$$

**Guidance:**
- Loop over `m` examples. For each, predict with `np.dot(X[i], w) + b`, subtract `y[i]`, square it, accumulate.
- Divide the total by `2*m` at the end (remember: `2*m`, group it — `/(2*m)`, NOT `/2*m`).
- The division goes OUTSIDE the loop.
- **Do NOT** forget the `2` in the denominator (that's linear regression's signature; logistic won't have it).

```python
def compute_cost_linear(X, y, w, b):
    m = X.shape[0]
    # YOUR CODE HERE
    pass
```

---

## A3. `compute_gradient_linear(X, y, w, b)`

**Formulas:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})\,x_j^{(i)}$$
$$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})$$

**Guidance:**
- `dj_dw` must be an ARRAY of size `n` (one per feature): start with `dj_dw = np.zeros(n)`.
- `dj_db` is a scalar: start at `0`.
- Outer loop over examples `i`. Compute the error `err = prediction - y[i]` ONCE per example.
- Inner loop over features `j`: `dj_dw[j] += err * X[i,j]`.
- `dj_db += err` (no feature multiplier — b's derivative multiplier is 1).
- Divide BOTH by `m` after the loops.
- **Do NOT** use `2*m` here — the gradient has only `1/m` (the 2 cancelled in the derivative).
- **Do NOT** overwrite with `=`; accumulate with `+=`.

```python
def compute_gradient_linear(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    # YOUR CODE HERE
    return dj_dw, dj_db
```

---

## A4. `gradient_descent_linear(X, y, w_in, b_in, alpha, num_iters)`

**Update rule (repeat num_iters times):**
$$w_j = w_j - \alpha\frac{\partial J}{\partial w_j} \qquad b = b - \alpha\frac{\partial J}{\partial b}$$

**Guidance:**
- Copy `w_in`, `b_in` into `w`, `b` (use `w = w_in.copy()` so you don't modify the original).
- Loop `num_iters` times. Each iteration:
  1. Call `compute_gradient_linear` to get `dj_dw, dj_db`.
  2. Update: `w = w - alpha * dj_dw` (this works on the whole array at once — broadcasting).
  3. `b = b - alpha * dj_db`.
- Optionally every ~100 iters, print the cost so you can watch it drop.
- **Do NOT** compute the gradient once outside the loop — it must be recomputed every iteration (w,b changed).

```python
def gradient_descent_linear(X, y, w_in, b_in, alpha, num_iters):
    w = w_in.copy()
    b = b_in
    # YOUR CODE HERE
    return w, b
```

**Run it:**
```python
w0 = np.zeros(X_lin.shape[1])
b0 = 0.
w_lin, b_lin = gradient_descent_linear(X_lin, y_lin, w0, b0, alpha=1e-2, num_iters=1000)
print("Learned w:", w_lin, "b:", b_lin)
```

**WATCH OUT:** with these raw features (size ~1-3, age ~5-30), the scales differ.
If your cost EXPLODES to `inf`/`nan`, alpha is too big for the unscaled data.
Two options: (a) lower alpha a lot (try `1e-4` or smaller), or (b) bonus: z-score
normalize X first (see bonus section). This is the feature-scaling lesson biting
in practice — expect it and don't panic.

---

## A5. Predict on the whole set + check

```python
# predict all, compare to actual
for i in range(X_lin.shape[0]):
    pred = predict_single(X_lin[i], w_lin, b_lin)
    print(f"predicted {pred:.1f}  vs  actual {y_lin[i]:.1f}")
```

Predictions should land reasonably close to actuals. They won't be perfect (real-ish
data), but should track the trend.

---

# PART B — Logistic Regression

Same skeleton as Part A, with THREE changes:
1. Prediction gets wrapped in **sigmoid**.
2. Cost uses **logistic loss** instead of squared error (and `/m`, not `/2m`).
3. A **`predict`** function applies a 0.5 threshold to output 0/1.

The gradient formulas look IDENTICAL to linear — the only difference is that
`f_wb` now means `sigmoid(w·x + b)`.

## B1. `sigmoid(z)`

**Formula:**
$$g(z) = \frac{1}{1 + e^{-z}}$$

**Guidance:**
- Use `np.exp(-z)`. NOT `** z`. The `e` is `np.exp`, and the exponent is NEGATIVE z.
- One line. Works on scalar or array automatically (np.exp handles both).

```python
def sigmoid(z):
    # YOUR CODE HERE
    pass
```

**Test:** `sigmoid(0)` = 0.5, `sigmoid(large positive)` ≈ 1, `sigmoid(large negative)` ≈ 0.

---

## B2. `compute_cost_logistic(X, y, w, b)`

**Formula:**
$$J(w,b) = \frac{1}{m}\sum_{i=0}^{m-1}\left[-y^{(i)}\log(f^{(i)}) - (1-y^{(i)})\log(1-f^{(i)})\right]$$
where $f^{(i)} = \text{sigmoid}(w\cdot x^{(i)} + b)$

**Guidance:**
- Loop over examples. For each: compute `z = np.dot(X[i], w) + b`, then `f = sigmoid(z)`.
- Add the loss: `-y[i]*np.log(f) - (1-y[i])*np.log(1-f)`.
- **Do NOT** use a `+` between the two terms — it's a MINUS: `-y*log(f) - (1-y)*log(1-f)`.
- Divide by `m` (NOT `2m` — logistic has no 2).
- Division outside the loop.

```python
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    # YOUR CODE HERE
    pass
```

---

## B3. `compute_gradient_logistic(X, y, w, b)`

**Formulas (identical shape to linear, but f is sigmoid-based):**
$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum(f^{(i)} - y^{(i)})x_j^{(i)} \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum(f^{(i)} - y^{(i)})$$

**Guidance:**
- Structure is EXACTLY your linear gradient, except `f_wb = sigmoid(np.dot(X[i],w)+b)`.
- `err = f_wb - y[i]`, then `dj_dw[j] += err * X[i,j]`, `dj_db += err`.
- Divide both by `m`.
- **Do NOT** add any regularization (we're keeping it simple).

```python
def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    # YOUR CODE HERE
    return dj_dw, dj_db
```

---

## B4. `gradient_descent_logistic(...)`

**Guidance:**
- IDENTICAL to your linear gradient_descent, just call `compute_gradient_logistic` instead.
- Same update rule: `w = w - alpha*dj_dw`, `b = b - alpha*dj_db`.
- You can literally copy your linear version and swap the gradient function name.

```python
def gradient_descent_logistic(X, y, w_in, b_in, alpha, num_iters):
    w = w_in.copy()
    b = b_in
    # YOUR CODE HERE
    return w, b
```

**Run it:**
```python
w0 = np.zeros(X_log.shape[1])
b0 = 0.
w_lg, b_lg = gradient_descent_logistic(X_log, y_log, w0, b0, alpha=0.1, num_iters=10000)
print("Learned w:", w_lg, "b:", b_lg)
```

---

## B5. `predict_logistic(X, w, b)`

**Guidance:**
- For each example: compute `f = sigmoid(np.dot(X[i], w) + b)`.
- Apply threshold: predict `1` if `f >= 0.5`, else `0`.
- Store results in an array `p = np.zeros(m)`.
- Slick one-liner for the threshold: `p[i] = f >= 0.5` (True→1.0, False→0.0).

```python
def predict_logistic(X, w, b):
    m = X.shape[0]
    p = np.zeros(m)
    # YOUR CODE HERE
    return p
```

**Check accuracy:**
```python
preds = predict_logistic(X_log, w_lg, b_lg)
print("predictions:", preds)
print("actual:     ", y_log)
print("accuracy:", np.mean(preds == y_log) * 100, "%")
```
Aim for 100% here — this data is cleanly separable.

---

# BONUS (optional, if you want the full picture)

## Z-score normalization (fixes Part A's scaling problem)

**Formula (per feature/column):**
$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

```python
def zscore_normalize(X):
    mu = np.mean(X, axis=0)      # mean per column
    sigma = np.std(X, axis=0)    # std per column
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
```

Normalize `X_lin` first, then you can use a much bigger alpha (like `0.1`) and it'll
converge fast and clean. Remember: to predict a NEW raw house later, you must
normalize it with the SAME mu and sigma before predicting.

---

# SELF-CHECK: did you internalize it?

Answer these WITHOUT looking anything up:
1. Why does `dj_db` have no `x` multiplier but `dj_dw[j]` does?
2. Why does the cost have `/2m` but the gradient has `/m`?
3. What's the ONLY difference between the linear and logistic gradient code?
4. What does sigmoid output for a large negative z, and why does that make it good
   for classification?
5. In `predict_logistic`, why 0.5 as the threshold?

If you can answer all five cleanly, you've got Course 1 locked. If any feel shaky,
that's the one to revisit.

---

# COMMON MISTAKES TO AVOID (a checklist)

- [ ] `/(2*m)` grouped with parentheses, not `/2*m` (which multiplies by m!)
- [ ] Gradient uses `/m`, cost-linear uses `/2m`, cost-logistic uses `/m`
- [ ] `+=` to accumulate, never `=` (which overwrites)
- [ ] Divisions happen OUTSIDE the loops
- [ ] `dj_dw` initialized as `np.zeros(n)`, not `0`
- [ ] sigmoid is `1/(1+np.exp(-z))` — negative z, np.exp not `**`
- [ ] logistic loss uses MINUS between the two log terms
- [ ] Recompute gradient every iteration (inside the loop), not once
- [ ] If cost goes to inf/nan → alpha too big → lower it or normalize features
```
