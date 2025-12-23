import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement

N = 50
dimension = 2
degree = 2
train_ratio = 0.8
noise_std = 0.5
lambda_val = 10

np.random.seed(50)

# random data
X_df = pd.DataFrame(
    np.random.uniform(-1,1, size=(N, dimension)),

    columns=[f"x{i+1}" for i in range(dimension)]
)

# Design matrix 
def design_matrix_full(X, degree):
    N, D = X.shape
    Phi = [np.ones((N, 1))]

    for d in range(1, degree + 1):
        for indices in combinations_with_replacement(range(D), d):
            term = np.ones((N, 1))
            for i in indices:
                term *= X[:, i:i+1]
            Phi.append(term)

    return np.hstack(Phi)

# Generate target values
Phi_full = design_matrix_full(X_df.values, degree)
true_w = np.random.randn(Phi_full.shape[1], 1)
t = Phi_full @ true_w + noise_std * np.random.randn(N, 1)

df = X_df.copy()
df["t"] = t

# Train-Test Split
train_df = df.sample(frac=train_ratio, random_state=1)
test_df = df.drop(train_df.index)

X_train = train_df.iloc[:, :-1].values
t_train = train_df.iloc[:, -1].values.reshape(-1, 1)

X_test = test_df.iloc[:, :-1].values
t_test = test_df.iloc[:, -1].values.reshape(-1, 1)

# Training using Ridge Regression
Phi_train = design_matrix_full(X_train, degree)
I = np.eye(Phi_train.shape[1])
w_ridge = np.linalg.inv(Phi_train.T @ Phi_train + lambda_val * I) @ (Phi_train.T @ t_train)

y_train = Phi_train @ w_ridge
y_test = design_matrix_full(X_test, degree) @ w_ridge

# Error
mse_train = 0.5 * np.mean((y_train - t_train) ** 2)
mse_test = 0.5 * np.mean((y_test - t_test) ** 2)

print("Dimension:", dimension)
print("Degree:", degree)
print("Regularization λ:", lambda_val)
print("Number of weights:", Phi_train.shape[1])
print("Weights:", w_ridge)
print("Training MSE:", mse_train)
print("Testing MSE:", mse_test)

# plotting
if dimension == 2:
    x1_range = np.linspace(df["x1"].min(), df["x1"].max(), 60)
    x2_range = np.linspace(df["x2"].min(), df["x2"].max(), 60)

    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))

    Phi_grid = design_matrix_full(X_grid, degree)
    Y_grid = (Phi_grid @ w_ridge).reshape(X1_grid.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X1_grid, X2_grid, Y_grid,
        alpha=0.5
    )

    ax.scatter(
        X_train[:, 0], X_train[:, 1], t_train.ravel(),
        color="blue", label="Train"
    )

    ax.scatter(
        X_test[:, 0], X_test[:, 1], t_test.ravel(),
        color="red", label="Test"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("t")
    ax.set_title(f"2D Ridge Polynomial Regression (Degree={degree})")
    ax.legend()

    plt.show()
else:
    print("Visualization supported only for 2D input")