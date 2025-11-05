import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("src/train_post_EDA.csv")

def regressor(feature):
    print("Using Feature:", feature)

    model = LinearRegression()
    kfold = KFold(n_splits=5, shuffle=True, random_state=25)

    X = df[[feature]]
    y = df['Emergency Visits / 100000 temp residual']

    rmse_scores, r2_scores = [], []
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))

    print("RMSE per fold:", rmse_scores)
    print("Mean RMSE:", np.mean(rmse_scores))
    print("R² per fold:", r2_scores)
    print("Mean R²:", np.mean(r2_scores))

    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred, color='black', linewidth=2, label='Regression Line')

    plt.xlabel(feature)
    plt.ylabel("Emergency Visits per 100000 (Adjusted)")
    plt.title(f"Linear Regression: {feature} vs Emergency Visits")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Regression Equation: y = {:.3f}x + {:.3f}".format(model.coef_[0], model.intercept_))
    print()


regressor("Energy Burden % of Income")
regressor("Park within 1/2 Mile")
regressor("Imperviousness")
regressor("% w/o Internet")
