"""
Model fitting utilities (OLS and LASSO) with cross-validation support.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score


def fit_ols_model(df_drug: pd.DataFrame) -> pd.DataFrame:
    y = df_drug["Y"].values
    X = sm.add_constant(df_drug.drop(columns=["Y"]).values)
    model = sm.OLS(y, X).fit()
    coef_names = ["Intercept"] + df_drug.columns.drop("Y").tolist()
    return pd.DataFrame({"coef": model.params, "se": model.bse}, index=coef_names)


def cross_validate_model(df_drug: pd.DataFrame, nfold: int = 5,
                         nrep: int = 10) -> np.ndarray:
    y = df_drug["Y"].values
    X = df_drug.drop(columns=["Y"]).values
    cv = RepeatedKFold(n_splits=nfold, n_repeats=nrep, random_state=123)
    mse_scores = -cross_val_score(
        LinearRegression(), X, y,
        scoring='neg_mean_squared_error', cv=cv
    )
    return mse_scores


def fit_lasso_model(df_drug: pd.DataFrame, nfold: int = 5, nrep: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df_drug["Y"].values
    X = df_drug.drop(columns=["Y"]).values

    model = LassoCV(cv=5, random_state=123, n_alphas=100, max_iter=10000)
    model.fit(X, y)

    coef = np.r_[model.intercept_, model.coef_]
    names = ["Intercept"] + df_drug.columns.drop("Y").tolist()
    coef_df = pd.DataFrame({"mutation": names, "coef": coef})

    cv = RepeatedKFold(n_splits=nfold, n_repeats=nrep, random_state=123)
    coef_samples = []
    mse_scores = []

    for train_idx, test_idx in cv.split(X):
        lasso = Lasso(alpha=model.alpha_, max_iter=10000)
        lasso.fit(X[train_idx], y[train_idx])
        coef_samples.append(np.r_[lasso.intercept_, lasso.coef_])

        preds = lasso.predict(X[test_idx])
        mse_scores.append(np.mean((y[test_idx] - preds) ** 2))

    coef_samples = np.array(coef_samples)
    if coef_samples.shape[0] > 1:
        coef_se = coef_samples.std(axis=0, ddof=1)
    else:
        coef_se = np.zeros_like(coef)

    coef_df['se'] = coef_se
    return coef_df, np.array(mse_scores)

