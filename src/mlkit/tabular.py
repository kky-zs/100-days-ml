import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


def make_preprocessor(scale=True):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), make_column_selector(dtype_exclude=np.number)),
    ])


def cv_report(estimator, X, y, name, cv, scoring):
    res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=True)
    row = {"model": name}
    for k in scoring:
        row[f"{k} (cv)"] = f"{res['test_' + k].mean():.3f} ± {res['test_' + k].std():.3f}"
        row[f"{k} (train)"] = f"{res['train_' + k].mean():.3f}"
    row["fit time (s)"] = round(res["fit_time"].mean(), 1)
    return row


def single_feature_cv_auc(X, y, cols=None, random_state=42):
    out = {}
    for col in (cols or list(X.columns)):
        s = X[[col]]
        if pd.api.types.is_numeric_dtype(s[col]):
            prep = SimpleImputer(strategy="median")
        else:
            prep = Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
        pipe = make_pipeline(prep, DecisionTreeClassifier(max_depth=4, random_state=random_state))
        out[col] = cross_val_score(pipe, s, y, cv=3, scoring="roc_auc").mean()
    return pd.Series(out).sort_values(ascending=False)