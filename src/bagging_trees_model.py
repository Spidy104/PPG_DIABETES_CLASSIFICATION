import warnings
warnings.filterwarnings("ignore", message=".*valid feature names.*", category=UserWarning)

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import files

# —————————————————————————————
# 1) Load & merge feature + label data
# —————————————————————————————
feat = pd.read_csv(files.PPG_MY_OWN)
meta = pd.read_csv(files.METADATA_PATH)
df = feat.merge(meta[['subject_ID','diabetes_label']], on="subject_ID", how="inner")

# feature names list
FEATURE_COLS = [c for c in df.columns if c not in ("subject_ID","diabetes_label")]

X = df[FEATURE_COLS].values
y = df["diabetes_label"].values
groups = df["subject_ID"].values

# 2) Subject‑wise Stratified CV
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# 3) Helper to restore DataFrame for LightGBM
to_df = FunctionTransformer(lambda X: pd.DataFrame(X, columns=FEATURE_COLS))

# 4) Define pipelines + hyperparameter grids
models = {
    "RandomForest": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "params": {
            "clf__n_estimators":      [100, 200, 300],
            "clf__max_depth":         [None, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf":  [1, 2, 4],
            "clf__max_features":      ["sqrt", "log2", 0.5]
        }
    },
    "SVM": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "params": {
            "clf__C":     [0.1, 1, 10],
            "clf__gamma": ["scale", "auto", 0.1]
        }
    },
    "GradientBoosting": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        "params": {
            "clf__n_estimators":   [100, 200],
            "clf__learning_rate":  [0.01, 0.1],
            "clf__max_depth":      [3, 5, 7],
            "clf__subsample":      [0.6, 0.8, 1.0]
        }
    },
    "LightGBM": {
        "pipe": Pipeline([
            ("scaler",  StandardScaler()),
            ("to_df",   to_df),  # restore names
            ("clf",     LGBMClassifier(random_state=42, verbosity=-1))
        ]),
        "params": {
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.01, 0.1],
            "clf__num_leaves":    [31, 50, 100],
            "clf__max_depth":     [-1, 5, 10]
        }
    }
}

# 5) Tune each model, save, and collect OOF preds
best_estimators = {}
y_pred_all = {}
y_proba_all = {}

for name, spec in models.items():
    print(f"\n=== Tuning {name} ===")
    search = RandomizedSearchCV(
        estimator=spec["pipe"],
        param_distributions=spec["params"],
        n_iter=20,
        scoring="accuracy",
        cv=cv.split(X, y, groups),
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X, y)
    best = search.best_estimator_
    best_estimators[name] = best
    print(f"{name} best params: {search.best_params_}")
    print(f"{name} CV acc: {search.best_score_:.4f}")

    # save tuned pipeline
    joblib.dump(best, f"{name.lower()}_model.joblib")

    # out‑of‑fold predictions & probabilities
    y_pred_all[name]  = cross_val_predict(best, X, y, cv=cv.split(X, y, groups), n_jobs=-1)
    y_proba_all[name] = cross_val_predict(
        best, X, y, method="predict_proba", cv=cv.split(X, y, groups), n_jobs=-1
    )[:,1]

# 6) Plot Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(14,12))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, y_pred_all.items()):
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No","Yes"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.suptitle("Base Learner Confusion Matrices", fontsize=18)
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()

# 7) Plot ROC Curves
plt.figure(figsize=(10,8))
for name, y_proba in y_proba_all.items():
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1], "--", color="gray")
plt.title("Base Learner ROC Curves", fontsize=16)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 8) Build Stacking & Voting Ensembles
estimators = [(n, est) for n, est in best_estimators.items()]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=cv.split(X, y, groups),
    n_jobs=-1
)
vote = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=-1
)

# 9) Evaluate Ensembles
for ens_name, ens in [("Stacking", stack), ("Voting", vote)]:
    scores = cross_val_score(
        ens, X, y,
        cv=cv,
        groups=groups,
        scoring="accuracy",
        n_jobs=-1
    )
    print(f"{ens_name} CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# 10) Save Ensembles
joblib.dump(stack, "stacking_model.joblib")
joblib.dump(vote, "voting_model.joblib")
print("\nAll models and ensembles saved as .joblib files.")
