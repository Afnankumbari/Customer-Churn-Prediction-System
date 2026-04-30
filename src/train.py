"""
train.py
--------
Trains multiple ML models, compares them, saves the best as pickle.
Models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocess import build_preprocessor, prepare_xy, load_data, engineer_features

BG = "#0F1117"
FG = "#EAEAEA"
GRID_C = "#2A2D3E"
ACCENT = "#E63946"
GREEN = "#2A9D8F"


def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      class_weight="balanced", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                           max_depth=4, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, class_weight="balanced",
                                   random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=3, random_state=42, verbosity=0
        )
    return models


def train_and_evaluate(csv_path: str, model_out: str = "models/best_model.pkl",
                       assets_dir: str = "assets") -> dict:
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # ── Load & prepare ────────────────────────────────────────────────────────
    df = load_data(csv_path)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Churn rate: {y.mean():.1%}")

    preprocessor = build_preprocessor()
    models = get_models()
    results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        # CV AUC
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1).mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "pipeline":   pipe,
            "cv_auc":     round(cv_auc, 4),
            "test_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "accuracy":   round(accuracy_score(y_test, y_pred), 4),
            "f1":         round(f1_score(y_test, y_pred), 4),
            "y_pred":     y_pred,
            "y_prob":     y_prob,
        }
        print(f"  [{name}] CV-AUC={cv_auc:.4f} | Test-AUC={results[name]['test_auc']:.4f} "
              f"| Acc={results[name]['accuracy']:.4f} | F1={results[name]['f1']:.4f}")

    # ── Pick best model (test AUC) ────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best_result = results[best_name]
    print(f"\n🏆 Best model: {best_name} (AUC={best_result['test_auc']})")

    # ── Save best pipeline ────────────────────────────────────────────────────
    with open(model_out, "wb") as f:
        pickle.dump({"model": best_result["pipeline"],
                     "name":  best_name,
                     "auc":   best_result["test_auc"]}, f)
    print(f"💾 Model saved → {model_out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_model_comparison(results, assets_dir)
    _plot_roc_curves(results, X_test, y_test, assets_dir)
    _plot_confusion_matrix(best_result, best_name, y_test, assets_dir)
    _plot_feature_importance(best_result["pipeline"], best_name, assets_dir)

    return results


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": GRID_C, "axes.labelcolor": FG,
        "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": GRID_C,
    })


def _plot_model_comparison(results: dict, out_dir: str):
    _style()
    names = list(results.keys())
    metrics = ["cv_auc", "test_auc", "accuracy", "f1"]
    labels  = ["CV AUC", "Test AUC", "Accuracy", "F1 Score"]
    colors  = ["#E63946", "#2A9D8F", "#F4A261", "#457B9D"]

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor=BG)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color=FG)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc_curves(results, X_test, y_test, out_dir):
    _style()
    colors = ["#E63946", "#2A9D8F", "#F4A261", "#457B9D", "#A8DADC"]
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for (name, res), color in zip(results.items(), colors):
        RocCurveDisplay.from_predictions(
            y_test, res["y_prob"], name=f"{name} (AUC={res['test_auc']})",
            ax=ax, color=color, lw=2
        )
    ax.plot([0, 1], [0, 1], "w--", lw=1, label="Random")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "08_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(result, name, y_test, out_dir):
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    cm = confusion_matrix(y_test, result["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="RdYlGn_r")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold", color=FG)
    ax.tick_params(colors=FG)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "09_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(pipe, name, out_dir):
    """Works for tree-based models; falls back gracefully."""
    _style()
    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]

    if not hasattr(clf, "feature_importances_"):
        return

    try:
        feat_names = (
            pre.transformers_[0][2]  # numeric
            + list(pre.transformers_[1][1]
                   .named_steps["encoder"]
                   .get_feature_names_out(pre.transformers_[1][2]))
        )
        importances = clf.feature_importances_
        top_n = 15
        idx = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.85, top_n))
        ax.barh([feat_names[i] for i in idx], importances[idx],
                color=colors_bar, edgecolor=BG)
        ax.set_title(f"Top {top_n} Feature Importances — {name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "10_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  (Feature importance plot skipped: {e})")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/customers.csv"
    train_and_evaluate(csv, model_out="models/best_model.pkl", assets_dir="assets")
