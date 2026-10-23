"""
EDA and Visualization Case Study: Breast Cancer Wisconsin Diagnostic Dataset
Reproducible undergraduate project. Dataset is bundled with scikit-learn.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
from sklearn.inspection import permutation_importance
import joblib

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
MOD = ROOT / "models"
for p in [FIG, RES, MOD]:
    p.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings("ignore", category=UserWarning)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close()

def main():
    # 1. Load the real-world dataset included with scikit-learn.
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    df.drop(columns=["target"], inplace=True)
    feature_cols = list(bunch.feature_names)
    X = df[feature_cols].copy()
    y = bunch.target.to_numpy()  # 0 malignant, 1 benign

    overview = {
        "rows": int(df.shape[0]), "columns_including_target": int(df.shape[1]),
        "predictor_features": len(feature_cols), "feature_names": feature_cols,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in df["diagnosis"].value_counts().items()},
        "class_mapping": {"0": "malignant", "1": "benign"},
        "dataset_source": "Wisconsin Breast Cancer Diagnostic dataset, bundled in scikit-learn; originally from UCI ML Repository"
    }
    (RES / "dataset_overview.json").write_text(json.dumps(overview, indent=2))
    df.describe().T.to_csv(RES / "descriptive_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(RES / "missing_values.csv")
    df.dtypes.astype(str).rename("dtype").to_csv(RES / "data_types.csv")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)
        X = df[feature_cols].copy()
        y = df["diagnosis"].map({"malignant": 0, "benign": 1}).to_numpy()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="mean radius", hue="diagnosis", bins=28, kde=True, element="step", stat="density", common_norm=False)
    plt.title("Figure 1. Distribution of Mean Radius by Diagnosis")
    plt.xlabel("Mean radius"); plt.ylabel("Density")
    savefig("fig01_mean_radius_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="diagnosis", y="mean area")
    plt.title("Figure 2. Mean Area by Diagnosis")
    plt.xlabel("Diagnosis"); plt.ylabel("Mean area")
    savefig("fig02_mean_area_boxplot.png")

    plt.figure(figsize=(7, 5))
    df["diagnosis"].value_counts().reindex(["malignant", "benign"]).plot(kind="bar")
    plt.title("Figure 3. Class Distribution")
    plt.xlabel("Diagnosis"); plt.ylabel("Number of samples")
    savefig("fig03_class_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="mean radius", y="mean perimeter", hue="diagnosis", alpha=.75)
    plt.title("Figure 4. Mean Radius vs Mean Perimeter")
    plt.xlabel("Mean radius"); plt.ylabel("Mean perimeter")
    savefig("fig04_radius_perimeter_scatter.png")

    plt.figure(figsize=(13, 10))
    sns.heatmap(df[feature_cols].corr(), cmap="coolwarm", center=0, xticklabels=False, yticklabels=False, cbar_kws={"label":"Pearson correlation"})
    plt.title("Figure 5. Correlation Heatmap of 30 Numeric Features")
    savefig("fig05_correlation_heatmap.png")

    selected = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "diagnosis"]
    pair = sns.pairplot(df[selected], hue="diagnosis", corner=True, diag_kind="hist", plot_kws={"alpha":.55, "s":18})
    pair.fig.suptitle("Figure 6. Pairplot of Selected Features", y=1.02)
    pair.savefig(FIG / "fig06_selected_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    outlier_flags = (X.lt(q1 - 1.5*iqr) | X.gt(q3 + 1.5*iqr))
    outlier_summary = outlier_flags.sum().sort_values(ascending=False)
    outlier_summary.rename("iqr_outlier_count").to_csv(RES / "outlier_summary.csv")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, random_state=SEED, stratify=y_temp)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    configs = [
        {"hidden_layer_sizes": (32,), "alpha": .0001, "learning_rate_init": .001},
        {"hidden_layer_sizes": (64, 32), "alpha": .0001, "learning_rate_init": .001},
        {"hidden_layer_sizes": (32,), "alpha": .001, "learning_rate_init": .005},
    ]
    tuning, best, best_score, best_history = [], None, -1, None
    for ci, cfg in enumerate(configs):
        model = MLPClassifier(hidden_layer_sizes=cfg["hidden_layer_sizes"], activation="relu", solver="adam",
            alpha=cfg["alpha"], learning_rate_init=cfg["learning_rate_init"], batch_size=32,
            max_iter=1, warm_start=True, random_state=SEED, shuffle=True)
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        for epoch in range(1, 101):
            model.fit(X_train_s, y_train)
            train_losses.append(float(model.loss_))
            probs_val = np.clip(model.predict_proba(X_val_s), 1e-9, 1-1e-9)
            val_losses.append(float(-np.mean(np.log(probs_val[np.arange(len(y_val)), y_val]))))
            train_accs.append(float(accuracy_score(y_train, model.predict(X_train_s))))
            val_accs.append(float(accuracy_score(y_val, model.predict(X_val_s))))
        pred_val = model.predict(X_val_s)
        score = f1_score(y_val, pred_val)
        tuning.append({"configuration": ci+1, **{k: list(v) if isinstance(v, tuple) else v for k,v in cfg.items()},
                       "validation_accuracy": float(accuracy_score(y_val, pred_val)), "validation_f1": float(score), "epochs": 100})
        if score > best_score:
            best_score, best = score, model
            best_history = {"train_loss": train_losses, "val_loss": val_losses, "train_accuracy": train_accs,
                            "val_accuracy": val_accs, "config": cfg, "config_number": ci+1}
    pd.DataFrame(tuning).to_csv(RES / "hyperparameter_comparison.csv", index=False)
    (RES / "training_history.json").write_text(json.dumps(best_history, indent=2))

    test_probs = best.predict_proba(X_test_s)[:, 1]
    test_pred = best.predict(X_test_s)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision_benign": float(precision_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "test_recall_benign": float(recall_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "test_f1_benign": float(f1_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "test_roc_auc_benign": float(roc_auc_score(y_test, test_probs)), "test_samples": int(len(y_test)),
        "selected_configuration": int(best_history["config_number"]),
        "selected_hyperparameters": {k: (list(v) if isinstance(v, tuple) else v) for k,v in best_history["config"].items()},
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)}, "random_seed": SEED
    }
    (RES / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RES / "classification_report.txt").write_text(classification_report(y_test, test_pred, target_names=["malignant","benign"]))
    joblib.dump({"model": best, "scaler": scaler, "feature_names": feature_cols}, MOD / "mlp_breast_cancer.joblib")

    h = best_history
    plt.figure(figsize=(8,5)); plt.plot(h["train_loss"], label="Training loss"); plt.plot(h["val_loss"], label="Validation log loss")
    plt.title("Figure 7. Training and Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); savefig("fig07_loss_curves.png")
    plt.figure(figsize=(8,5)); plt.plot(h["train_accuracy"], label="Training accuracy"); plt.plot(h["val_accuracy"], label="Validation accuracy")
    plt.title("Figure 8. Training and Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(.5,1.02); plt.legend(); savefig("fig08_accuracy_curves.png")
    cm = confusion_matrix(y_test, test_pred, labels=[0,1])
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred. malignant","Pred. benign"], yticklabels=["Actual malignant","Actual benign"])
    plt.title("Figure 9. Test-Set Confusion Matrix"); plt.xlabel("Predicted class"); plt.ylabel("Actual class"); savefig("fig09_confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, test_probs, pos_label=1); auc = roc_auc_score(y_test, test_probs)
    plt.figure(figsize=(7,5)); plt.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})"); plt.plot([0,1],[0,1], linestyle="--", label="Chance baseline")
    plt.title("Figure 10. ROC Curve on Held-Out Test Set"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(loc="lower right"); savefig("fig10_roc_curve.png")
    perm = permutation_importance(best, X_test_s, y_test, n_repeats=10, random_state=SEED, scoring="f1")
    imp = pd.DataFrame({"feature": feature_cols, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(RES / "permutation_feature_importance.csv", index=False)
    plt.figure(figsize=(9,6)); imp.head(12).sort_values("importance_mean").set_index("feature")["importance_mean"].plot(kind="barh")
    plt.title("Figure 11. Top 12 Permutation Feature Importances"); plt.xlabel("Mean decrease in F1 when shuffled"); plt.ylabel("Feature"); savefig("fig11_feature_importance.png")
    summary = {
        "dataset_rows": len(df), "feature_count": len(feature_cols), "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_count, "iqr_features_with_outliers": int((outlier_summary > 0).sum()),
        "iqr_total_flagged_cells": int(outlier_summary.sum()),
        "class_counts": {str(k): int(v) for k,v in df["diagnosis"].value_counts().items()}, "metrics": metrics,
        "top_features": imp.head(5).to_dict(orient="records"),
        "correlation_radius_perimeter": float(df["mean radius"].corr(df["mean perimeter"])),
        "mean_radius_by_class": {k: float(v) for k,v in df.groupby("diagnosis")["mean radius"].mean().items()},
        "mean_area_by_class": {k: float(v) for k,v in df.groupby("diagnosis")["mean area"].mean().items()},
        "figures": sorted(p.name for p in FIG.glob("*.png"))
    }
    (RES / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
