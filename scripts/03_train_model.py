"""
scripts/03_train_model.py
Entrena el modelo Random Forest para predecir transición urbana.

Métricas de evaluación:
  - AUC-ROC
  - F1 Score
  - Figure of Merit (FOM) — estándar para modelos LULC
  - Kappa de Cohen

Salidas:
  - models/rf_model.pkl              → modelo serializado
  - results/reports/metrics.csv      → métricas de evaluación
  - results/reports/feature_importance.csv
  - results/reports/roc_curve.png
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # no requiere display
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, cohen_kappa_score,
    classification_report, RocCurveDisplay
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RF_CONFIG, PATHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "dist_urban_edge", "dist_center", "dist_road",
    "ndvi_mean", "neighbor_3x3", "neighbor_5x5", "neighbor_9x9"
]


# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def load_training_data():
    """Carga el dataset de entrenamiento generado en el paso anterior."""
    dataset_path = os.path.join(PATHS["processed"], "training_dataset.parquet")

    if not os.path.exists(dataset_path):
        log.error(f"Dataset no encontrado: {dataset_path}")
        log.error("Ejecuta primero: python scripts/02_preprocess.py")
        sys.exit(1)

    log.info(f"Cargando dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    X = df[FEATURE_NAMES].values
    y = df["urban"].values

    log.info(f"  {len(df):,} muestras | "
             f"{y.mean()*100:.1f}% positivos (urbanizados) | "
             f"{X.shape[1]} features")

    # Eliminar filas con NaN (puede haber en bordes del raster)
    mask = ~np.any(np.isnan(X), axis=1)
    X, y = X[mask], y[mask]
    log.info(f"  {mask.sum():,} muestras válidas después de eliminar NaN")

    return X, y


# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Entrena el clasificador Random Forest con los parámetros de config."""
    log.info("Entrenando Random Forest...")
    log.info(f"  n_estimators={RF_CONFIG['n_estimators']}, "
             f"max_depth={RF_CONFIG['max_depth']}, "
             f"class_weight={RF_CONFIG['class_weight']}")

    model = RandomForestClassifier(
        n_estimators=RF_CONFIG["n_estimators"],
        max_depth=RF_CONFIG["max_depth"],
        min_samples_leaf=RF_CONFIG["min_samples_leaf"],
        class_weight=RF_CONFIG["class_weight"],
        n_jobs=RF_CONFIG["n_jobs"],
        random_state=RF_CONFIG["random_state"],
        oob_score=True,   # estimación out-of-bag gratis
    )

    model.fit(X_train, y_train)
    log.info(f"  ✓ Entrenamiento completado | OOB score: {model.oob_score_:.4f}")
    return model


# ─────────────────────────────────────────────────────────────
# VALIDACIÓN Y MÉTRICAS
# ─────────────────────────────────────────────────────────────

def figure_of_merit(y_true, y_pred) -> float:
    """
    Figure of Merit (FOM) — métrica estándar para modelos de cambio LULC.
    Ponencia de Pontius et al. (2008).

    FOM = |correcto_urbanizado| / |union(observado, simulado)|
    """
    # Verdaderos positivos, falsos positivos, falsos negativos
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    fom = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return fom


def evaluate_model(model, X_test, y_test, X_train, y_train) -> dict:
    """Calcula métricas completas de evaluación."""
    log.info("Evaluando modelo...")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc":      round(roc_auc_score(y_test, y_pred_proba), 4),
        "f1_score":     round(f1_score(y_test, y_pred), 4),
        "kappa":        round(cohen_kappa_score(y_test, y_pred), 4),
        "fom":          round(figure_of_merit(y_test, y_pred), 4),
        "oob_score":    round(model.oob_score_, 4),
        "n_train":      len(y_train),
        "n_test":       len(y_test),
        "pct_positive": round(y_test.mean() * 100, 2),
    }

    log.info(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    log.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    log.info(f"  Kappa:    {metrics['kappa']:.4f}")
    log.info(f"  FOM:      {metrics['fom']:.4f}")
    log.info(f"  OOB:      {metrics['oob_score']:.4f}")

    log.info("\n" + classification_report(y_test, y_pred, target_names=["no-urbano", "urbano"]))

    return metrics, y_pred_proba


def cross_validate_model(X, y) -> dict:
    """Validación cruzada 5-fold para estimación robusta del rendimiento."""
    log.info("Validación cruzada (5-fold)...")

    model_cv = RandomForestClassifier(
        n_estimators=100,   # menos árboles para CV más rápida
        max_depth=RF_CONFIG["max_depth"],
        class_weight=RF_CONFIG["class_weight"],
        n_jobs=RF_CONFIG["n_jobs"],
        random_state=RF_CONFIG["random_state"],
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model_cv, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    log.info(f"  CV AUC-ROC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    return {"cv_auc_mean": round(auc_scores.mean(), 4), "cv_auc_std": round(auc_scores.std(), 4)}


# ─────────────────────────────────────────────────────────────
# IMPORTANCIA DE VARIABLES
# ─────────────────────────────────────────────────────────────

def save_feature_importance(model) -> pd.DataFrame:
    """Guarda y visualiza la importancia de cada feature."""
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    df_imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances,
        "std": std,
    }).sort_values("importance", ascending=False)

    out_path = os.path.join(PATHS["results_reports"], "feature_importance.csv")
    df_imp.to_csv(out_path, index=False)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E53935" if imp > 0.15 else "#FF7043" if imp > 0.10 else "#78909C"
              for imp in df_imp["importance"]]
    ax.barh(df_imp["feature"], df_imp["importance"], xerr=df_imp["std"],
            color=colors, capsize=4, edgecolor="white")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de Variables — Random Forest", fontsize=13, fontweight="bold")
    ax.axvline(x=1/len(FEATURE_NAMES), color="gray", linestyle="--", alpha=0.5, label="Importancia uniforme")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PATHS["results_reports"], "feature_importance.png"), dpi=150)
    plt.close()

    log.info(f"  ✓ Importancia de features guardada: {out_path}")
    return df_imp


# ─────────────────────────────────────────────────────────────
# CURVA ROC
# ─────────────────────────────────────────────────────────────

def save_roc_curve(model, X_test, y_test):
    """Genera y guarda la curva ROC."""
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                    color="#E53935", name="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Curva ROC — Predicción de Expansión Urbana", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(PATHS["results_reports"], "roc_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  ✓ Curva ROC guardada: {out_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PASO 3: Entrenamiento del modelo Random Forest")
    log.info("=" * 60)

    # Cargar datos
    X, y = load_training_data()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=RF_CONFIG["test_size"],
        stratify=y,
        random_state=RF_CONFIG["random_state"]
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Validación cruzada (estimación robusta)
    cv_metrics = cross_validate_model(X, y)

    # Entrenamiento final sobre todos los datos de entrenamiento
    model = train_random_forest(X_train, y_train)

    # Evaluación
    metrics, y_pred_proba = evaluate_model(model, X_test, y_test, X_train, y_train)
    metrics.update(cv_metrics)

    # Guardar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(PATHS["results_reports"], "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"  ✓ Métricas guardadas: {metrics_path}")

    # Importancia de features y curva ROC
    save_feature_importance(model)
    save_roc_curve(model, X_test, y_test)

    # Guardar modelo
    model_path = PATHS["rf_model"]
    joblib.dump(model, model_path)
    log.info(f"\n✓ Modelo guardado: {model_path}")

    # Resumen
    log.info("\n" + "=" * 60)
    log.info("RESUMEN DEL MODELO:")
    log.info(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    log.info(f"  FOM:      {metrics['fom']:.4f}")
    log.info(f"  Kappa:    {metrics['kappa']:.4f}")
    log.info(f"  CV AUC:   {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    log.info("=" * 60)

    if metrics["auc_roc"] < 0.75:
        log.warning("AUC < 0.75: considera revisar la clasificación LULC o agregar más features.")


if __name__ == "__main__":
    main()
