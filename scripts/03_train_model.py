"""
scripts/03_train_model.py  —  v2.0
Entrena dos modelos:
  1. LightGBM de transición urbana (reemplaza Random Forest)
  2. LightGBM de reglas CA (aprende cuándo una celda se convierte)

Contribuciones originales vs López-Rivera (2021):
  - Clasificador: gradient boosting (LightGBM) en lugar de RNA
  - Features kársticas: LST, distancia a cenotes, vulnerabilidad acuífero
  - CA de reglas aprendidas en lugar de umbral estadístico fijo
"""
import os, sys, logging, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, RocCurveDisplay

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import LGBM_CONFIG, PATHS, FEATURE_NAMES, CA_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    log.warning("LightGBM no instalado. Usando RandomForest como fallback.")
    from sklearn.ensemble import RandomForestClassifier
    LGB_AVAILABLE = False


def figure_of_merit(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    return tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0.0


def build_lgbm(n_pos, n_neg):
    """Construye el clasificador LightGBM con manejo explícito de desbalance."""
    scale = n_neg / n_pos if n_pos > 0 else 1.0
    if LGB_AVAILABLE:
        return lgb.LGBMClassifier(
            n_estimators=LGBM_CONFIG["n_estimators"],
            learning_rate=LGBM_CONFIG["learning_rate"],
            num_leaves=LGBM_CONFIG["num_leaves"],
            max_depth=LGBM_CONFIG["max_depth"],
            min_child_samples=LGBM_CONFIG["min_child_samples"],
            subsample=LGBM_CONFIG["subsample"],
            colsample_bytree=LGBM_CONFIG["colsample_bytree"],
            scale_pos_weight=scale,
            n_jobs=LGBM_CONFIG["n_jobs"],
            random_state=LGBM_CONFIG["random_state"],
            verbose=-1,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, max_depth=20, class_weight="balanced",
            n_jobs=-1, random_state=42, oob_score=True
        )


def load_training_data():
    path = os.path.join(PATHS["processed"], "training_dataset.parquet")
    if not os.path.exists(path):
        log.error(f"Dataset no encontrado: {path}")
        log.error("Ejecuta primero: python scripts/02_preprocess.py")
        sys.exit(1)
    df = pd.read_parquet(path)
    available = [f for f in FEATURE_NAMES if f in df.columns]
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        log.warning(f"Features kársticas no disponibles (datos sintéticos): {missing}")
        log.warning("  Con datos LANDSAT reales estarán incluidas.")
    X = df[available].values
    y = df["urban"].values
    mask = ~np.any(np.isnan(X), axis=1)
    X, y = X[mask], y[mask]
    log.info(f"Dataset: {len(y):,} muestras | {y.mean()*100:.1f}% positivos | {X.shape[1]} features")
    return X, y, available


def train_transition_model(X, y, feature_names):
    """Modelo 1: transición urbana (celda no-urbana → urbana)."""
    log.info("=" * 58)
    log.info("MODELO 1: Transición urbana — LightGBM")
    log.info("=" * 58)
    n_pos, n_neg = y.sum(), (1-y).sum()
    log.info(f"  Positivos: {n_pos:,} | Negativos: {n_neg:,} | Ratio: {n_neg/n_pos:.1f}:1")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=LGBM_CONFIG["test_size"], stratify=y,
        random_state=LGBM_CONFIG["random_state"]
    )

    # CV rápida con 100 estimadores
    model_cv = build_lgbm(y_tr.sum(), (1-y_tr).sum())
    if LGB_AVAILABLE:
        model_cv.set_params(n_estimators=100)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_cv, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info(f"  CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Modelo final
    model = build_lgbm(y_tr.sum(), (1-y_tr).sum())
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "model":        "LightGBM_transition",
        "auc_roc":      round(roc_auc_score(y_te, y_prob), 4),
        "f1_score":     round(f1_score(y_te, y_pred), 4),
        "kappa":        round(cohen_kappa_score(y_te, y_pred), 4),
        "fom":          round(figure_of_merit(y_te, y_pred), 4),
        "cv_auc_mean":  round(cv_scores.mean(), 4),
        "cv_auc_std":   round(cv_scores.std(), 4),
        "n_features":   X.shape[1],
    }
    for k,v in metrics.items():
        if k not in ("model","n_features"): log.info(f"  {k}: {v}")

    _plot_importance(model, feature_names, "lgbm_feature_importance.png")
    _plot_roc(model, X_te, y_te, "LightGBM — Transición Urbana", "lgbm_roc.png")
    joblib.dump(model, PATHS["lgbm_model"])
    return model, metrics


def build_ca_training_data(model, X, y):
    """
    Construye el dataset para el modelo de reglas CA.
    X_ca = estado de vecindad + P_LightGBM + features kársticas locales
    y_ca = ¿se convirtió la celda? (mismo y)
    """
    log.info("\nConstruyendo dataset para reglas CA aprendidas...")
    p_lgbm = model.predict_proba(X)[:,1]
    # Agregar P_LightGBM como feature adicional
    X_ca = np.column_stack([X, p_lgbm])
    log.info(f"  Dataset CA: {len(y):,} muestras | {X_ca.shape[1]} features (incluyendo P_LGB)")
    return X_ca, y


def train_ca_rules_model(X_ca, y):
    """Modelo 2: reglas CA aprendidas."""
    log.info("\n" + "=" * 58)
    log.info("MODELO 2: Reglas CA aprendidas — LightGBM")
    log.info("=" * 58)
    n_pos, n_neg = y.sum(), (1-y).sum()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ca, y, test_size=0.25, stratify=y, random_state=42
    )
    model_ca = build_lgbm(y_tr.sum(), (1-y_tr).sum())
    if LGB_AVAILABLE:
        model_ca.set_params(n_estimators=200, learning_rate=0.08)
    model_ca.fit(X_tr, y_tr)

    y_prob = model_ca.predict_proba(X_te)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    ca_metrics = {
        "model":    "LightGBM_CA_rules",
        "auc_roc":  round(roc_auc_score(y_te, y_prob), 4),
        "fom":      round(figure_of_merit(y_te, y_pred), 4),
    }
    log.info(f"  AUC-ROC: {ca_metrics['auc_roc']:.4f} | FOM: {ca_metrics['fom']:.4f}")
    joblib.dump(model_ca, PATHS["ca_model"])
    return model_ca, ca_metrics


def _plot_importance(model, feature_names, fname):
    try:
        imp = model.feature_importances_
        order = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(7,4))
        colors = ["#7C4DFF" if v>0.15 else "#5C6BC0" if v>0.10 else "#90A4AE" for v in imp[order]]
        ax.barh([feature_names[i] for i in order], imp[order], color=colors, edgecolor="white")
        ax.axvline(1/len(feature_names), color="gray", linestyle="--", alpha=0.5, label="Importancia uniforme")
        ax.set_title("Importancia de Variables — LightGBM v2.0", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importancia (split gain)")
        ax.legend(fontsize=9)
        # Destacar features kársticas
        yticks = ax.get_yticklabels()
        for tick in yticks:
            if any(k in tick.get_text() for k in ["lst","cenote","karst"]):
                tick.set_color("#7C4DFF")
                tick.set_fontweight("bold")
        plt.tight_layout()
        fig.savefig(os.path.join(PATHS["results_reports"], fname), dpi=150)
        plt.close()
        log.info(f"  → {fname}")
    except Exception as e:
        log.warning(f"No se pudo graficar importancia: {e}")


def _plot_roc(model, X_te, y_te, title, fname):
    try:
        fig, ax = plt.subplots(figsize=(5,5))
        RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, color="#7C4DFF", name="LightGBM")
        ax.plot([0,1],[0,1],"k--",alpha=0.4)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(PATHS["results_reports"], fname), dpi=150)
        plt.close()
        log.info(f"  → {fname}")
    except Exception as e:
        log.warning(f"No se pudo graficar ROC: {e}")


def main():
    log.info("=" * 58)
    log.info("PASO 3 v2.0: Entrenamiento LightGBM + CA reglas aprendidas")
    log.info("=" * 58)

    X, y, available_features = load_training_data()
    model1, m1 = train_transition_model(X, y, available_features)
    X_ca, y_ca = build_ca_training_data(model1, X, y)
    model2, m2 = train_ca_rules_model(X_ca, y_ca)

    all_metrics = pd.DataFrame([{**m1, **{f"ca_{k}":v for k,v in m2.items() if k!="model"}}])
    all_metrics.to_csv(os.path.join(PATHS["results_reports"], "metrics.csv"), index=False)

    log.info("\n" + "=" * 58)
    log.info("RESUMEN v2.0")
    log.info(f"  LightGBM AUC-ROC:  {m1['auc_roc']}")
    log.info(f"  LightGBM FOM:      {m1['fom']}")
    log.info(f"  CA-rules AUC-ROC:  {m2['auc_roc']}")
    log.info(f"  Modelos guardados: {PATHS['lgbm_model']}")
    log.info(f"                     {PATHS['ca_model']}")
    log.info("=" * 58)


if __name__ == "__main__":
    main()
