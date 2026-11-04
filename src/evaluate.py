"""Module d'évaluation partagé (utilisé par les deux membres).

    python -m src.evaluate     -> tableau comparatif + figures
"""
import json

import matplotlib
matplotlib.use("Agg")  # pas d'écran dans le conteneur
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src import config

NOMS_METRIQUES = ["accuracy", "precision", "recall", "f1"]


def _vers_entiers(y_pred):
    """Accepte des entiers (sklearn) ou des probabilités (Keras, seuil 0.5)."""
    y = np.asarray(y_pred).ravel()
    if y.dtype.kind == "f" and not np.isin(y, (0.0, 1.0)).all():
        y = (y >= 0.5)
    return y.astype(int)


def compute_metrics(y_true, y_pred):
    """Renvoie accuracy, precision, recall, f1 (classe positive = 1), 4 décimales."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = _vers_entiers(y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }


def save_metrics(model_name, metrics):
    """Ajoute ou remplace la clé model_name dans results/metrics.json."""
    existant = {}
    if config.METRICS_FILE.exists():
        try:
            existant = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existant = {}
    existant[model_name] = metrics
    config.METRICS_FILE.write_text(
        json.dumps(existant, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _dessiner_matrice(ax, cm, titre):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(titre)
    ax.set_xticks([0, 1], [config.LABELS[0], config.LABELS[1]])
    ax.set_yticks([0, 1], [config.LABELS[0], config.LABELS[1]])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    seuil = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > seuil else "black")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Sauvegarde results/confusion_matrix_<model_name>.png."""
    cm = confusion_matrix(np.asarray(y_true).ravel().astype(int),
                          _vers_entiers(y_pred), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    _dessiner_matrice(ax, cm, model_name)
    fig.tight_layout()
    chemin = config.RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    return chemin


def _matrice_combinee(noms):
    """Réunit les matrices individuelles dans results/confusion_matrix.png."""
    images = [(n, config.RESULTS_DIR / f"confusion_matrix_{n}.png") for n in noms]
    images = [(n, p) for n, p in images if p.exists()]
    if not images:
        return
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for ax, (nom, chemin) in zip(axes[0], images):
        ax.imshow(plt.imread(chemin))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_PNG, dpi=120)
    plt.close(fig)


def compare_models():
    """Tableau comparatif trié par F1, CSV, barres groupées, matrices réunies."""
    if not config.METRICS_FILE.exists():
        print("results/metrics.json introuvable : lancez d'abord les entraînements.")
        return None
    data = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    lignes = {nom: {m: v.get(m) for m in NOMS_METRIQUES}
              for nom, v in data.items() if isinstance(v, dict) and "f1" in v}
    if not lignes:
        print("Aucune métrique exploitable dans metrics.json.")
        return None
    df = pd.DataFrame(lignes).T.sort_values("f1", ascending=False)
    print("\nComparaison des modèles (jeu de test, triés par F1) :\n")
    print(df.to_string())
    df.to_csv(config.COMPARISON_CSV, index_label="model")

    ax = df.plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    ax.set_title("Comparaison des modèles (test)")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(config.COMPARISON_PNG, dpi=120)
    plt.close()

    _matrice_combinee(list(df.index))
    return df


if __name__ == "__main__":
    compare_models()
