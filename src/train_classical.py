"""Entraînement des 3 modèles classiques : Naive Bayes, Régression logistique, SVM.

Règle d'or : l'hyperparamètre de chaque modèle est choisi UNIQUEMENT sur la
validation. Le jeu de test n'est lu qu'une fois, à la fin, pour rapporter
les métriques finales (aucun choix ne dépend de lui).

Sorties :
    models/naive_bayes.pkl, logistic_regression.pkl, svm.pkl
    models/classical_best.pkl      (meilleur F1 en validation)
    results/classical_best.json    (nom du meilleur + hyperparamètres retenus)
    results/metrics.json           (métriques test, via src.evaluate)
    results/top_features.json      (bonus : mots les plus discriminants)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics

# Le tokenizer par défaut de scikit-learn ignore les mots d'une lettre (le "n"
# de "n est pas") et la ponctuation (! et ?), pourtant conservés par clean_text.
# Ce motif les garde ; il marche aussi si clean_text laisse "n'" avec son apostrophe.
TOKEN_PATTERN = r"(?u)\b\w+'?|[!?]+"

# Réglages TF-IDF communs aux trois modèles
TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2, max_df=0.9,
                    sublinear_tf=True, max_features=50000,
                    token_pattern=TOKEN_PATTERN)

# Un seul hyperparamètre par modèle, 3 valeurs testées sur la validation
MODELES = {
    "naive_bayes": {"hyperparam": "alpha", "valeurs": [0.1, 0.5, 1.0],
                    "fichier": config.NAIVE_BAYES_PATH},
    "logistic_regression": {"hyperparam": "C", "valeurs": [1.0, 5.0, 20.0],
                            "fichier": config.LOGISTIC_REGRESSION_PATH},
    "svm": {"hyperparam": "C", "valeurs": [0.1, 0.5, 1.0],
            "fichier": config.SVM_PATH},
}


def charger(chemin):
    """Renvoie (textes nettoyés, étiquettes) depuis un CSV du projet."""
    df = pd.read_csv(chemin)
    return df["clean_review"].fillna("").tolist(), df["label"].astype(int).to_numpy()


def construire(nom, valeur):
    """Pipeline TF-IDF + classifieur pour une valeur d'hyperparamètre."""
    if nom == "naive_bayes":
        clf = MultinomialNB(alpha=valeur)
    elif nom == "logistic_regression":
        clf = LogisticRegression(C=valeur, max_iter=1000, random_state=config.SEED)
    elif nom == "svm":
        # LinearSVC n'a pas de predict_proba : on le calibre
        clf = CalibratedClassifierCV(LinearSVC(C=valeur, random_state=config.SEED), cv=3)
    else:
        raise ValueError(f"Modèle inconnu : {nom}")
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("clf", clf)])


def regler(nom, X_train, y_train, X_val, y_val):
    """Teste les valeurs sur la VALIDATION et garde le meilleur F1.

    Renvoie (pipeline, valeur retenue, F1 validation, temps d'entraînement).
    """
    info = MODELES[nom]
    meilleur = None
    for valeur in info["valeurs"]:
        t0 = time.time()
        pipe = construire(nom, valeur)
        pipe.fit(X_train, y_train)
        duree = round(time.time() - t0, 2)
        f1_val = compute_metrics(y_val, pipe.predict(X_val))["f1"]
        print(f"  {nom:<20} {info['hyperparam']}={valeur:<5} -> F1 validation = {f1_val:.4f}")
        if meilleur is None or f1_val > meilleur[2]:
            meilleur = (pipe, valeur, f1_val, duree)
    return meilleur


def sauver_top_features(pipe, n=20):
    """Bonus : les n mots/bigrammes les plus positifs et négatifs (régression logistique)."""
    mots = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_[0]
    ordre = np.argsort(coefs)
    resultat = {
        "positifs": [{"mot": mots[i], "poids": round(float(coefs[i]), 4)}
                     for i in ordre[::-1][:n]],
        "negatifs": [{"mot": mots[i], "poids": round(float(coefs[i]), 4)}
                     for i in ordre[:n]],
    }
    config.TOP_FEATURES_PATH.write_text(
        json.dumps(resultat, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    print("Chargement : train et validation (le test reste fermé jusqu'à la fin)")
    X_train, y_train = charger(config.TRAIN_CSV)
    X_val, y_val = charger(config.VALIDATION_CSV)
    print(f"  train={len(X_train)}  validation={len(X_val)}")

    pipes, choix = {}, {}
    for nom, info in MODELES.items():
        print(f"\n== {nom} ==")
        pipe, valeur, f1_val, duree = regler(nom, X_train, y_train, X_val, y_val)
        pipes[nom] = pipe
        choix[nom] = {"hyperparam": info["hyperparam"], "valeur": valeur,
                      "val_f1": f1_val, "train_time_s": duree}
        joblib.dump(pipe, info["fichier"])
        print(f"  -> retenu : {info['hyperparam']}={valeur} (F1 val {f1_val:.4f}, {duree}s)")

    # Meilleur modèle : décidé sur la validation, avant de toucher au test
    best = max(choix, key=lambda n: choix[n]["val_f1"])
    joblib.dump(pipes[best], config.CLASSICAL_BEST_PATH)
    config.CLASSICAL_BEST_JSON_PATH.write_text(
        json.dumps({"name": best, "selected_on": "validation F1", "models": choix},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Évaluation finale : le test est lu ici pour la première et dernière fois
    print("\nÉvaluation finale sur le test (aucun réglage après ce point)")
    X_test, y_test = charger(config.TEST_CSV)
    for nom, pipe in pipes.items():
        y_pred = pipe.predict(X_test)
        metriques = compute_metrics(y_test, y_pred)
        metriques["train_time_s"] = choix[nom]["train_time_s"]
        save_metrics(nom, metriques)
        plot_confusion_matrix(y_test, y_pred, nom)
        print(f"  {nom:<20} acc={metriques['accuracy']:.4f}  F1={metriques['f1']:.4f}")

    sauver_top_features(pipes["logistic_regression"])
    print(f"\nMeilleur modèle classique (validation) : {best}")
    print(f"Fichiers : {config.MODELS_DIR} et {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
