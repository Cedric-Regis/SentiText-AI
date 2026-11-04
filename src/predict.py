"""Seule passerelle entre les modèles entraînés et l'interface."""

import joblib

from src.config import (
    LABELS,
    LOGISTIC_REGRESSION_PATH,
    LSTM_PATH,
    NAIVE_BAYES_PATH,
    SVM_PATH,
)
from src.preprocessing import clean_text

MODEL_FILES = {
    "Naive Bayes": NAIVE_BAYES_PATH,
    "Logistic Regression": LOGISTIC_REGRESSION_PATH,
    "SVM": SVM_PATH,
    "LSTM": LSTM_PATH,
}


def _load_model(path):
    """Charge un modèle ; TensorFlow n'est importé que si le LSTM est présent."""
    if path.suffix == ".keras":
        import keras

        return keras.models.load_model(path)
    return joblib.load(path)


def load_models() -> dict:
    """Renvoie {nom affiché: modèle} pour tous les modèles présents et lisibles."""
    models = {}
    for name, path in MODEL_FILES.items():
        if not path.exists():
            continue
        try:
            models[name] = _load_model(path)
        except Exception as error:  # un fichier abîmé ne doit pas bloquer les autres
            print(f"[predict] {name} ignoré ({path.name}) : {error}")
    return models


def _proba_positive(model, clean: str) -> float:
    """P(positif) pour un texte nettoyé, quel que soit le type de modèle."""
    if hasattr(model, "predict_proba"):  # Pipeline scikit-learn
        probas = model.predict_proba([clean])[0]
        return float(probas[list(model.classes_).index(1)])
    # Modèle Keras : entrée de forme (1, 1), sortie sigmoïde.
    # tf.constant est obligatoire : Keras 3 refuse les tableaux NumPy de chaînes.
    import tensorflow as tf

    return float(model.predict(tf.constant([[clean]]), verbose=0)[0][0])


def predict(text: str, model_name: str, models: dict) -> dict:
    """Prédit le sentiment d'un texte brut avec le modèle demandé."""
    clean = clean_text(text)
    if not clean:
        raise ValueError("Le texte est vide après nettoyage : saisis une phrase.")
    if model_name not in models:
        raise ValueError(f"Modèle indisponible : {model_name}")

    proba = _proba_positive(models[model_name], clean)
    label = int(proba >= 0.5)
    return {
        "model": model_name,
        "label": label,
        "label_name": LABELS[label],
        "confidence": proba if label == 1 else 1 - proba,
        "proba_positive": proba,
        "clean_text": clean,
    }


def predict_all(text: str, models: dict) -> list:
    """Applique tous les modèles chargés à un même texte."""
    return [predict(text, name, models) for name in models]


if __name__ == "__main__":
    modeles = load_models()
    print(f"Modèles chargés : {list(modeles) or 'aucun'}")
    exemples = [
        "Ce film est un chef-d'œuvre, j'ai adoré du début à la fin !",
        "Nul, ennuyeux, une perte de temps.",
        "Ce n'est pas un mauvais film, mais je ne le reverrai jamais.",
    ]
    for phrase in exemples:
        print(f"\n{phrase}")
        for r in predict_all(phrase, modeles):
            print(f"  {r['model']:<20} {r['label_name']:<8} {r['confidence']:.1%}")
