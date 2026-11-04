"""Entraînement du LSTM bidirectionnel, sans aucun embedding pré-entraîné.

    python -m src.train_lstm           entraînement complet sur data/*.csv
    python -m src.train_lstm --quick   test rapide : 1000 textes, 1 epoch, aucun fichier écrit
                                       dans models/ ni results/ (fonctionne même sans CSV)
"""

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'écran dans le conteneur
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from src.config import (
    DATASET_NAME,
    LSTM_CURVES_PATH,
    LSTM_HISTORY_PATH,
    LSTM_PATH,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    VALIDATION_CSV,
)
from src.preprocessing import clean_text

MAX_TOKENS = 20000
SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
MAX_EPOCHS = 8
QUICK_SIZES = {"train": 1000, "validation": 200, "test": 200}
PHRASE_EXEMPLE = clean_text("Ce film est un chef-d'œuvre, j'ai adoré !")


def read_csv_split(path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lit un CSV du contrat et renvoie (textes nettoyés, labels)."""
    df = pd.read_csv(path, usecols=["clean_review", "label"]).dropna()
    df = df[df["clean_review"].str.strip() != ""]
    if limit is not None and limit < len(df):
        df = df.sample(limit, random_state=SEED)
    return df["clean_review"].to_numpy(dtype=str), df["label"].to_numpy(dtype=int)


def download_split(split: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Mode --quick sans CSV : petit échantillon téléchargé et nettoyé à la volée."""
    from datasets import load_dataset

    data = load_dataset(DATASET_NAME, split=split).shuffle(seed=SEED).select(range(limit))
    texts = [clean_text(review) for review in data["review"]]
    return np.array(texts, dtype=str), np.array(data["label"], dtype=int)


def load_split(name: str, path: Path, quick: bool) -> tuple[np.ndarray, np.ndarray]:
    """Charge un split ; en mode complet, le CSV est obligatoire."""
    if not quick:
        return read_csv_split(path)
    if path.exists():
        return read_csv_split(path, QUICK_SIZES[name])
    print(f"{path.name} absent : échantillon téléchargé depuis {DATASET_NAME}")
    return download_split(name, QUICK_SIZES[name])


def make_dataset(texts: np.ndarray, labels: np.ndarray, shuffle: bool = False) -> tf.data.Dataset:
    """Dataset de chaînes de forme (batch, 1), comme l'entrée du modèle."""
    inputs = tf.constant(texts.reshape(-1, 1))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels.astype("float32")))
    if shuffle:
        dataset = dataset.shuffle(len(texts), seed=SEED)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(train_texts: np.ndarray) -> keras.Model:
    """Construit le réseau ; le vocabulaire est appris sur le train uniquement."""
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_sequence_length=SEQUENCE_LENGTH,
        standardize=None,  # texte déjà nettoyé par clean_text
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(1024))

    longueurs = [len(text.split()) for text in train_texts]
    print(f"Taille du vocabulaire : {vectorizer.vocabulary_size()}")
    couverture = np.mean(np.array(longueurs) <= SEQUENCE_LENGTH)
    print(f"95e percentile de la longueur : {np.percentile(longueurs, 95):.0f} tokens")
    print(f"Critiques non tronquées à {SEQUENCE_LENGTH} tokens : {couverture:.1%}")

    inputs = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inputs)
    x = layers.Embedding(MAX_TOKENS, EMBEDDING_DIM, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="bilstm_sentiment")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model, train_ds, val_ds, epochs: int) -> tuple[dict, float]:
    """Entraîne avec arrêt anticipé ; renvoie l'historique et la durée en secondes."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])
    return history.history, time.time() - start


def plot_history(history: dict, path: Path) -> None:
    """Courbes de perte et d'accuracy : un panneau par mesure, train et validation."""
    panneaux = [("loss", "Perte (binary crossentropy)"), ("accuracy", "Accuracy")]
    series = [("", "Entraînement", "#2a78d6"), ("val_", "Validation", "#eb6834")]
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (mesure, titre) in zip(axes, panneaux):
        for prefixe, nom, couleur in series:
            ax.plot(epochs, history[prefixe + mesure], color=couleur, linewidth=2,
                    marker="o", markersize=6, markeredgecolor="white", label=nom)
        ax.set_title(titre, loc="left", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_xticks(list(epochs))
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict_labels(model, texts: np.ndarray) -> np.ndarray:
    """Prédiction = 1 si la probabilité est au moins 0.5."""
    inputs = tf.constant(texts.reshape(-1, 1))
    probas = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0).ravel()
    return (probas >= 0.5).astype(int)


def check_reload(model, path: Path) -> None:
    """Sauvegarde puis recharge le modèle : la prédiction doit rester identique."""
    model.save(path)
    reloaded = keras.models.load_model(path)
    entree = tf.constant([[PHRASE_EXEMPLE]])
    avant = float(model.predict(entree, verbose=0)[0][0])
    apres = float(reloaded.predict(entree, verbose=0)[0][0])
    if not np.isclose(avant, apres, atol=1e-5):
        raise RuntimeError(f"Modèle rechargé différent : {avant:.6f} contre {apres:.6f}")
    print(f"Rechargement vérifié : P(positif) = {apres:.4f} sur « {PHRASE_EXEMPLE} »")


def save_results(y_test, y_pred, history: dict, duration: float) -> None:
    """Enregistre historique, courbes, métriques et matrice de confusion."""
    from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics

    historique = {nom: [float(v) for v in valeurs] for nom, valeurs in history.items()}
    LSTM_HISTORY_PATH.write_text(json.dumps(historique, indent=2), encoding="utf-8")
    plot_history(history, LSTM_CURVES_PATH)

    metrics = compute_metrics(y_test, y_pred)
    print(f"Métriques sur le test : {metrics}")
    save_metrics("lstm", {**metrics, "train_time_s": round(duration, 1)})
    plot_confusion_matrix(y_test, y_pred, "lstm")


def main(quick: bool = False) -> None:
    keras.utils.set_random_seed(SEED)

    # On vérifie les prérequis avant d'entraîner, pas après plusieurs minutes de calcul.
    if not quick and not TRAIN_CSV.exists():
        sys.exit(f"{TRAIN_CSV} introuvable : lance d'abord make data, ou utilise --quick.")
    if not quick and importlib.util.find_spec("src.evaluate") is None:
        sys.exit("src/evaluate.py introuvable (Membre 1) : ajoute-le, ou utilise --quick.")

    x_train, y_train = load_split("train", TRAIN_CSV, quick)
    x_val, y_val = load_split("validation", VALIDATION_CSV, quick)
    print(f"Train : {len(x_train)} textes, validation : {len(x_val)} textes")

    model = build_model(x_train)
    model.summary()
    history, duration = train(
        model,
        make_dataset(x_train, y_train, shuffle=True),
        make_dataset(x_val, y_val),
        epochs=1 if quick else MAX_EPOCHS,
    )
    print(f"Temps d'entraînement : {duration / 60:.1f} min")

    # Le test n'est lu qu'ici, une fois l'entraînement terminé.
    x_test, y_test = load_split("test", TEST_CSV, quick)
    y_pred = predict_labels(model, x_test)

    if quick:
        with tempfile.TemporaryDirectory() as dossier:
            check_reload(model, Path(dossier) / "lstm_quick.keras")
        print(f"Mode --quick : accuracy sur {len(y_test)} textes = {(y_pred == y_test).mean():.4f}")
        return

    check_reload(model, LSTM_PATH)
    save_results(y_test, y_pred, history, duration)
    print(f"Modèle sauvegardé : {LSTM_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le LSTM de SentiText AI")
    parser.add_argument("--quick", action="store_true", help="test rapide : 1000 textes, 1 epoch")
    main(quick=parser.parse_args().quick)
