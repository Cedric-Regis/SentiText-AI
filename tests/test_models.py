"""Tests des modèles entraînés (ignorés si le fichier du modèle n'existe pas encore)."""
import inspect

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, train_classical
from src.preprocessing import clean_text

POSITIF = "ce film est un chef-d'œuvre, j'ai adoré"
NEGATIF = "nul, ennuyeux, une perte de temps"

PKL = {
    "naive_bayes": config.NAIVE_BAYES_FILE,
    "logistic_regression": config.LOGREG_FILE,
    "svm": config.SVM_FILE,
    "classical_best": config.CLASSICAL_BEST_FILE,
}


@pytest.mark.parametrize("nom", list(PKL))
def test_modele_classique(nom):
    chemin = PKL[nom]
    if not chemin.exists():
        pytest.skip(f"{chemin.name} absent")
    modele = joblib.load(chemin)
    textes = [clean_text(POSITIF), clean_text(NEGATIF)]
    proba = modele.predict_proba(textes)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert list(modele.predict(textes)) == [1, 0]


def test_lstm():
    if not config.LSTM_FILE.exists():
        pytest.skip(f"{config.LSTM_FILE.name} absent")
    keras = pytest.importorskip("keras")
    modele = keras.models.load_model(config.LSTM_FILE)
    p_pos = float(modele.predict(np.array([[clean_text(POSITIF)]]), verbose=0)[0][0])
    p_neg = float(modele.predict(np.array([[clean_text(NEGATIF)]]), verbose=0)[0][0])
    assert p_pos >= 0.5
    assert p_neg < 0.5


def test_tokenizer_garde_negation_et_ponctuation():
    """Le TF-IDF doit voir le "n" de la négation et les ! ?, pas les jeter."""
    analyseur = TfidfVectorizer(token_pattern=train_classical.TOKEN_PATTERN).build_analyzer()
    for texte in (clean_text("ce n'est pas super !"), "ce n' est pas super !"):
        jetons = analyseur(texte)
        assert any(j in ("n", "n'") for j in jetons)
        assert "!" in jetons


def test_test_jamais_utilise_pour_regler():
    """Le réglage (regler) ne doit jamais toucher au jeu de test."""
    code = inspect.getsource(train_classical.regler).lower()
    for interdit in ("test_csv", "x_test", "y_test"):
        assert interdit not in code
    # Le CSV de test n'est référencé que dans main(), après le choix du meilleur modèle
    for fonction in (train_classical.charger, train_classical.construire,
                     train_classical.regler, train_classical.sauver_top_features):
        assert "TEST_CSV" not in inspect.getsource(fonction)
    source_main = inspect.getsource(train_classical.main)
    assert source_main.index("CLASSICAL_BEST_FILE") < source_main.index("TEST_CSV")
