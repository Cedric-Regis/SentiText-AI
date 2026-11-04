"""Configuration commune de SentiText AI.

Source de vérité unique : chemins, graine, tailles d'échantillon.
Tous les scripts importent ce module, jamais de chemin en dur.
"""
from pathlib import Path

# --- Racine du projet (le dossier qui contient src/) ---
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Dossiers ---
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Création des dossiers s'ils n'existent pas
for _dossier in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    _dossier.mkdir(parents=True, exist_ok=True)

# --- Reproductibilité et tailles d'échantillon ---
SEED = 42
N_TRAIN = 12000
N_VAL = 2000
N_TEST = 2000

# --- Étiquettes ---
LABELS = {0: "négatif", 1: "positif"}

# --- Dataset Hugging Face ---
DATASET_NAME = "tblard/allocine"

# --- Fichiers de données ---
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

# --- Fichiers de modèles ---
NAIVE_BAYES_FILE = MODELS_DIR / "naive_bayes.pkl"
LOGREG_FILE = MODELS_DIR / "logistic_regression.pkl"
SVM_FILE = MODELS_DIR / "svm.pkl"
CLASSICAL_BEST_FILE = MODELS_DIR / "classical_best.pkl"
LSTM_FILE = MODELS_DIR / "lstm_sentiment.keras"
LSTM_VOCAB_FILE = MODELS_DIR / "lstm_vocab.json"

# --- Fichiers de résultats ---
METRICS_FILE = RESULTS_DIR / "metrics.json"
DATA_SUMMARY_FILE = RESULTS_DIR / "data_summary.json"
REVIEW_LENGTH_HIST = RESULTS_DIR / "review_length_hist.png"
CLASSICAL_BEST_JSON = RESULTS_DIR / "classical_best.json"
TOP_FEATURES_FILE = RESULTS_DIR / "top_features.json"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"
COMPARISON_PNG = RESULTS_DIR / "comparison.png"
CONFUSION_PNG = RESULTS_DIR / "confusion_matrix.png"
LSTM_HISTORY_FILE = RESULTS_DIR / "lstm_history.json"
LSTM_CURVES_PNG = RESULTS_DIR / "lstm_training_curves.png"
