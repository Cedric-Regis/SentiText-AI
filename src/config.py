"""Source de vérité du projet : chemins, graine et tailles d'échantillon."""

from pathlib import Path

# Racine du dépôt (le dossier parent de src/)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for _dossier in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    _dossier.mkdir(parents=True, exist_ok=True)

# Reproductibilité et échantillonnage
SEED = 42
N_TRAIN = 12000
N_VAL = 2000
N_TEST = 2000
LABELS = {0: "négatif", 1: "positif"}
DATASET_NAME = "tblard/allocine"

# Données
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Modèles
NAIVE_BAYES_PATH = MODELS_DIR / "naive_bayes.pkl"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression.pkl"
SVM_PATH = MODELS_DIR / "svm.pkl"
CLASSICAL_BEST_PATH = MODELS_DIR / "classical_best.pkl"
LSTM_PATH = MODELS_DIR / "lstm_sentiment.keras"

# Résultats
METRICS_PATH = RESULTS_DIR / "metrics.json"
DATA_SUMMARY_PATH = RESULTS_DIR / "data_summary.json"
REVIEW_LENGTH_HIST_PATH = RESULTS_DIR / "review_length_hist.png"
CLASSICAL_BEST_JSON_PATH = RESULTS_DIR / "classical_best.json"
TOP_FEATURES_PATH = RESULTS_DIR / "top_features.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "comparison.csv"
COMPARISON_PNG_PATH = RESULTS_DIR / "comparison.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
LSTM_HISTORY_PATH = RESULTS_DIR / "lstm_history.json"
LSTM_CURVES_PATH = RESULTS_DIR / "lstm_training_curves.png"
