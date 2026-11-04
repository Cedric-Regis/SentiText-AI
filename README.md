# SentiText AI

*Comprendre les mots, détecter les émotions.*

Classification du sentiment (positif / négatif) de critiques de films en français, à partir du dataset
[Allociné](https://huggingface.co/datasets/tblard/allocine). Quatre modèles sont entraînés et comparés :
Naive Bayes, régression logistique, SVM et LSTM bidirectionnel. Aucun modèle pré-entraîné n'est utilisé.

## Lancement avec Docker

```bash
docker compose build   # ou make build (5 à 10 min la première fois)
make train             # données, puis modèles classiques, puis LSTM
make evaluate          # tableau comparatif et figures
make app               # puis http://localhost:8501
```

Les fichiers produits restent sur la machine, dans `data/`, `models/` et `results/`.
Après une modification du code, relancer `make build`. `make test` lance les tests.

## Alternative sans Docker

Python 3.11 requis. TensorFlow 2.18 n'est pas disponible sur Mac Intel : Docker y est obligatoire.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.prepare_data
python -m src.train_classical
python -m src.train_lstm
python -m src.evaluate
streamlit run app.py
```
