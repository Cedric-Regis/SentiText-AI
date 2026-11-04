"""Préparation des données : chargement Allociné, exploration, échantillonnage.

    python -m src.prepare_data

Produit data/train.csv, validation.csv, test.csv (review, clean_review, label),
results/data_summary.json et results/review_length_hist.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.preprocessing import clean_text


def charger_allocine():
    """Charge le dataset ; réessaie avec trust_remote_code si le script bloque."""
    from datasets import load_dataset
    try:
        return load_dataset(config.DATASET_NAME)
    except Exception as erreur:
        print(f"Chargement standard échoué ({erreur}), nouvel essai avec trust_remote_code=True")
        return load_dataset(config.DATASET_NAME, trust_remote_code=True)


def explorer(nom, df):
    """Affiche taille, classes, valeurs manquantes, longueur des critiques."""
    longueurs = df["review"].fillna("").str.split().str.len()
    print(f"\n[{nom}] {len(df)} textes")
    print("  classes :", df["label"].value_counts().sort_index().to_dict())
    print("  valeurs manquantes :", int(df["review"].isna().sum()))
    print(f"  longueur (mots) : moyenne={longueurs.mean():.1f}  médiane={longueurs.median():.0f}")
    return longueurs


def nettoyer_split(df):
    """Ajoute clean_review, supprime les lignes vides et les doublons."""
    df = df[["review", "label"]].dropna().copy()
    df["clean_review"] = df["review"].map(clean_text)
    df = df[df["clean_review"].str.len() > 0]
    df = df.drop_duplicates(subset="clean_review")
    return df.reset_index(drop=True)


def echantillonner(df, n):
    """Échantillon stratifié de n textes (graine fixe)."""
    if len(df) <= n:
        return df.copy()
    ech, _ = train_test_split(df, train_size=n, stratify=df["label"],
                              random_state=config.SEED)
    return ech.sample(frac=1, random_state=config.SEED).reset_index(drop=True)


def main():
    brut = charger_allocine()
    splits, longueurs = {}, {}
    for nom in ("train", "validation", "test"):
        df = brut[nom].to_pandas()
        longueurs[nom] = explorer(nom, df)
        splits[nom] = nettoyer_split(df)

    # Aucune critique de validation/test ne doit se trouver dans le train
    interdits = set(splits["validation"]["clean_review"]) | set(splits["test"]["clean_review"])
    avant = len(splits["train"])
    splits["train"] = splits["train"][~splits["train"]["clean_review"].isin(interdits)]
    print(f"\nFuites retirées du train : {avant - len(splits['train'])}")

    tailles = {"train": config.N_TRAIN, "validation": config.N_VAL, "test": config.N_TEST}
    chemins = {"train": config.TRAIN_CSV, "validation": config.VAL_CSV, "test": config.TEST_CSV}
    resume = {}
    for nom, df in splits.items():
        ech = echantillonner(df, tailles[nom])
        ech[["review", "clean_review", "label"]].to_csv(chemins[nom], index=False)
        resume[nom] = {"taille": len(ech),
                       "classes": {str(k): int(v) for k, v in
                                   ech["label"].value_counts().sort_index().items()}}
        print(f"{nom}: {len(ech)} textes -> {chemins[nom].name}")

    config.DATA_SUMMARY_FILE.write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    plt.hist(longueurs["train"].clip(upper=500), bins=50, color="steelblue")
    plt.xlabel("Longueur (mots, tronquée à 500)")
    plt.ylabel("Nombre de critiques")
    plt.title("Distribution des longueurs - train")
    plt.tight_layout()
    plt.savefig(config.REVIEW_LENGTH_HIST, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
