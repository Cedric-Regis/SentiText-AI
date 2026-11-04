"""Interface Streamlit de SentiText AI."""

import json

import pandas as pd
import streamlit as st

from src.config import (
    COMPARISON_PNG_PATH,
    CONFUSION_MATRIX_PATH,
    LSTM_CURVES_PATH,
    METRICS_PATH,
    RESULTS_DIR,
)
from src.predict import load_models, predict, predict_all

EXEMPLES = {
    "Positif": "Un film magnifique, des acteurs excellents et une histoire qui m'a ému du début à la fin.",
    "Négatif": "Scénario ridicule, acteurs mauvais : je me suis ennuyé pendant deux heures.",
    "Ironique": "Bravo, deux heures de ma vie que je ne reverrai jamais, quel talent !",
}

STYLE = """
<style>
.badge {display: inline-block; padding: .35rem 1rem; border-radius: 999px;
        color: #fff; font-weight: 600; font-size: 1.1rem; margin-bottom: .75rem;}
.badge-positif {background: #1a7f4b;}
.badge-negatif {background: #c0362c;}
</style>
"""


@st.cache_resource(show_spinner="Chargement des modèles...")
def get_models() -> dict:
    return load_models()


def choisir_exemple(texte: str) -> None:
    """Callback : remplit la zone de texte avant qu'elle soit redessinée."""
    st.session_state["texte"] = texte


def afficher_resultat(resultat: dict) -> None:
    """Badge, score de confiance et texte nettoyé (F02, F04, F05)."""
    positif = resultat["label"] == 1
    classe, libelle = ("badge-positif", "Positif") if positif else ("badge-negatif", "Négatif")
    st.markdown(f'<span class="badge {classe}">{libelle}</span>', unsafe_allow_html=True)
    st.write(f"Confiance ({resultat['model']}) : **{resultat['confidence']:.1%}**")
    st.progress(resultat["confidence"])
    with st.expander("Voir le texte nettoyé"):
        st.text(resultat["clean_text"])


def afficher_comparaison(resultats: list) -> None:
    """Tableau des prédictions de tous les modèles sur le même texte."""
    lignes = [
        {
            "Modèle": r["model"],
            "Prédiction": r["label_name"].capitalize(),
            "Confiance": f"{r['confidence']:.1%}",
        }
        for r in resultats
    ]
    st.dataframe(pd.DataFrame(lignes), hide_index=True, use_container_width=True)
    with st.expander("Voir le texte nettoyé"):
        st.text(resultats[0]["clean_text"])


def onglet_analyser(models: dict, modele: str, comparer: bool) -> None:
    st.write("Essayer un exemple :")
    colonnes = st.columns(len(EXEMPLES))
    for colonne, (nom, texte) in zip(colonnes, EXEMPLES.items()):
        colonne.button(nom, on_click=choisir_exemple, args=(texte,), use_container_width=True)

    texte = st.text_area("Votre critique de film", key="texte", height=140)
    if not st.button("Analyser", type="primary"):
        return

    if not models:
        st.warning("Aucun modèle trouvé : lance d'abord `make train`.")
        return
    try:
        if comparer:
            afficher_comparaison(predict_all(texte, models))
        else:
            afficher_resultat(predict(texte, modele, models))
    except ValueError as erreur:
        st.warning(str(erreur))
    except Exception as erreur:
        st.warning(f"La prédiction a échoué : {erreur}")


def lire_metriques() -> dict:
    """Lit metrics.json ; renvoie {} s'il est absent ou illisible."""
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def afficher_image(chemin, legende: str) -> bool:
    if not chemin.exists():
        return False
    st.image(str(chemin), caption=legende)
    return True


def onglet_performances() -> None:
    metriques = lire_metriques()
    if metriques:
        tableau = pd.DataFrame(metriques).T.rename_axis("Modèle")
        if "f1" in tableau.columns:
            tableau = tableau.sort_values("f1", ascending=False)
        st.subheader("Résultats sur le jeu de test")
        st.dataframe(tableau, use_container_width=True)

    a_une_figure = afficher_image(COMPARISON_PNG_PATH, "Comparaison des modèles")
    if CONFUSION_MATRIX_PATH.exists():
        a_une_figure |= afficher_image(CONFUSION_MATRIX_PATH, "Matrices de confusion")
    else:
        for chemin in sorted(RESULTS_DIR.glob("confusion_matrix_*.png")):
            nom = chemin.stem.removeprefix("confusion_matrix_")
            a_une_figure |= afficher_image(chemin, f"Matrice de confusion : {nom}")
    a_une_figure |= afficher_image(LSTM_CURVES_PATH, "Courbes d'apprentissage du LSTM")

    if not metriques and not a_une_figure:
        st.info("Aucun résultat pour l'instant : lance `make train`, puis `make evaluate`.")


def onglet_a_propos() -> None:
    st.markdown(
        """
**Dataset** : critiques de films en français [Allociné](https://huggingface.co/datasets/tblard/allocine),
échantillon stratifié de 12 000 textes d'entraînement, 2 000 de validation et 2 000 de test.

**Modèles**, tous entraînés par l'équipe sur ces données :
- Naive Bayes, régression logistique et SVM calibré, sur une représentation TF-IDF ;
- LSTM bidirectionnel, avec un vocabulaire et un embedding appris.

**Aucun modèle pré-entraîné** : ni BERT, ni CamemBERT, ni Word2Vec, ni embeddings téléchargés.

**Limites** : l'ironie et le sarcasme sont souvent mal compris, et les modèles connaissent
le vocabulaire du cinéma : ils sont moins fiables sur d'autres domaines.
"""
    )


def main() -> None:
    st.set_page_config(page_title="SentiText AI", page_icon="🎬")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("SentiText AI")
    st.caption("Comprendre les mots, détecter les émotions.")

    try:
        models = get_models()
    except Exception as erreur:
        st.warning(f"Impossible de charger les modèles : {erreur}")
        models = {}

    with st.sidebar:
        st.header("Réglages")
        if models:
            modele = st.selectbox("Modèle", list(models))
            comparer = st.checkbox(f"Comparer les {len(models)} modèles")
        else:
            modele, comparer = None, False
            st.warning("Aucun modèle trouvé : lance d'abord `make train`.")

    analyser, performances, a_propos = st.tabs(["Analyser", "Performances", "À propos"])
    with analyser:
        onglet_analyser(models, modele, comparer)
    with performances:
        onglet_performances()
    with a_propos:
        onglet_a_propos()


if __name__ == "__main__":
    main()
