"""Prétraitement du texte, commun à l'entraînement et à l'application.

La fonction clean_text doit rester identique partout : ne la modifier
qu'après accord des deux membres.
"""
import re

# Balises HTML : <br />, <b>, etc.
_HTML = re.compile(r"<[^>]+>")
# URL : http(s)://... ou www....
_URL = re.compile(r"(https?://\S+|www\.\S+)")
# Tout ce qui n'est pas lettre (accents inclus), chiffre, espace, ! ou ?
# Les négations (ne, n', pas, jamais, rien, aucun, plus, sans) sont des
# mots : elles sont conservées car elles inversent le sentiment.
_PONCTUATION = re.compile(r"[^\w\s!?]", flags=re.UNICODE)
_ESPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Nettoie un texte français et renvoie une chaîne propre.

    Étapes : minuscules, suppression des balises HTML et des URL,
    normalisation des apostrophes, suppression de la ponctuation sauf
    ! et ? (qui portent du sentiment), espaces multiples réduits.
    """
    if text is None:
        return ""
    texte = str(text).lower()
    texte = _HTML.sub(" ", texte)
    texte = _URL.sub(" ", texte)
    # Apostrophes typographiques -> apostrophe simple, puis séparation
    # (n'aime -> n aime) pour garder la négation "n" comme mot à part.
    texte = texte.replace("’", "'").replace("`", "'").replace("´", "'")
    texte = texte.replace("'", " ")
    # Underscore compte comme \w : on le retire aussi
    texte = texte.replace("_", " ")
    texte = _PONCTUATION.sub(" ", texte)
    texte = _ESPACES.sub(" ", texte).strip()
    return texte
