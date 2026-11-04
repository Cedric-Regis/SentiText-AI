"""Nettoyage du texte, identique à l'entraînement et dans l'application."""

import html
import re

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_APOSTROPHES = re.compile(r"[’‘ʼ`´]")
# Tout ce qui n'est ni lettre, ni chiffre, ni espace, ni apostrophe, ni ! ou ?
_PONCTUATION = re.compile(r"[^\w\s'!?]|_")
_APOSTROPHE_ISOLEE = re.compile(r"(?<!\w)'")
_ELISION = re.compile(r"(?<=\w)'(?=\w)")
_EXCLAMATION = re.compile(r"([!?]+)")


def clean_text(text) -> str:
    """Nettoie une critique sans retirer les négations (ne, n', pas, jamais, rien...).

    « n'est » devient « n' est » et « super! » devient « super ! » : ces mots,
    « ! » et « ? » restent ainsi des tokens séparés pour le LSTM.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text).lower()
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _APOSTROPHES.sub("'", text)
    text = _PONCTUATION.sub(" ", text)
    text = _APOSTROPHE_ISOLEE.sub(" ", text)
    text = _ELISION.sub("' ", text)
    text = _EXCLAMATION.sub(r" \1 ", text)
    return " ".join(text.split())
