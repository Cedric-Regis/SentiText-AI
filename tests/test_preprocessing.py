"""Tests simples de clean_text."""
from src.preprocessing import clean_text


def test_minuscules_et_espaces():
    assert clean_text("  Ce FILM   est  Super  ") == "ce film est super"


def test_html_et_url_supprimes():
    texte = "Génial <br /> voir https://example.com/film et www.test.fr"
    resultat = clean_text(texte)
    assert "<" not in resultat
    assert "http" not in resultat
    assert "www" not in resultat
    assert "génial" in resultat


def test_ponctuation_sauf_exclamation_interrogation():
    resultat = clean_text("Quel film, vraiment... super ! Non ?")
    assert "," not in resultat
    assert "." not in resultat
    assert "!" in resultat
    assert "?" in resultat


def test_negations_conservees():
    resultat = clean_text("Ce n’est pas bon, je n'ai jamais ri, rien de drôle, sans intérêt")
    mots = resultat.split()
    for negation in ("pas", "jamais", "rien", "sans", "n"):
        assert negation in mots
