from src.preprocessing import clean_text


def test_minuscules_html_et_url():
    assert clean_text("<b>SUPER</b> film http://exemple.fr www.exemple.fr") == "super film"


def test_negations_conservees():
    resultat = clean_text("Ce n’est pas bon, je ne regarderai jamais, rien, aucun, plus, sans.")
    for mot in ["n'", "pas", "ne", "jamais", "rien", "aucun", "plus", "sans"]:
        assert mot in resultat.split()


def test_ponctuation_sauf_exclamation_et_question():
    assert clean_text("Génial,   vraiment... non?! (oui)") == "génial vraiment non ?! oui"


def test_texte_vide_ou_invalide():
    assert clean_text("") == ""
    assert clean_text("  ... ") == ""
    assert clean_text(None) == ""
