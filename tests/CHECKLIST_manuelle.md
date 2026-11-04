# Checklist de tests manuels de l'application

Lancer `make app`, ouvrir http://localhost:8501, puis saisir chaque phrase avec le mode « Comparer les 4 modèles ».

| # | Cas | Phrase | Attendu |
|---|-----|--------|---------|
| 1 | Positif | Un chef-d'œuvre absolu, j'ai adoré du début à la fin ! | Positif |
| 2 | Positif | Les acteurs sont excellents et l'histoire est très touchante. | Positif |
| 3 | Négatif | Nul, ennuyeux, une perte de temps. | Négatif |
| 4 | Négatif | Scénario ridicule et mise en scène catastrophique. | Négatif |
| 5 | Négation | Ce film n'est pas mauvais du tout. | Positif (cas difficile) |
| 6 | Négation | Je ne le recommande à personne, ce n'est vraiment pas bon. | Négatif |
| 7 | Ironie | Bravo, deux heures de ma vie que je ne reverrai jamais, quel talent ! | Négatif (limite connue : erreur probable) |
| 8 | Très court | Génial ! | Positif |
| 9 | Hors domaine | Le service de ce restaurant était lent et les plats froids. | Négatif (hors cinéma : moins fiable) |
| 10 | Vide | (rien, ou seulement `...`) | Message d'avertissement, pas de plantage |

À vérifier aussi :

- [ ] Les 4 modèles apparaissent dans la liste de la barre latérale.
- [ ] Les 3 boutons d'exemple remplissent la zone de texte.
- [ ] Le badge est vert ou rouge, avec un score de confiance et sa barre.
- [ ] « Voir le texte nettoyé » affiche le texte après `clean_text`.
- [ ] L'onglet Performances affiche le tableau, les figures et les courbes du LSTM.
- [ ] L'onglet À propos s'affiche.
