# Pipeline de donnees

Ce document decrit le pipeline de preparation des donnees utilise
pour le dataset GTSRB.

## Etapes

1. Chargement des images et des annotations CSV par classe.
2. Redimensionnement a 32x32 pixels et conversion RGB.
3. Normalisation des pixels dans l'intervalle [0, 1].
4. Encodage one-hot des labels (43 classes).
5. Separation des donnees en train, validation et test.

## Parametres

- Taille d'image: 32x32
- Classes: 43
- Repartition: 70% train, 15% validation, 15% test

## Fichiers

- Code: `src/data_preprocessing.py`
- Donnees: `data/GTSRB/Training`

Note: le dossier `data/` peut ne pas apparaitre sur GitHub s'il est vide (Git ne
versionne pas les repertoires vides). Voir `docs/README.md` pour les instructions
de mise en place du dataset.
