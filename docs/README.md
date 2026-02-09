# GTRSB-Project-Anthony-AMAR - Documentation

Ce projet implemente un pipeline complet de classification de panneaux routiers
allemands (GTSRB) avec un reseau de neurones convolutif (CNN).

## Structure

```
GTRSB-Project-Anthony-AMAR/
├── data/
│   └── GTSRB/
│       └── Training/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── best_model.keras
│   ├── history.json
│   └── evaluation_metrics.json
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── correct_predictions.png
│   └── error_predictions.png
├── notebooks/
│   └── GTSRB_CNN_Projet.ipynb
├── docs/
│   ├── README.md
│   ├── pipeline.md
│   ├── architecture.md
│   └── rapport.md
├── README.md
└── requirements.txt
```

Note: le dossier `data/` peut ne pas apparaitre sur GitHub tant qu'il est vide
(Git ne versionne pas les repertoires vides). Il doit etre cree/rempli en local
avec le dataset.

## Installation

```
pip install -r requirements.txt
```

## Utilisation

Entrainement complet:

```
python src/train.py
```

Test rapide:

```
python src/train.py --quick-test
```

Evaluation:

```
python src/evaluate.py
```

## Donnees

Le chemin attendu par defaut est `data/GTSRB/Training`. Le code accepte
egalement le chemin existant `data/GTRSB-Training_fixed/GTSRB/Training`
si le dataset n'a pas ete deplace.

Si vous ne voyez pas `data/` sur GitHub, c'est normal (dossier vide non pousse).
Creez la structure localement et decompressez le dataset pour obtenir au final:

- `data/GTSRB/Training`

Alternative (si vous avez conserve l'ancien emplacement):

- `data/GTRSB-Training_fixed/GTSRB/Training`
