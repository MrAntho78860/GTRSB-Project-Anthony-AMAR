# Projet GTSRB - Documentation

Ce projet implemente un pipeline complet de classification de panneaux routiers
allemands (GTSRB) avec un reseau de neurones convolutif (CNN).

## Structure

```
Projet compensation/
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

Pour correspondre a la structure de l'archive fournie, placez le dossier
`Training` de l'archive zip du dataset dans:
`\GTRSB-Project - Anthony AMAR\data\GTRSB-Training_fixed\GTSRB`
