# Architecture CNN

Le modele principal est un CNN inspire de VGGNet, adapte a GTSRB.

## Structure generale

- 3 blocs convolutifs (32 -> 64 -> 128 filtres)
- Chaque bloc: 2 convolutions + BatchNorm + ReLU + MaxPool + Dropout
- 1 couche dense de 512 neurones
- Sortie softmax sur 43 classes

## Justification

- Filtres 3x3: extraction efficace de motifs locaux
- BatchNormalization: stabilite d'entrainement
- Dropout progressif: regularisation pour limiter le surapprentissage

## Fichiers

- Modele principal: `src/model.py` (fonction `build_cnn`)
- Modele leger: `src/model.py` (fonction `build_cnn_light`)
