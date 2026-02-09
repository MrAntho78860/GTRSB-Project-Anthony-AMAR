"""
Module d'architecture CNN pour GTSRB
====================================

Ce module définit l'architecture du réseau de neurones convolutif (CNN)
pour la reconnaissance de panneaux routiers allemands.

Architecture inspirée de VGGNet avec des adaptations pour le dataset GTSRB:
- 3 blocs convolutifs avec complexité croissante
- BatchNormalization pour la stabilité
- Dropout pour la régularisation

Auteur: Projet académique CNRS-FIDLE
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation, Input, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dimensions d'entrée par défaut
DEFAULT_INPUT_SHAPE = (32, 32, 3)

# Nombre de classes
NUM_CLASSES = 43


# =============================================================================
# ARCHITECTURE CNN PRINCIPALE
# =============================================================================

def build_cnn(input_shape: tuple = DEFAULT_INPUT_SHAPE, 
              num_classes: int = NUM_CLASSES,
              dropout_rate: float = 0.25,
              dense_dropout: float = 0.5) -> Sequential:
    """
    Construit le modèle CNN pour la classification GTSRB.
    
    Architecture:
    - 3 blocs convolutifs (32 → 64 → 128 filtres)
    - Chaque bloc: 2x (Conv2D + BatchNorm + ReLU) + MaxPool + Dropout
    - 1 couche dense de 512 neurones
    - Sortie softmax pour 43 classes
    
    Arguments:
        input_shape: Forme des images d'entrée (H, W, C)
        num_classes: Nombre de classes de sortie
        dropout_rate: Taux de dropout après les blocs conv (défaut: 0.25)
        dense_dropout: Taux de dropout après le dense layer (défaut: 0.5)
    
    Returns:
        Model Keras compilé
    
    Justification des choix architecturaux:
    ----------------------------------------
    1. Filtres 3x3: Standard depuis VGGNet, détecte efficacement les patterns locaux
    2. BatchNormalization: Stabilise l'entraînement, permet des learning rates plus élevés
    3. Dropout progressif (0.25 → 0.5): Régularisation douce au début, forte avant le dense
    4. 512 neurones denses: Compromis entre capacité et risque de surapprentissage
    """
    
    model = Sequential(name='GTSRB_CNN')
    
    # =========================================================================
    # BLOC 1: Extraction des features de bas niveau (bords, contours)
    # =========================================================================
    # Première couche: doit spécifier input_shape
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='conv1_1'))
    model.add(BatchNormalization(name='bn1_1'))
    model.add(Activation('relu', name='relu1_1'))
    
    model.add(Conv2D(32, (3, 3), padding='same', name='conv1_2'))
    model.add(BatchNormalization(name='bn1_2'))
    model.add(Activation('relu', name='relu1_2'))
    
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Dropout(dropout_rate, name='dropout1'))
    
    # =========================================================================
    # BLOC 2: Extraction des features de niveau intermédiaire (formes)
    # =========================================================================
    model.add(Conv2D(64, (3, 3), padding='same', name='conv2_1'))
    model.add(BatchNormalization(name='bn2_1'))
    model.add(Activation('relu', name='relu2_1'))
    
    model.add(Conv2D(64, (3, 3), padding='same', name='conv2_2'))
    model.add(BatchNormalization(name='bn2_2'))
    model.add(Activation('relu', name='relu2_2'))
    
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(dropout_rate, name='dropout2'))
    
    # =========================================================================
    # BLOC 3: Extraction des features de haut niveau (motifs complexes)
    # =========================================================================
    model.add(Conv2D(128, (3, 3), padding='same', name='conv3_1'))
    model.add(BatchNormalization(name='bn3_1'))
    model.add(Activation('relu', name='relu3_1'))
    
    model.add(Conv2D(128, (3, 3), padding='same', name='conv3_2'))
    model.add(BatchNormalization(name='bn3_2'))
    model.add(Activation('relu', name='relu3_2'))
    
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(Dropout(dropout_rate, name='dropout3'))
    
    # =========================================================================
    # CLASSIFICATION HEAD
    # =========================================================================
    model.add(Flatten(name='flatten'))
    
    # Couche dense avec régularisation forte
    model.add(Dense(512, name='dense1'))
    model.add(BatchNormalization(name='bn_dense'))
    model.add(Activation('relu', name='relu_dense'))
    model.add(Dropout(dense_dropout, name='dropout_dense'))
    
    # Couche de sortie: 43 classes avec activation softmax
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    return model


def build_cnn_light(input_shape: tuple = DEFAULT_INPUT_SHAPE,
                    num_classes: int = NUM_CLASSES) -> Sequential:
    """
    Version légère du CNN pour tests rapides.
    
    Architecture simplifiée avec moins de paramètres:
    - 2 blocs convolutifs seulement
    - Moins de filtres (16 → 32)
    
    Arguments:
        input_shape: Forme des images d'entrée
        num_classes: Nombre de classes de sortie
    
    Returns:
        Model Keras non compilé
    """
    model = Sequential(name='GTSRB_CNN_Light')
    
    # Bloc 1
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Bloc 2
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Classification
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_model_summary(model: Sequential) -> str:
    """
    Retourne un résumé du modèle sous forme de chaîne de caractères.
    """
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)


def count_parameters(model: Sequential) -> dict:
    """
    Compte les paramètres du modèle.
    
    Returns:
        dict avec 'total', 'trainable' et 'non_trainable'
    """
    trainable = sum([w.numpy().size for w in model.trainable_weights])
    non_trainable = sum([w.numpy().size for w in model.non_trainable_weights])
    
    return {
        'total': trainable + non_trainable,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def print_model_architecture(model: Sequential):
    """
    Affiche l'architecture du modèle de façon détaillée.
    """
    print("\n" + "=" * 70)
    print("ARCHITECTURE DU MODELE CNN")
    print("=" * 70)
    
    model.summary()
    
    print("\n" + "-" * 70)
    total_params = model.count_params()
    print(f"Total des parametres: {total_params:,}")
    print(f"Taille estimee: {total_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 70)


# =============================================================================
# MAIN (pour tests)
# =============================================================================

if __name__ == "__main__":
    print("Construction du modele CNN...\n")
    
    # Construire le modèle principal
    model = build_cnn()
    print_model_architecture(model)
    
    print("\nResume des couches:")
    for i, layer in enumerate(model.layers):
        print(f"   {i:2d}. {layer.name:20s} | Output: {layer.output_shape}")
