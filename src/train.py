"""
Script d'entraînement du modèle CNN pour GTSRB
==============================================

Ce script gère l'ensemble du processus d'entraînement:
- Configuration des hyperparamètres
- Callbacks pour le suivi et l'optimisation
- Entraînement avec validation
- Sauvegarde du modèle et de l'historique

Auteur: Projet académique CNRS-FIDLE
"""

import os
import json
import argparse
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Imports locaux
from data_preprocessing import prepare_data, get_default_data_path
from model import build_cnn, build_cnn_light, print_model_architecture


# =============================================================================
# CONFIGURATION DES HYPERPARAMÈTRES
# =============================================================================

class Config:
    """Configuration centralisée des hyperparamètres."""
    
    # Chemins
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = get_default_data_path(BASE_DIR)
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
    
    # Hyperparamètres d'entraînement
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Reduce LR on plateau
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_MIN = 1e-6
    
    # Data augmentation
    USE_DATA_AUGMENTATION = True
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.15
    SHEAR_RANGE = 0.1


# =============================================================================
# FONCTIONS D'AUGMENTATION DE DONNÉES
# =============================================================================

def create_data_generator(config: Config = Config()) -> ImageDataGenerator:
    """
    Crée un générateur de données avec augmentation.
    
    L'augmentation de données est essentielle pour:
    - Prévenir le surapprentissage
    - Rendre le modèle robuste aux variations
    - Simuler différentes conditions de capture
    
    Transformations appliquées:
    - Rotation: ±15° (panneaux vus sous différents angles)
    - Translation: ±10% (panneaux décentrés)
    - Zoom: ±15% (différentes distances)
    - Cisaillement: ±10° (perspectives variées)
    
    Note: Pas de flip horizontal car les panneaux ont un sens!
    """
    return ImageDataGenerator(
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        zoom_range=config.ZOOM_RANGE,
        shear_range=config.SHEAR_RANGE,
        fill_mode='nearest'
        # PAS de horizontal_flip ni vertical_flip!
        # Les panneaux routiers ne doivent pas être inversés
    )


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(config: Config = Config()) -> list:
    """
    Crée les callbacks pour l'entraînement.
    
    Callbacks utilisés:
    1. ModelCheckpoint: Sauvegarde du meilleur modèle
    2. EarlyStopping: Arrêt si pas d'amélioration
    3. ReduceLROnPlateau: Réduction du learning rate
    4. CSVLogger: Log de l'historique
    
    Returns:
        Liste de callbacks Keras
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    callbacks = [
        # Sauvegarder le meilleur modèle (basé sur val_accuracy)
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # Arrêt anticipé si pas d'amélioration
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Réduire le learning rate si plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN,
            verbose=1
        )
    ]
    
    return callbacks


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def plot_training_history(history: dict, save_path: str = None):
    """
    Trace les courbes d'apprentissage (loss et accuracy).
    
    Arguments:
        history: Dictionnaire avec 'loss', 'accuracy', 'val_loss', 'val_accuracy'
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # === COURBE DE LOSS ===
    ax1 = axes[0]
    epochs = range(1, len(history['loss']) + 1)
    
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Entraînement', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    
    ax1.set_xlabel('Époque', fontsize=12)
    ax1.set_ylabel('Loss (Categorical Crossentropy)', fontsize=12)
    ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Marquer le minimum de val_loss
    min_val_loss = min(history['val_loss'])
    min_epoch = history['val_loss'].index(min_val_loss) + 1
    ax1.axvline(x=min_epoch, color='green', linestyle='--', alpha=0.7)
    ax1.annotate(f'Best: {min_val_loss:.4f}\n(epoch {min_epoch})',
                 xy=(min_epoch, min_val_loss),
                 xytext=(min_epoch + 2, min_val_loss + 0.1),
                 fontsize=9, color='green')
    
    # === COURBE D'ACCURACY ===
    ax2 = axes[1]
    
    ax2.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Entraînement', marker='o', markersize=4)
    ax2.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    
    ax2.set_xlabel('Époque', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Marquer le maximum de val_accuracy
    max_val_acc = max(history['val_accuracy'])
    max_epoch = history['val_accuracy'].index(max_val_acc) + 1
    ax2.axvline(x=max_epoch, color='green', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {max_val_acc:.4f}\n(epoch {max_epoch})',
                 xy=(max_epoch, max_val_acc),
                 xytext=(max_epoch + 2, max_val_acc - 0.1),
                 fontsize=9, color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Courbes sauvegardees: {save_path}")
    
    plt.show()


# =============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================

def train_model(config: Config = Config(), 
                quick_test: bool = False,
                use_light_model: bool = False) -> tuple:
    """
    Entraîne le modèle CNN sur le dataset GTSRB.
    
    Arguments:
        config: Configuration des hyperparamètres
        quick_test: Si True, entraînement court pour validation
        use_light_model: Si True, utilise le modèle léger
    
    Returns:
        tuple: (model, history)
    """
    print("\n" + "=" * 70)
    print("ENTRAINEMENT DU MODELE CNN GTSRB")
    print("=" * 70)
    
    # Réduire les epochs pour un test rapide
    epochs = 5 if quick_test else config.EPOCHS
    
    # =========================================================================
    # 1. CHARGEMENT DES DONNÉES
    # =========================================================================
    print("\nChargement des donnees...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(config.DATA_PATH)
    
    # =========================================================================
    # 2. CONSTRUCTION DU MODÈLE
    # =========================================================================
    print("\nConstruction du modele...")
    if use_light_model:
        model = build_cnn_light()
        print("   Mode leger active")
    else:
        model = build_cnn()
    
    print_model_architecture(model)
    
    # =========================================================================
    # 3. COMPILATION
    # =========================================================================
    print("\nCompilation du modele...")
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Optimiseur: Adam (lr={config.LEARNING_RATE})")
    print("   Loss: Categorical Crossentropy")
    print("   Metrique: Accuracy")
    
    # =========================================================================
    # 4. CALLBACKS
    # =========================================================================
    callbacks = create_callbacks(config)
    print(f"\nCallbacks configures: {len(callbacks)}")
    
    # =========================================================================
    # 5. AUGMENTATION DE DONNÉES
    # =========================================================================
    if config.USE_DATA_AUGMENTATION and not quick_test:
        print("\nAugmentation de donnees activee")
        datagen = create_data_generator(config)
        datagen.fit(X_train)
        
        # Entraînement avec générateur
        print("\nLancement de l'entrainement...")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Epochs: {epochs}")
        print(f"   Images d'entrainement: {len(X_train)}")
        print()
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
            steps_per_epoch=len(X_train) // config.BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Entraînement sans augmentation (plus rapide pour tests)
        print("\nLancement de l'entrainement (sans augmentation)...")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Epochs: {epochs}")
        print()
        
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # =========================================================================
    # 6. RÉSULTATS FINAUX
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENTRAINEMENT TERMINE")
    print("=" * 70)
    
    # Évaluation sur le test set
    print("\nEvaluation sur le jeu de test...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("\nRESULTATS FINAUX:")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Sauvegarder l'historique
    history_dict = history.history
    history_path = os.path.join(config.MODELS_DIR, 'history.json')
    with open(history_path, 'w') as f:
        # Convertir les valeurs numpy en float pour JSON
        history_serializable = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
        json.dump(history_serializable, f, indent=2)
    print(f"Historique sauvegarde: {history_path}")
    
    # Tracer les courbes d'apprentissage
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plot_path = os.path.join(config.FIGURES_DIR, 'training_curves.png')
    plot_training_history(history_dict, save_path=plot_path)
    
    return model, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraînement du CNN GTSRB')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Taille du batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--quick-test', action='store_true', help='Test rapide (5 epochs)')
    parser.add_argument('--light', action='store_true', help='Utiliser le modèle léger')
    
    args = parser.parse_args()
    
    # Mettre à jour la config si nécessaire
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    # Lancer l'entraînement
    model, history = train_model(
        config=config,
        quick_test=args.quick_test,
        use_light_model=args.light
    )
