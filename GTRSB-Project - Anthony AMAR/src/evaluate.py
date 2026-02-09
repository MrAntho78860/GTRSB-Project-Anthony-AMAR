"""
Module d'évaluation du modèle CNN pour GTSRB
============================================

Ce module fournit des outils complets pour évaluer
les performances du modèle entraîné:
- Métriques détaillées par classe
- Matrice de confusion
- Analyse des erreurs
- Visualisation des prédictions

Auteur: Projet académique CNRS-FIDLE
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode headless sans GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from tensorflow.keras.models import load_model

# Imports locaux
from data_preprocessing import (
    prepare_data, CLASS_NAMES, NUM_CLASSES, get_default_data_path
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DATA_PATH = get_default_data_path(BASE_DIR)


# =============================================================================
# FONCTIONS D'ÉVALUATION
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Évalue le modèle sur le jeu de test et retourne des métriques détaillées.
    
    Arguments:
        model: Modèle Keras entraîné
        X_test: Images de test
        y_test: Labels de test (one-hot encoded)
    
    Returns:
        dict avec les métriques: accuracy, precision, recall, f1, loss
    """
    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Loss et accuracy globales
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Métriques par classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_samples': len(y_test)
    }
    
    print("\n" + "=" * 60)
    print("RESULTATS DE L'EVALUATION")
    print("=" * 60)
    print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Loss:      {loss:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n   Échantillons de test: {len(y_test)}")
    print("=" * 60)
    
    return metrics


def get_classification_report(model, X_test: np.ndarray, y_test: np.ndarray) -> str:
    """
    Génère un rapport de classification détaillé par classe.
    
    Arguments:
        model: Modèle Keras entraîné
        X_test: Images de test
        y_test: Labels (one-hot)
    
    Returns:
        str: Rapport de classification formaté
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Générer le rapport avec les noms de classes
    report = classification_report(
        y_true, y_pred,
        target_names=[f"{i}: {name[:15]}" for i, name in enumerate(CLASS_NAMES)],
        digits=3
    )
    
    print("\n" + "=" * 80)
    print("RAPPORT DE CLASSIFICATION PAR CLASSE")
    print("=" * 80)
    print(report)
    
    return report


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray,
                          save_path: str = None, normalize: bool = True):
    """
    Trace la matrice de confusion.
    
    Arguments:
        model: Modèle Keras
        X_test, y_test: Données de test
        save_path: Chemin pour sauvegarder
        normalize: Normaliser les valeurs (pourcentages)
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculer la matrice
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matrice de Confusion (Normalisée)'
    else:
        fmt = 'd'
        title = 'Matrice de Confusion'
    
    # Créer la figure
    plt.figure(figsize=(16, 14))
    
    # Heatmap avec seaborn
    sns.heatmap(
        cm, annot=False, fmt=fmt, cmap='Blues',
        xticklabels=range(NUM_CLASSES),
        yticklabels=range(NUM_CLASSES),
        cbar_kws={'label': 'Proportion' if normalize else 'Nombre'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vérité terrain', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardee: {save_path}")
    
    plt.close()


def plot_predictions(model, X_test: np.ndarray, y_test: np.ndarray,
                     n_samples: int = 16, correct: bool = True,
                     save_path: str = None):
    """
    Affiche une grille de prédictions (correctes ou incorrectes).
    
    Arguments:
        model: Modèle Keras
        X_test, y_test: Données de test
        n_samples: Nombre d'exemples à afficher
        correct: Si True, affiche les prédictions correctes, sinon les erreurs
        save_path: Chemin pour sauvegarder
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Trouver les indices selon le critère
    if correct:
        indices = np.where(y_pred == y_true)[0]
        title = 'Exemples de Prédictions Correctes'
        color = 'green'
    else:
        indices = np.where(y_pred != y_true)[0]
        title = 'Exemples d\'Erreurs de Prédiction'
        color = 'red'
    
    # Sélectionner aléatoirement
    n_samples = min(n_samples, len(indices))
    selected = np.random.choice(indices, n_samples, replace=False)
    
    # Grille de visualisation
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, idx in enumerate(selected):
        ax = axes[i]
        
        # Image
        ax.imshow(X_test[idx])
        
        # Titre avec prédiction et vérité
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        confidence = y_pred_proba[idx, pred_label] * 100
        
        if correct:
            ax.set_title(f'Classe {true_label}\nConf: {confidence:.1f}%', fontsize=9, color=color)
        else:
            ax.set_title(f'Vrai: {true_label}\nPrédit: {pred_label} ({confidence:.1f}%)',
                        fontsize=9, color=color)
        
        ax.axis('off')
    
    # Cacher les axes vides
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', color=color)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions sauvegardees: {save_path}")
    
    plt.close()


def analyze_top_errors(model, X_test: np.ndarray, y_test: np.ndarray, top_n: int = 10):
    """
    Analyse les erreurs les plus fréquentes.
    
    Identifie les confusions les plus courantes entre classes.
    
    Arguments:
        model: Modèle Keras
        X_test, y_test: Données de test
        top_n: Nombre de confusions à afficher
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Mettre les valeurs diagonales à 0 pour ne garder que les erreurs
    np.fill_diagonal(cm, 0)
    
    # Trouver les top erreurs
    errors = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                errors.append((cm[i, j], i, j))
    
    errors.sort(reverse=True)
    
    print("\n" + "=" * 70)
    print("ANALYSE DES ERREURS LES PLUS FREQUENTES")
    print("=" * 70)
    print(f"\n{'Nb erreurs':<12} {'Vrai':<6} {'Prédit':<6} {'Description'}")
    print("-" * 70)
    
    for count, true_class, pred_class in errors[:top_n]:
        true_name = CLASS_NAMES[true_class][:20]
        pred_name = CLASS_NAMES[pred_class][:20]
        print(f"{count:<12} {true_class:<6} {pred_class:<6} {true_name} -> {pred_name}")
    
    print("=" * 70)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def run_full_evaluation(model_path: str = None):
    """
    Exécute une évaluation complète du modèle.
    
    Arguments:
        model_path: Chemin vers le modèle (défaut: models/best_model.keras)
    """
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE DU MODELE GTSRB")
    print("=" * 70)
    
    # Chemin par défaut
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'best_model.keras')
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print("   Veuillez d'abord entraîner le modèle avec train.py")
        return None
    
    # Charger le modèle
    print(f"\nChargement du modele: {model_path}")
    model = load_model(model_path)
    
    # Charger les données
    print("\nChargement des donnees de test...")
    _, _, X_test, _, _, y_test = prepare_data(DATA_PATH)
    
    # Créer le dossier figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Métriques globales
    print("\n" + "-" * 40)
    print("1. Metriques globales")
    print("-" * 40)
    metrics = evaluate_model(model, X_test, y_test)
    
    # 2. Rapport par classe
    print("\n" + "-" * 40)
    print("2. Rapport de classification")
    print("-" * 40)
    report = get_classification_report(model, X_test, y_test)
    
    # 3. Matrice de confusion
    print("\n" + "-" * 40)
    print("3. Matrice de confusion")
    print("-" * 40)
    cm_path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(model, X_test, y_test, save_path=cm_path)
    
    # 4. Exemples de prédictions correctes
    print("\n" + "-" * 40)
    print("4. Exemples de predictions correctes")
    print("-" * 40)
    correct_path = os.path.join(FIGURES_DIR, 'correct_predictions.png')
    plot_predictions(model, X_test, y_test, correct=True, save_path=correct_path)
    
    # 5. Exemples d'erreurs
    print("\n" + "-" * 40)
    print("5. Exemples d'erreurs")
    print("-" * 40)
    errors_path = os.path.join(FIGURES_DIR, 'error_predictions.png')
    plot_predictions(model, X_test, y_test, correct=False, save_path=errors_path)
    
    # 6. Analyse des top erreurs
    print("\n" + "-" * 40)
    print("6. Analyse des erreurs frequentes")
    print("-" * 40)
    analyze_top_errors(model, X_test, y_test)
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(MODELS_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetriques sauvegardees: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE TERMINEE")
    print("=" * 70)
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Évaluation du modèle GTSRB')
    parser.add_argument('--model', type=str, default=None,
                       help='Chemin vers le modèle à évaluer')
    
    args = parser.parse_args()
    
    run_full_evaluation(args.model)
