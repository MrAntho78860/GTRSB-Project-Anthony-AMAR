"""
Module de prétraitement des données GTSRB
==========================================

Ce module fournit toutes les fonctions nécessaires pour charger,
prétraiter et préparer les données du German Traffic Sign Recognition Benchmark
pour l'entraînement d'un réseau de neurones convolutif.

Auteur: Projet académique CNRS-FIDLE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

# Taille des images après redimensionnement
IMG_SIZE = 32

# Nombre de classes (panneaux routiers allemands)
NUM_CLASSES = 43

# Proportions pour la séparation des données
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# NOMS DES CLASSES DE PANNEAUX
# =============================================================================

CLASS_NAMES = [
    "Limite 20 km/h",           # 0
    "Limite 30 km/h",           # 1
    "Limite 50 km/h",           # 2
    "Limite 60 km/h",           # 3
    "Limite 70 km/h",           # 4
    "Limite 80 km/h",           # 5
    "Fin limite 80 km/h",       # 6
    "Limite 100 km/h",          # 7
    "Limite 120 km/h",          # 8
    "Dépassement interdit",     # 9
    "Dépassement interdit 3.5t",# 10
    "Priorité à droite",        # 11
    "Route prioritaire",        # 12
    "Cédez le passage",         # 13
    "Stop",                     # 14
    "Circulation interdite",    # 15
    "Interdit aux 3.5t",        # 16
    "Sens interdit",            # 17
    "Danger",                   # 18
    "Virage gauche",            # 19
    "Virage droite",            # 20
    "Double virage",            # 21
    "Route cahoteuse",          # 22
    "Route glissante",          # 23
    "Chaussée rétrécie droite", # 24
    "Travaux",                  # 25
    "Feux tricolores",          # 26
    "Piétons",                  # 27
    "Passage enfants",          # 28
    "Passage cyclistes",        # 29
    "Neige/verglas",            # 30
    "Passage animaux",          # 31
    "Fin restrictions",         # 32
    "Tourner à droite",         # 33
    "Tourner à gauche",         # 34
    "Tout droit",               # 35
    "Tout droit ou droite",     # 36
    "Tout droit ou gauche",     # 37
    "Serrer à droite",          # 38
    "Serrer à gauche",          # 39
    "Rond-point",               # 40
    "Fin interdit dépasser",    # 41
    "Fin interdit dépasser 3.5t"# 42
]


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

def get_default_data_path(base_dir: str | None = None) -> str:
    """
    Détermine le chemin par défaut du dataset GTSRB.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = [
        os.path.join(base_dir, 'data', 'GTSRB', 'Training'),
        os.path.join(base_dir, 'data', 'GTRSB-Training_fixed', 'GTSRB', 'Training'),
        os.path.join(base_dir, 'GTSRB-Training_fixed', 'GTSRB', 'Training')
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def load_gtsrb_data(data_path: str, verbose: bool = True) -> tuple:
    """
    Charge le dataset GTSRB depuis le chemin spécifié.
    
    Cette fonction parcourt les 43 dossiers de classes et charge toutes les images
    avec leurs labels correspondants.
    
    Arguments:
        data_path: Chemin vers le dossier 'Training' du dataset GTSRB
        verbose: Afficher la progression du chargement
    
    Returns:
        tuple (images, labels): 
            - images: liste d'arrays numpy (images brutes)
            - labels: liste d'entiers (classes 0-42)
    
    Exemple:
        >>> images, labels = load_gtsrb_data('data/GTSRB/Training')
        >>> print(f"Charge: {len(images)} images")
    """
    import csv
    
    images = []
    labels = []
    
    # Itérateur avec barre de progression
    class_range = tqdm(range(NUM_CLASSES), desc="Chargement des classes") if verbose else range(NUM_CLASSES)
    
    for class_id in class_range:
        # Chemin vers le dossier de la classe
        class_folder = os.path.join(data_path, format(class_id, '05d'))
        annotation_file = os.path.join(class_folder, f'GT-{format(class_id, "05d")}.csv')
        
        if not os.path.exists(annotation_file):
            print(f"Avertissement: fichier d'annotations introuvable: {annotation_file}")
            continue
        
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header
            
            for row in reader:
                img_path = os.path.join(class_folder, row[0])
                
                if os.path.exists(img_path):
                    # Charger l'image avec PIL
                    img = Image.open(img_path)
                    images.append(np.array(img))
                    labels.append(int(row[7]))  # La 8ème colonne contient le label
    
    if verbose:
        print(f"\nChargement termine: {len(images)} images, {NUM_CLASSES} classes")
        
    return images, labels


# =============================================================================
# FONCTIONS DE PRÉTRAITEMENT
# =============================================================================

def preprocess_images(images: list, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Prétraite les images pour l'entraînement du CNN.
    
    Opérations effectuées:
    1. Redimensionnement à une taille uniforme (32x32 par défaut)
    2. Normalisation des valeurs de pixels [0, 255] -> [0, 1]
    
    Arguments:
        images: Liste d'images (arrays numpy de tailles variables)
        target_size: Taille cible (carrée)
    
    Returns:
        np.ndarray de forme (N, target_size, target_size, 3) avec valeurs [0, 1]
    
    Note:
        La normalisation [0, 1] est préférée à [-1, 1] pour les réseaux
        utilisant ReLU, car elle évite les valeurs négatives en entrée.
    """
    processed = []
    
    for img in tqdm(images, desc="Prétraitement des images"):
        # Convertir en PIL Image si nécessaire
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        
        # Redimensionner à la taille cible
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convertir en RGB si nécessaire
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Convertir en array numpy
        img_array = np.array(pil_img, dtype=np.float32)
        
        # Normaliser [0, 255] -> [0, 1]
        img_array = img_array / 255.0
        
        processed.append(img_array)
    
    return np.array(processed)


def encode_labels(labels: list, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Encode les labels en format one-hot pour la classification multi-classes.
    
    Arguments:
        labels: Liste d'entiers représentant les classes (0-42)
        num_classes: Nombre total de classes
    
    Returns:
        np.ndarray de forme (N, num_classes) avec encodage one-hot
    
    Exemple:
        >>> labels = [0, 1, 2]
        >>> encoded = encode_labels(labels, num_classes=43)
        >>> print(encoded.shape)  # (3, 43)
    """
    labels_array = np.array(labels, dtype=np.int32)
    return to_categorical(labels_array, num_classes=num_classes)


def split_data(X: np.ndarray, y: np.ndarray, 
               train_ratio: float = TRAIN_RATIO,
               val_ratio: float = VAL_RATIO,
               random_state: int = 42) -> tuple:
    """
    Sépare les données en ensembles d'entraînement, validation et test.
    
    La stratification est utilisée pour garantir une distribution équilibrée
    des classes dans chaque ensemble.
    
    Arguments:
        X: Images prétraitées (N, H, W, C)
        y: Labels encodés en one-hot (N, num_classes)
        train_ratio: Proportion pour l'entraînement (défaut: 0.70)
        val_ratio: Proportion pour la validation (défaut: 0.15)
        random_state: Graine aléatoire pour la reproductibilité
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Note:
        test_ratio = 1 - train_ratio - val_ratio (défaut: 0.15)
    """
    # Convertir one-hot en labels pour la stratification
    y_labels = np.argmax(y, axis=1)
    
    # Première séparation: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(1 - train_ratio),
        stratify=y_labels,
        random_state=random_state
    )
    
    # Recalculer les proportions pour la deuxième séparation
    temp_labels = np.argmax(y_temp, axis=1)
    val_proportion = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    
    # Deuxième séparation: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_proportion),
        stratify=temp_labels,
        random_state=random_state
    )
    
    print("\nRepartition des donnees:")
    print(f"   Entrainement: {len(X_train)} images ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation:   {len(X_val)} images ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:         {len(X_test)} images ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# FONCTION PRINCIPALE DE PRÉPARATION
# =============================================================================

def prepare_data(data_path: str, img_size: int = IMG_SIZE) -> tuple:
    """
    Pipeline complet de préparation des données.
    
    Enchaîne toutes les étapes: chargement, prétraitement, encodage, séparation.
    
    Arguments:
        data_path: Chemin vers le dossier Training du GTSRB
        img_size: Taille des images après redimensionnement
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("=" * 60)
    print("PREPARATION DES DONNEES GTSRB")
    print("=" * 60)
    
    # 1. Charger les images brutes
    print("\nEtape 1/4: Chargement des donnees...")
    images, labels = load_gtsrb_data(data_path)
    
    # 2. Prétraiter les images
    print("\nEtape 2/4: Pretraitement des images...")
    X = preprocess_images(images, target_size=img_size)
    print(f"   Forme des données: {X.shape}")
    
    # 3. Encoder les labels
    print("\nEtape 3/4: Encodage des labels...")
    y = encode_labels(labels)
    print(f"   Forme des labels: {y.shape}")
    
    # 4. Séparer les données
    print("\nEtape 4/4: Separation train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print("\n" + "=" * 60)
    print("DONNEES PRETES POUR L'ENTRAINEMENT")
    print("=" * 60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def visualize_class_distribution(labels: list, save_path: str = None):
    """
    Affiche la distribution des classes dans le dataset.
    
    Crée un histogramme montrant le nombre d'images par classe,
    utile pour identifier les déséquilibres dans le dataset.
    
    Arguments:
        labels: Liste des labels (entiers 0-42)
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    plt.figure(figsize=(14, 6))
    
    # Compter les occurrences de chaque classe
    unique, counts = np.unique(labels, return_counts=True)
    
    # Créer le graphique
    bars = plt.bar(unique, counts, color='steelblue', edgecolor='navy', alpha=0.8)
    
    # Personnalisation
    plt.xlabel('Classe de panneau', fontsize=12)
    plt.ylabel('Nombre d\'images', fontsize=12)
    plt.title('Distribution des classes dans le dataset GTSRB', fontsize=14, fontweight='bold')
    plt.xticks(range(NUM_CLASSES), rotation=90, fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter la moyenne
    mean_count = np.mean(counts)
    plt.axhline(y=mean_count, color='red', linestyle='--', label=f'Moyenne: {mean_count:.0f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardee: {save_path}")
    
    plt.show()


def visualize_sample_images(images: list, labels: list, n_per_class: int = 1, save_path: str = None):
    """
    Affiche une grille d'exemples d'images par classe.
    
    Arguments:
        images: Liste d'images prétraitées
        labels: Liste de labels correspondants
        n_per_class: Nombre d'exemples par classe
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    fig, axes = plt.subplots(7, 7, figsize=(14, 14))
    axes = axes.flatten()
    
    # Trouver un exemple de chaque classe
    labels_array = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    if len(labels_array.shape) > 1:  # Si one-hot encoded
        labels_array = np.argmax(labels_array, axis=1)
    
    for class_id in range(NUM_CLASSES):
        # Trouver les indices de cette classe
        indices = np.where(labels_array == class_id)[0]
        
        if len(indices) > 0:
            idx = indices[0]  # Prendre le premier exemple
            
            ax = axes[class_id]
            
            # Gérer les images normalisées ou non
            img = images[idx]
            if img.max() <= 1.0:
                ax.imshow(img)
            else:
                ax.imshow(img.astype(np.uint8))
            
            ax.set_title(f'{class_id}', fontsize=9)
            ax.axis('off')
    
    # Cacher les axes supplémentaires
    for i in range(NUM_CLASSES, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Exemples d\'images par classe', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardee: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN (pour tests)
# =============================================================================

if __name__ == "__main__":
    data_path = get_default_data_path()
    print(f"Chemin des donnees: {data_path}")

    if os.path.exists(data_path):
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)

        print("\nDimensions finales:")
        print(f"   X_train: {X_train.shape}")
        print(f"   X_val:   {X_val.shape}")
        print(f"   X_test:  {X_test.shape}")
    else:
        print(f"Chemin non trouve: {data_path}")
