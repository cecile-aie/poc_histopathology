
---

# 🧬 Baseline Morphologique & Statistical Shape Model (SSM) — README Technique

*Notebook : `04_baseline_ssm.ipynb`*

Ce document décrit **la configuration technique** de la baseline morphologique du projet P9 (histopathologie), incluant :

* l’extraction de descripteurs shape+texture,
* la construction du modèle statistique de forme (**SSM**),
* l’entraînement des classifieurs morphologiques,
* les analyses inter-classes,
* et le rôle du SSM dans l’évaluation future des images synthétiques (PixCell / cGAN).

Le but n’est pas de faire une justification scientifique complète, mais de **rendre explicite l’ensemble du pipeline mis en œuvre**, dans l’esprit du README cGAN.

---

# 1. Données & Préprocessing

## 1.1 Sources

* **Train morphologique** : `NCT-CRC-HE-100K`
* **Test indépendant** : `CRC-VAL-HE-7K`
* Split **anti-fuite** strict : aucune image des sources NCT ne doit apparaître dans l’évaluation.

## 1.2 Génération des masques binaires

Les masques sont dérivés des images RGB rescalées en 256×256.
Chaque masque représente la **zone de tissu utile**, extraite via :

* conversion HSV,
* seuillages sur V / S (tissu vs fond),
* opérations morphologiques simples (érosion + ouverture).

Les masques sont ensuite :

* convertis en booléens,
* centrés et padés à 256×256 pour homogénéité.

Sortie :
`mask: np.ndarray (256×256 bool)`

Ces masques constituent l’entrée de tout le module SSM.

---

# 2. Analyse morphologique (scikit-image)

## 2.1 Objectif

Évaluer si les tuiles possèdent un **signal discriminant basé uniquement sur forme/texture**, sans CNN ni deep learning.

## 2.2 Extraction de features

Pour chaque image/mask, le notebook calcule un vecteur de features regroupés en 3 familles :

### 🔹 2.2.1 Descripteurs de forme (Shape)

* Moments de Hu (7 valeurs)
* Aire normale
* Compacité
* Nombre d’objets connectés
* Ratio des composantes morphologiques
* Eccentricité, Extent, Solidity
* Largeur/hauteur de la bounding box

### 🔹 2.2.2 Texture GLCM

* Contrast
* Homogeneity
* Energy
* Dissimilarity
* Angular Second Moment

### 🔹 2.2.3 Features additionnels

* Variation locale (LaplacianVar)
* Tenengrad (netteté)
* Entropie lumineuse
* Ratio tissu/fond (contours HSV)

L’ensemble forme `df_morpho_ext` (quelques centaines de colonnes).

## 2.3 Classification

Deux classifieurs sont entraînés :

* **SVM RBF**
* **RandomForest**

Ceux-ci sont évalués sur `crc-val-he-7k` afin de garantir l’absence de fuite.

### 🔹 Résultats :

* **Accuracy ≈ 0.74**, **Macro-F1 ≈ 0.74**
* ADI/BACK : très faciles
* LYM/TUM : bien capturés
* MUC/STR/DEB : proches → confusions fréquentes

Ce bloc constitue la **baseline morphologique stricte**.

---

# 3. Statistical Shape Model (SSM)

## 3.1 Objectif

Modéliser **la variabilité des formes** à partir des masques binaires afin d’obtenir :

* une *forme moyenne* par classe,
* des *modes principaux* (axes PCA),
* une base pour comparer réel ↔ synthétique.

Le SSM ne cherche **pas** à classer : c’est un **outil d’exploration et d’explicabilité**.

## 3.2 Pipeline SSM (par classe)

### 🔹 1. Alignement des masques

Fonction : `align_masks()`

Opérations :

* calcul du barycentre du masque,
* recentrage dans un cadre 256×256,
* aucun redressement rotation/scale dans la version actuelle (POC linéaire).

### 🔹 2. Flatten

Chaque masque aligné devient un vecteur :

```
flat_mask = mask.reshape(256*256)
```

### 🔹 3. PCA

On applique une PCA classique :

```
PCA(n_components=10)
```

Mesures extraites :

* `mean_shape`
* `components_` (modes)
* `explained_variance_ratio_`

Résultat observé :

* Variance cumulée des 3 premières composantes : **≈ 1.5–2.5%**
  → Indique que les formes présentent une grande variabilité non linéaire.

### 🔹 4. Visualisation et sauvegarde

Pour chaque classe :

* forme moyenne,
* premier mode (épaisseur / extension),
* deuxième mode (courbure / dispersion),
* projection global real vs synth.

Les modèles sont sauvegardés dans
`/workspace/models/04_baseline_ssm/*.npy`


---

# 4. Analyse inter-classes via SSM

## 4.1 PCA global (toutes classes)

On peut projeter les masques de plusieurs classes dans l’espace SSM afin de visualiser les séparations morphologiques.

Observations :

* chevauchement massif dans l’espace PCA,
* certaines classes (LYM, ADI) forment des sous-nuages plus compacts,
* TUM/MUC/STR/DEB se superposent fortement.

➡️ **Le SSM linéaire ne permet pas de séparer les classes** → cohérent, car c'est un *modèle de forme* et non de texture.

## 4.2 Utilité future

Le SSM n’a **pas** vocation à être un classifieur mais un :

### ✔️ outil de mesure morphologique

Calcule la distance d’une forme (réelle ou synthétique) à la forme moyenne + modes d’une classe.

### ✔️ outil de comparaison GAN

Projection des masques synthétiques générés par PixCell ou le cGAN dans l’espace SSM →
détection :

* d’anomalies de forme,
* de biais morphologiques,
* de sous-types mal couverts.

→ **Boussole morphologique du pipeline GAN**.

---

# 5. Limitations & Garde-fous

## 5.1 Limitations du SSM linéaire

* ne gère pas les rotations,
* ne gère pas les changements d’échelle,
* modèle incapable de capturer les déformations non linéaires (commun en histopatho),
* variance expliquée très faible → les modes PCA sont faibles.

## 5.2 Solutions prévues

* alignement complet **Procrustes (translation + rotation + scale)**
* représentations non linéaires :

  * autoencoder de forme,
  * PCA sur **distance transform**,
  * contours paramétriques (splines),
  * UMAP/t-SNE dédiés masque.

Ces extensions permettront un SSM plus riche, mais non nécessaires dans ce POC.

---

# 6. Métriques & Outputs

## 6.1 Fichiers générés

### 📁 `df_morpho_ext.csv`

Tableau complet des features morphologiques utilisés par SVM/RF.

### 📁 `/workspace/models/04_baseline_ssm/*.npy`

* `mean.npy` : forme moyenne
* `components.npy` : modes PCA
* `explained_var.npy` : variance par composante
* `pca_model.pkl` : modèle PCA sérialisé

### 📁 `/workspace/samples/04_baseline_ssm/`

Figures :

* formes moyennes,
* modes ±3σ,
* projections inter-classes,
* projections réel vs Synth (utilisé plus tard pour PixCell).

## 6.2 Logs

Le notebook génère :

* tableaux F1-score par classe,
* matrices de confusion,
* tableaux de variance PCA.

---

# 7. Résumé “one-liner” du setup

> **Baseline morphologique = extraction shape+texture pour classification traditionnelle (~0.74 F1).
> Baseline SSM = PCA sur masques alignés pour modéliser la variabilité de forme, outil d’explicabilité morphologique indispensable pour l’analyse des images synthétiques GAN / diffusion.**

---

# 8. Structure du dossier

```
04_baseline_ssm/
│
├── preprocessing/               # Génération / alignement des masques
├── df_morpho_ext.csv           # Features morphologiques
├── models/                     # SSM sauvegardés (PCA, modes, forme moyenne)
├── figures/                    # Visualisations
└── README.md                   # (ce fichier)
```

---

