# Baseline radiomique sur tuiles histopathologiques (PyRadiomics)

Ce notebook met en place une baseline **radiomique** pour la classification des tuiles histopathologiques RGB issues du dataset **NCT-CRC-HE-100K** (train) et **CRC-VAL-HE-7K** (test).  
L’objectif est de construire une représentation tabulaire de features radiomiques à partir des images, puis de comparer la performance d’un classifieur classique à la baseline CNN.

---

## 1. Objectifs du notebook

- Extraire des **descripteurs radiomiques** à partir des tuiles RGB pré-traitées.
- Constituer un **jeu de données tabulaire** (features radiomiques + labels de classes).
- Entraîner et évaluer plusieurs **classifieurs scikit-learn**.
- Comparer les performances à la **baseline CNN MobileNet** (référence downstream).

---

## 2. Jeu de données et pré-traitement

- **Jeux de données :**
  - Train : tuiles NCT-CRC-HE-100K, équilibrées à **300 images / classe**.
  - Test : tuiles CRC-VAL-HE-7K, sous-échantillonnées à **60 images / classe**.
- **Pré-traitement des images :**
  - Tuiles au format **256×256 RGB**.
  - Normalisation de couleur **Vahadane** (cohérence avec la baseline CNN).
  - Gestion de la qualité d’image et chargement via `HistoDataset` (mêmes règles que pour le CNN).
- **ROI / masque :**
  - La ROI correspond à **toute la tuile**, avec une fine bordure mise à 0 pour stabiliser PyRadiomics.
  - Gestion des particularités 2D/3D : conversion minimale pour respecter les attentes de la librairie.

---

## 3. Extraction de features radiomiques (PyRadiomics)

L’extraction est réalisée avec la bibliothèque **[PyRadiomics](https://pyradiomics.readthedocs.io)** sur chaque canal **R, G, B**.

### 3.1 Configuration principale

- Paramètres clés :
  - `binWidth = 15` (quantification des niveaux de gris).
  - `force2D = True` (calcul en 2D sur la tuile).
  - Distances GLCM : `distances = [1, 2, 3, 4, 5]`.
- Types d’images :
  - `Original`, `LoG(σ=0.5–3.0)`, `Wavelet`, `Square`, `Logarithm`, `Gradient`, etc.
- Classes de features activées :
  - **`firstorder`**
  - **`glcm`** (Gray Level Co-occurrence Matrix)
  - **`glszm`** (Gray Level Size Zone Matrix)
  - **`glrlm`** (Gray Level Run Length Matrix)
  - **`ngtdm`** (Neighbourhood Gray Tone Difference Matrix)
  - **`gldm`** (Gray Level Dependence Matrix)

Chaque tuile est donc décrite par un **vecteur riche de centaines de descripteurs radiomiques**, combinant intensités, textures et structures à plusieurs échelles.

### 3.2 Post-traitement des features

- Suppression des colonnes :
  - constantes,
  - entièrement NaN,
  - contenant des valeurs infinies.
- Normalisation et filtrage :
  - on conserve un ensemble de **features non redondants et exploitables**.
- Sauvegarde :
  - `radiomics_train.parquet`
  - `radiomics_test.parquet`

Ces fichiers servent ensuite d’entrée à la partie classification scikit-learn.

---

## 4. Modèle de classification (scikit-learn)

### 4.1 Pipeline

Construction d’un pipeline scikit-learn standard :

1. `VarianceThreshold` – suppression des features quasi constantes.
2. `StandardScaler` – centrage/réduction des descripteurs.
3. Classifieur supervisé (modèle linéaire ou arbre).

### 4.2 Modèles testés

Mini-recherche sur trois modèles simples :

- **Régression logistique**
  - pénalisation L2, `solver='saga'`, `max_iter=4000`.
- **SVM linéaire**
  - `LinearSVC`, `max_iter=4000`.
- **Random Forest**
  - 200 arbres, `max_depth=None`.

### 4.3 Critères d’évaluation

- **Accuracy globale** sur le test set.
- **Recall de la classe TUM** (priorité clinique à la détection des régions tumorales).
- Sorties du notebook :
  - **rapport de classification** complet,
  - **matrice de confusion** sur les 9 classes,
  - sauvegarde du meilleur pipeline :
    - `models/radiomics_best_<Model>.joblib`.

---

## 5. Principaux résultats et interprétation

- La baseline radiomique parvient à une **séparation non triviale** des 9 classes :
  - certaines classes “faciles” (fonds, artefacts très distincts) sont bien identifiées.
- Cependant :
  - l’**accuracy globale** reste **nettement inférieure** à celle de la baseline CNN MobileNet,
  - le **rappel sur la classe TUM** est limité,
  - les **patterns histologiques fins** restent souvent confondus entre classes.

**Conclusion :**

- Les descripteurs radiomiques fournissent une baseline :
  - **interprétable**,  
  - utile pour comprendre quels types de signaux (intensité, texture, zones homogènes, etc.) sont discriminants.
- Mais ils sont **insuffisants** pour capturer toute la complexité morphologique des tuiles histopathologiques RGB.  
  Cela justifie :
  - l’usage d’un **CNN** comme baseline essentielle,
  - puis l’introduction de modèles **génératifs (GAN, diffusion)** pour enrichir l’espace des données.

---

## 6. Annexe – Classes de features PyRadiomics utilisées

Cette section résume le sens des six familles de features activées dans la configuration PyRadiomics.

### 6.1 `firstorder` – Statistiques du premier ordre

Features basées sur l’**histogramme** des intensités, **sans relation spatiale** entre pixels.

- Mesurent : la **distribution globale** des intensités.
- Exemples :
  - Mean, Median, Minimum, Maximum
  - Standard Deviation (hétérogénéité globale)
  - Skewness (asymétrie)
  - Kurtosis (aplatissement)

---

### 6.2 `glcm` – Gray Level Co-occurrence Matrix

La GLCM quantifie la **fréquence de paires de pixels** d’intensités $(i, j)$ séparés par une certaine **distance** et un certain **angle**.

- Mesurent : la **texture** via les relations spatiales entre niveaux de gris.
- Points clés :
  - quantification des niveaux de gris via `binWidth` (ici 15),
  - distances GLCM `[1–5]` → textures à plusieurs échelles.
- Exemples :
  - Contrast (texture rugueuse vs lisse),
  - Energy (uniformité),
  - Homogeneity (proximité à la diagonale de la GLCM),
  - Correlation (dépendance linéaire entre pixels voisins).

---

### 6.3 `glszm` – Gray Level Size Zone Matrix

Quantifie les **zones** (groupes de pixels connectés) d’une **intensité uniforme** donnée et d’une **taille particulière**.

- Mesurent : la **taille** et l’**homogénéité** des régions d’intensité uniforme.
- Exemples :
  - Small Area Emphasis (petites zones),
  - Large Area Emphasis (grandes zones),
  - Gray Level Variance (variabilité des niveaux de gris des zones),
  - Zone Size Variance (variabilité des tailles de zones).

---

### 6.4 `glrlm` – Gray Level Run Length Matrix

Quantifie les **runs** : séquences de pixels adjacents, dans une direction donnée, partageant la **même intensité**.

- Mesurent : la texture **directionnelle**, liée à des structures linéaires ou allongées.
- Exemples :
  - Short Run Emphasis (runs courts → texture fine),
  - Long Run Emphasis (runs longs → texture grossière),
  - Gray Level Non-Uniformity (variabilité des niveaux de gris des runs),
  - Run Length Non-Uniformity (variabilité des longueurs de runs).

---

### 6.5 `ngtdm` – Neighbourhood Gray Tone Difference Matrix

Mesure la **différence entre l’intensité d’un pixel** et l’**intensité moyenne de son voisinage**.

- Mesurent : l’**hétérogénéité locale** et les petites variations de texture.
- Exemples :
  - Coarseness (texture grossière vs fine),
  - Contrast (force des transitions d’intensité),
  - Busyness (degré de “activité” de l’image),
  - Complexity (complexité structurelle).

---

### 6.6 `gldm` – Gray Level Dependence Matrix

Quantifie les **groupes de pixels “dépendants”**, c’est-à-dire présentant des intensités similaires au pixel central dans un rayon donné.

- Mesurent : la **dépendance locale** entre pixels d’intensité proche (zones homogènes).
- Exemples :
  - Small Dependence Emphasis (petits groupes),
  - Large Dependence Emphasis (grands groupes),
  - Dependence Non-Uniformity (variabilité des dépendances),
  - Gray Level Variance (variabilité des niveaux de gris au sein des dépendances).

---

### 6.7 Rôle des filtres d’image

Les features sont calculées sur :

- **Image `Original`**
- **Images filtrées** :
  - `LoG` (Laplacian of Gaussian) : met en évidence les **bords** et petites structures.
  - `Wavelet` : décomposition en **fréquences spatiales** (horizontales, verticales, diagonales).
  - `Gradient` : variations locales d’intensité (bords).
  - `Logarithm`, `Square` : re-mapping de l’intensité pour amplifier ou compresser certaines plages.

En combinant ces **six familles de features** avec plusieurs **filtres d’image**, on obtient un vecteur de caractéristiques **très riche et multi-échelle**, adapté à l’analyse fine des textures histopathologiques, même si, dans ce projet, il reste moins performant qu’un CNN moderne pour la classification des 9 classes.

---
