
-----

# 🔬 Generative AI for Digital Pathology (P9 POC)

> **Génération d'images histologiques réalistes : Comparatif GAN vs Diffusion (PixCell)**

[](https://pytorch.org/) [](https://www.docker.com/) [](https://streamlit.io/) [](https://www.google.com/search?q=LICENSE)

Ce projet est une **Preuve de Concept (POC)** visant à démontrer comment l'IA générative peut résoudre les problèmes de rareté de données et de biais colorimétrique en histopathologie. Nous explorons et comparons deux architectures majeures pour générer des tissus colorectaux synthétiques : **StyleGAN2-ADA** (rapide) et **PixCell + Adapter/LoRA** (haute fidélité).

-----

## 🎯 Objectifs

L'histopathologie digitale souffre d'un manque de données annotées pour les classes rares et d'une forte variabilité technique (coloration H\&E). Ce projet vise à :

1.  **Générer des images synthétiques** biologiquement plausibles (9 classes de tissus).
2.  **Comparer deux paradigmes** : L'approche adversariale (GAN) vs l'approche par Diffusion (Foundation Models).
3.  **Valider l'utilité clinique** : Mesurer si l'ajout de ces images améliore les performances d'un classifieur de diagnostic (*Downstream Task*).

-----

## 🚀 Installation & Démarrage (Docker)

Tout l'environnement est conteneurisé. Pas besoin de gérer les versions de CUDA ou PyTorch à la main \!

### Pré-requis

  * Docker & Docker Compose
  * Drivers NVIDIA (pour le support GPU)

### Lancement

L'image Docker inclut **PyTorch 2.4, CUDA 12.4, Diffusers, Timm et Openslide**.

```bash
# 1. Construire et lancer le conteneur
docker-compose up --build

# 2. Accéder aux services :
# 📓 JupyterLab : http://localhost:8888
# 🎈 Streamlit  : http://localhost:8501
# 📈 TensorBoard: http://localhost:6006
```

-----

## 📂 Structure du Projet

L'architecture est modulaire pour séparer la logique de génération, l'évaluation et l'interface utilisateur.
Les dossiers de ressources et sorties (/models, /data, /outputs, /checkpoints) sont à placer à la racine.

```text
.
├── p9dg/                  # 📦 Core Package
│   └── histo_dataset.py   # DataGenerator avec normalisation Vahadane & Filtrage Qualité
├── metrics/               # 📏 Métriques génériques
│   ├── cnn_eval.py        # Eval downstream (MobileNetV2 + Calibration)
│   └── fid_lpips_eval.py  # Calculateur batch FID/LPIPS
├── gan_metrics/           # 📐 Métriques spécifiques
│   └── duet_fid.py        # Calcul FID spécialisé avec backbone PathoDuet
├── scripts/               # ⚙️ Utilitaires Backend
│   └── dashboard_backend.py # Logique de l'application Streamlit
├── utils/
│   └── class_mappings.py  # Mappings classes (TUM, STR...) & couleurs
├── notebooks/             # 📓 Laboratoire d'expérimentation (détail ci-dessous)
├── streamlit_app.py       # 🎈 Application de démonstration
├── Dockerfile             # Définition de l'environnement
└── docker-compose.yml     # Orchestration des services
```
 
-----

## 📓 Guide des Notebooks

Les notebooks, situés dans le dossier `notebooks/`, tracent l'histoire complète du projet, de l'exploration des données à la validation finale.

### 🧹 1. Préparation & Données

  * **`p9_EDA.ipynb`** : Analyse Exploratoire des Données (distributions, inspection visuelle).
  * **`p9_PREPROCESSING.ipynb`** : Pipeline de normalisation (Vahadane) et création des datasets nettoyés.
  * **`01_test_datagenerator.ipynb`** : Validation technique du `HistoDataset` et du `QualityFilter`.

### 📏 2. Baselines (Les juges de paix)

  * **`02_baseline_cnn.ipynb`** : Entraînement du classifieur MobileNetV2 sur données réelles (Référence).
  * **`03_baseline_radiomics.ipynb`** : Baseline non-profonde basée sur des features de texture (PyRadiomics).
  * **`04_baseline_ssm.ipynb`** : Baseline morphologique (Statistical Shape Model) pour valider la géométrie des formes.

### ⚡ 3. Modélisation GAN (StyleGAN2)

  * **`05_StyleGAN.ipynb`** : Premiers pas et tests d'entraînement inconditionnel.
  * **`06b_cGAN_IA.ipynb`** : Entraînement principal du **cGAN** (Conditionnel) avec augmentation ADA.

### 🎨 4. Modélisation Diffusion (PixCell)

  * **`07_Diffusion_model.ipynb`** : Prise en main de PixCell et du backbone UNI2-h (approche Gated).
  * **`08_LoRA_Adapter_fallback_UNET.ipynb`** : Tentative intermédiaire d'adaptation simplifiée (U-Net classique).
  * **`08_UNI2h_Adapter_PixCell_LoRA.ipynb`** : **Le modèle final**. Fine-tuning hybride (Adapter + LoRA) pour une fidélité maximale.

### 🧪 5. Validation & Métriques

  * **`test_metrics_fid_lpips.ipynb`** : Validation unitaire des calculateurs de métriques d'image.
  * **`test_metrics_downstream.ipynb`** : Validation du pipeline d'évaluation clinique (ECE, Brier Score).
  * **`02b_baseline_cnn_synth.ipynb`** : Expérience *Downstream* finale (Entraînement sur mix Réel + Synthétique).

### 🔍 6. Visualisation Latente

  * **`viz_embeddings_PathoDuet.ipynb`** : Projection UMAP des images via le backbone PathoDuet.
  * **`viz_embeddings_UNI.ipynb`** : Exploration des géodésiques et interpolations dans l'espace UNI2-h.

-----

## 🛠️ Stack Technique

  * **Core :** Python 3.10, PyTorch 2.4, CUDA 12.4
  * **GenAI :** `diffusers` (HuggingFace), `timm` (Vision Transformers), StyleGAN2-ADA (PyTorch impl.)
  * **Medical :** `openslide-python`, `torch-staintools` (Normalisation)
  * **Ops :** Docker, Nvidia Container Toolkit

-----

## 📚 Documentation

| Modèle | Description |
|--------|-------------|
| [🟣 cGAN](docs/cGAN_README.md) | Modèle StyleGAN2 + tête PathoDuet |
| [🔵 PixCell (Diffusion)](docs/pixcell_README.md) | Pipeline diffusion UNI2-h |
| [🟢 Radiomics](docs/baseline_radiomics_README.md) | Extraction PyRadiomics |
| [🟠 SSM](docs/baseline_ssm_README.md) | Modèles de forme statistiques |
| [⚫ CNN Baseline](docs/baseline_CNN.md) | MobileNetV2 classifier |


-----

## 📝 Auteurs & Crédits

Ce projet s'appuie sur de nombreux travaux de recherche, notamment :

  * **NCT-CRC-HE-100K** (Kather et al.) pour le dataset.
  * **PixCell** & **UNI** (Mahmood Lab et al.) pour les Foundation Models en pathologie.
  * **Pathoduet** (Shengyi Hua & al.) pour le backbone spécialisé en histopathologie (utilisé pour FID-Duet et tête de sortie du discriminateur cGAN)

*Projet réalisé dans le cadre du parcours Ingénieur IA OpenClassRooms (p9-Développez une preuve de concept).*
