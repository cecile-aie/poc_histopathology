# Statut d'implémentation du Dashboard Streamlit

## ✅ Fait

### 1. Structure de base
- ✅ Module backend `p9dg/dashboard_backend.py` créé avec toutes les fonctions utilitaires
- ✅ Application Streamlit `streamlit_app.py` créée avec structure 3 colonnes
- ✅ Configuration WCAG (couleurs, tailles de police, contraste)

### 2. Fonctionnalités implémentées
- ✅ Sélection des classes (avec option "All classes")
- ✅ Pool d'images réelles (1-20 par classe, exclusion du test set)
- ✅ Interface de génération (cGAN et PixCell)
- ✅ Galerie avec pagination (2x4 grid, 8 images par page)
- ✅ Onglets "Class preview" et "Real vs Synth"
- ✅ Slider de proportion synthétique (0, 20, 40, 60, 80, 100%)
- ✅ Interface d'évaluation CNN
- ✅ Interface de calcul FID/LPIPS

### 3. Accessibilité WCAG
- ✅ Couleurs avec contraste suffisant (> 4.5:1)
- ✅ Tailles de police accessibles (minimum 16px)
- ✅ Navigation au clavier supportée (Tab, Enter, Space)
- ✅ Labels clairs et descriptifs

## ⚠️ À compléter

### 1. Génération d'images (backend)
Les fonctions `generate_with_cgan()` et `generate_with_pixcell()` sont des placeholders.
**Action requise:**
- Lire les notebooks `06b_cGAN_IA.ipynb` et `07_Diffusion_model.ipynb`
- Extraire le code exact de génération
- Adapter pour générer 100 images par classe
- Sauvegarder en PNG dans la structure `outputs/synth/{generator_type}/{experiment_id}/{class_name}/`

### 2. Chargement des modèles
Les fonctions `load_cgan_model()` et `load_pixcell_model()` nécessitent:
- **cGAN**: Reconstruire l'architecture Generator depuis le notebook (StyleGAN-lite conditionnel)
- **PixCell**: Le chargement est partiellement fait, mais nécessite la configuration UNI-2h

### 3. Évaluation CNN
La fonction `evaluate_cnn_on_index()` est un placeholder.
**Action requise:**
- Utiliser `cnn_eval.py` comme référence
- Créer un DataLoader depuis le DataFrame de test
- Appliquer le même preprocessing que lors de l'entraînement
- Calculer accuracy, F1-macro, matrice de confusion

### 4. FID/LPIPS
La fonction `compute_fid_lpips()` utilise `FIDLPIPSEvaluator` mais nécessite:
- Vérifier que le mode unpaired fonctionne correctement
- Gérer les cas où il n'y a pas assez d'images

## 📝 Notes d'implémentation

### Structure des fichiers générés
```
outputs/synth/
  ├── cgan/
  │   └── {experiment_id}/
  │       ├── TUM/
  │       │   ├── sample_0.png
  │       │   ├── sample_1.png
  │       │   └── ...
  │       └── NORM/
  │           └── ...
  └── pixcell/
      └── {experiment_id}/
          └── ...
```

### Session state
- `real_pool`: Pool d'images réelles (RealImagePool)
- `generated_index`: Index des images générées {class_name: [GeneratedImageInfo]}
- `experiment_id`: ID de l'expérience actuelle
- `cnn_results`: Résultats de l'évaluation CNN
- `fid_lpips_results`: Résultats FID/LPIPS

### Paramètres fixes
- `N_TEST_PER_CLASS = 100` (fixé dans le code)
- `FID_REF_IMAGES_PER_CLASS = 200` (subset immuable)
- `IMAGE_SIZE = 256` (taille des images)

## 🚀 Prochaines étapes

1. **Compléter la génération cGAN:**
   - Extraire le code Generator depuis `06b_cGAN_IA.ipynb`
   - Implémenter la génération conditionnelle avec class_id
   - Tester avec quelques classes

2. **Compléter la génération PixCell:**
   - Extraire le code depuis `07_Diffusion_model.ipynb`
   - Implémenter le conditionnement avec UNI-2h embeddings
   - Tester la génération

3. **Compléter l'évaluation CNN:**
   - Créer un DataLoader personnalisé depuis le DataFrame
   - Appliquer le preprocessing (normalisation ImageNet, Vahadane si nécessaire)
   - Calculer les métriques

4. **Tester end-to-end:**
   - Générer quelques images
   - Évaluer avec CNN
   - Calculer FID/LPIPS
   - Vérifier l'affichage dans la galerie

## 🔍 Points d'attention

- **Mémoire GPU**: La génération de 100 images par classe peut être lourde
- **Temps de calcul**: FID/LPIPS peut prendre plusieurs secondes par classe
- **Gestion d'erreurs**: Bien gérer les cas où une classe échoue (fallback)
- **Cache Streamlit**: Les modèles sont mis en cache avec `@st.cache_resource`



