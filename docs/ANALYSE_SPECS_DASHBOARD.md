# Analyse des spécifications du dashboard Streamlit

## ✅ Points clairs et bien définis

1. **Architecture générale** : 3 colonnes (config/génération, galerie, métriques)
2. **Modèles à utiliser** : MobileNetV2, cGAN, PixCell UNI2-h
3. **Dataset** : `data/CRC-VAL-HE-7K/` uniquement
4. **Structure de fichiers** : chemins bien définis

---

## ⚠️ Zones à clarifier

### 1. **Génération d'images synthétiques** ✅ CLARIFIÉ

#### 1.1. Conditionnement des générateurs ✅
- **Réponse** : 
  - Les images réelles sont dans `data/CRC-VAL-HE-7K/` avec sous-dossiers par classe
  - Références : notebooks `06b_cGAN_IA.ipynb` et `07_Diffusion_model.ipynb`
  - Utiliser `p9dg/utils/class_mappings.py` pour le mapping des classes
  - Le cGAN utilise `HistoDataset` avec `return_labels=True` et conditionnement via `class_id`
  - Les images réelles servent de conditionnement (référence) pour la génération

#### 1.2. Paramètres de génération ✅
- **Réponse** : 
  - POC donc **pas de paramètres à exposer** dans l'UI
  - Utiliser les paramètres par défaut des notebooks (seed fixe, etc.)

#### 1.3. Format de sortie des images générées ✅
- **Réponse** :
  - Format : **PNG**
  - Taille : **identique aux notebooks** (256x256 par défaut)
  - Structure : `workspace/outputs/synth/{generator_type}/{experiment_id}/{class_name}/sample_{k}.png`
  - Gestion d'erreurs : **Fallback si une classe échoue, continuer avec les autres classes**

### 2. **Pool d'images réelles** ✅ CLARIFIÉ

#### 2.1. Échantillonnage du pool ✅
- **Réponse** :
  - Le pool est **figé** tant que les classes sélectionnées et le nombre d'images par classe ne changent pas
  - Stocker le pool dans `st.session_state["real_pool"]` avec structure `{class_name: [list_of_paths]}`
  - Le pool est reconstruit uniquement si :
    - Les classes sélectionnées changent
    - Le nombre d'images par classe change (slider "Real images per class")

#### 2.2. Séparation train/test pour les images réelles ✅
- **Réponse** :
  - **Oui, exclure les images du pool de génération du test set** pour éviter le data leakage
  - Séparation claire :
    - Images pour génération : échantillonnées depuis `CRC-VAL-HE-7K/{class_name}/` (1-20 par classe selon le slider)
    - Images pour test : échantillonnées depuis le **reste** du dataset (excluant celles du pool de génération)

### 3. **Évaluation CNN (MobileNetV2)** ✅ CLARIFIÉ

#### 3.1. Architecture et préprocessing ✅
- **Réponse** :
  - `cnn_eval.py` est bien documenté et va chercher le modèle depuis `models/mobilenetv2_best.pt`
  - La construction du modèle est explicitée dans `02_baseline_cnn.ipynb`
  - **Important** : Sur le jeu de référence, on a déjà généré les prédictions, logits, ECE, temperature scaling, calibrage pour éviter de refaire l'inférence à chaque calcul de métrique pour un jeu synthétique
  - Réutiliser `cnn_eval.py` pour charger le modèle et calculer les métriques

#### 3.2. Mapping des classes ✅
- **Réponse** :
  - **Toujours se référer à `class_mappings.py`** pour le mapping des classes

#### 3.3. Taille du test set ✅ MODIFIÉ
- **Réponse** :
  - `N_TEST_PER_CLASS = 100` est **fixé dans le code** (pas un paramètre utilisateur)
  - Pas de slider pour définir cette valeur

#### 3.4. Mélange réel/synthétique ✅ MODIFIÉ IMPORTANT
- **Réponse** :
  - **Supprimer le slider "Synthetic images per real"** (nombre d'images synthétiques à générer)
  - **Générer toujours le maximum nécessaire** : 100 images synthétiques par classe (pour couvrir le cas 100% synthétique avec N_TEST_PER_CLASS=100)
  - Le slider de proportion reste : [0, 20, 40, 60, 80, 100] % de synthétique
  - **Afficher dans la colonne du milieu (galerie)** combien d'images synthétiques sont générées pour chaque classe lors de la génération

### 4. **Métriques FID/LPIPS** ✅ CLARIFIÉ

#### 4.1. Référence pour FID/LPIPS ✅
- **Réponse** :
  - **Ne pas utiliser toutes les images** de `CRC-VAL-HE-7K` (trop long, plusieurs secondes par classe)
  - Utiliser un **subset immuable par seed** : **200 images par classe**
  - Échantillonnage fixe avec seed pour reproductibilité

#### 4.2. Calcul global vs par classe ✅
- **Réponse** :
  - Référence : `test_metrics_fid_lpips.ipynb`
  - **POC simple** : pas de FID_UNI (trop lourd)
  - Métriques à calculer :
    - **FID global** et **par classe**
    - **LPIPS global** et **par classe**
  - **Mode unpaired** (car cGAN ne fonctionne pas avec la logique de paires)

#### 4.3. Cache des métriques ✅
- **Réponse** :
  - **Invalider les mesures précédentes** si le calcul est relancé
  - Cache avec `st.cache_data` keyed par (generator_type, selected_classes, experiment_id)

### 5. **Interface utilisateur**

#### 5.1. Galerie - Tab "Class preview"
- **Spécifié** : "2 rows × 4 columns (8 images at a time)"
- **Question** : Comment gérer la pagination si plus de 8 images sont générées ?
  - **Suggestion** : Ajouter des boutons "Previous/Next" ou un slider pour la page.

#### 5.2. Galerie - Tab "Real vs Synth"
- **Spécifié** : "Display a pair of images side-by-side"
- **Question** : Comment sélectionner les paires ?
  - Aléatoirement ?
  - Par ordre de génération ?
  - Permettre à l'utilisateur de choisir ?
  - **Suggestion** : Par défaut, aléatoirement (seed fixe), mais permettre de naviguer avec Previous/Next.

#### 5.3. Affichage des métriques
- **Question** : Format d'affichage des métriques FID/LPIPS ?
  - Tableau simple ?
  - Graphiques (bar charts) ?
  - Les deux ?
  - **Suggestion** : Les deux (tableau + graphiques) pour une meilleure visualisation.

### 6. **Gestion des erreurs et cas limites**

#### 6.1. Pas d'images générées
- **Question** : Que faire si l'utilisateur essaie d'évaluer sans avoir généré d'images ?
  - **Suggestion** : Désactiver les boutons d'évaluation et afficher un message explicite.

#### 6.2. Classes sans images réelles
- **Question** : Que faire si une classe sélectionnée n'a pas d'images dans `CRC-VAL-HE-7K` ?
  - **Suggestion** : Afficher un avertissement et exclure cette classe de la sélection.

#### 6.3. Mémoire GPU
- **Question** : Comment gérer les cas où la génération ou l'évaluation dépasse la mémoire GPU ?
  - **Suggestion** : Réduire automatiquement la taille des batches, ou afficher un message d'erreur clair.

### 7. **Structure du code backend**

#### 7.1. Module `dashboard_backend.py`
- **Spécifié** : "A backend helper module, e.g. workspace/p9dg/dashboard_backend.py"
- **Question** : Organisation des fonctions ?
  - Une classe principale `DashboardBackend` ?
  - Ou des fonctions indépendantes groupées par domaine (generation, evaluation, metrics) ?
  - **Suggestion** : Classe principale avec méthodes organisées par domaine pour faciliter la gestion de l'état.

#### 7.2. Réutilisation du code existant
- **Question** : Comment réutiliser `cnn_eval.py` et `fid_lpips_eval.py` ?
  - Importer directement les fonctions ?
  - Ou créer des wrappers pour adapter à l'interface Streamlit ?
  - **Suggestion** : Créer des wrappers pour isoler la logique Streamlit et faciliter les tests.

### 8. **Expérience utilisateur**

#### 8.1. Feedback pendant la génération
- **Question** : Comment afficher la progression pendant la génération (qui peut être longue) ?
  - **Suggestion** : Utiliser `st.progress()` et `st.status()` pour afficher la progression par classe.

#### 8.2. Sauvegarde des résultats
- **Question** : Faut-il permettre à l'utilisateur de sauvegarder/exporter les résultats (métriques, images) ?
  - **Suggestion** : Oui, ajouter des boutons "Export CSV" pour les métriques et "Download images" pour la galerie.

#### 8.3. Persistance entre sessions
- **Question** : Les images générées doivent-elles persister entre les redémarrages de l'app ?
  - **Suggestion** : Oui, les images sont sauvegardées sur disque, mais l'index dans `st.session_state` est perdu. Reconstruire l'index au démarrage en scannant `workspace/outputs/synth/`.

---

## 📋 Checklist de clarification recommandée

Avant de commencer l'implémentation, clarifier :

- [ ] **Génération** : Comment conditionner cGAN et PixCell ? Quels paramètres exposer ?
- [ ] **Pool réel** : Séparation claire génération/test ? Comment gérer le pool ?
- [ ] **CNN** : Comment charger exactement MobileNetV2 (architecture + poids) ? Comment récupérer `class_to_idx` ?
- [ ] **Métriques** : Référence exacte pour FID/LPIPS ? Comment calculer les scores globaux ?
- [ ] **UI** : Pagination galerie ? Sélection des paires Real vs Synth ?
- [ ] **Erreurs** : Gestion des cas limites (pas d'images, mémoire GPU, etc.) ?
- [ ] **Code** : Structure du backend (classe vs fonctions) ? Réutilisation du code existant ?
- [ ] **UX** : Feedback progression ? Export résultats ? Persistance entre sessions ?

---

## 💡 Suggestions d'amélioration

1. **Ajouter un onglet "Configuration avancée"** (collapsible) pour les paramètres de génération
2. **Ajouter un système de logs** pour tracer les opérations (génération, évaluation)
3. **Ajouter une visualisation de la distribution des classes** dans le dataset réel
4. **Permettre la comparaison entre deux générateurs** (côte à côte dans la galerie)
5. **Ajouter des statistiques sur les images générées** (taille, format, nombre par classe)

