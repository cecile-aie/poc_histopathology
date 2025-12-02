---

# cGAN conditionnel – Configuration technique du POC

*Notebook : `06b_cGAN_IA.ipynb`*

Ce document décrit la configuration technique du cGAN conditionnel utilisé dans le POC histopathologie (NCT-CRC-HE-100K), y compris :

* l’architecture (G/D, tête PathoDuet)
* les réglages d’entraînement (hyperparamètres)
* les différentes régularisations / garde-fous rajoutés pour stabiliser le training
* les métriques loggées pour le suivi (CSV, FID/LPIPS, downstream CNN)

L’objectif n’est pas de justifier chaque choix, mais de rendre explicite la complexité du setup.

---

## 1. Données & pré-processing

### 1.1 Dataset & splits

* Dataset principal : `NCT-CRC-HE-100K` (train)
* Dataset de validation/test : `CRC-VAL-HE-7K`
* Gestion via `HistoDataset` (`histo_dataset.py`) :

  * Split sans fuite train / val / test

  * Filtre qualité basé sur plusieurs métriques (LaplacianVar, entropie, ratio de tissu, saturation, blockiness spatial & DCT, etc.)

  * Mapping de classes basé sur `class_mappings.py` :

    * `ADI`  – Tissu adipeux
    * `BACK` – Arrière-plan
    * `DEB`  – Débris / nécrose
    * `LYM`  – Lymphocytes
    * `MUC`  – Mucus
    * `MUS`  – Muscle lisse
    * `NORM` – Muqueuse colique normale
    * `STR`  – Stroma associé au cancer
    * `TUM`  – Épithélium tumoral (adénocarcinome)

  * `num_classes = 9`

### 1.2 Pré-processing images

* Taille d’entrée : `IMAGE_SIZE = 256` (patchs 256×256×3)
* Pixel range :

  * Chargement en [0,1]
  * Re-scaling en `[-1,1]` : `PIXEL_RANGE = "-1_1"`
* Normalisation d’hematoxyline-éosine :

  * `VAHADANE_ENABLE = True` (normalisation de stain type Vahadane si lib dispo)
* Échantillonnage équilibré par classe :

  * `balance_per_class = True`
  * `samples_per_class_per_epoch = SAMPLES_PER_CLASS`
  * Valeur utilisée en POC : `SAMPLES_PER_CLASS = 1000` (et plus dans les runs récents)

### 1.3 DataLoader

* `BATCH_SIZE = 8`
* `NUM_WORKERS = 0` (robuste en environnement notebook / Docker)
* `PIN_MEMORY = (DEVICE == "cuda")`
* `DROP_LAST = True`
* `SHUFFLE = True`

Un loader dédié `loader_real_eval` est construit sur un sous-ensemble tenu à part pour les métriques FID/LPIPS.

---

## 2. Architecture du modèle

### 2.1 Variables globales

* `Z_DIM = 512` (dimension du bruit latent)
* Image : `3 × 256 × 256`
* Nombre de classes : `num_classes = 9`

---

### 2.2 Générateur – StyleGAN-lite conditionnel

**Classe : `Generator`**

* Style-based generator “lite” inspiré de StyleGAN, avec conditionnalité de classe :

  * Embedding de classe :

    ```python
    self.embed = nn.Embedding(num_classes, z_dim)
    ```
  * Mapping network :

    * `MappingNetwork(z_dim=512, w_dim=512, n_layers=8)`
    * Suites de couches `Linear + LeakyReLU(0.2)`
    * Sortie : style vector `w` de dimension 512

* Injection de condition :

  * On forme `zc = z + embed(y)` puis on passe `zc` dans le `MappingNetwork` pour obtenir `w`.

* Construction du réseau :

  * Constante apprise : `const` de taille `(1, fmap_base, 4, 4)`

  * Progression par blocs jusqu’à `img_res = 256` :

    ```python
    FMAP_BASE = 512  # canaux de base augmentés
    ```

    Pour chaque résolution :

    * Deux `ModulatedConv2d` (upscale + conv) + `LeakyReLU(0.2)`
    * Injection de bruit par bloc :

      * Paramètre `noise_weight` par bloc, initialisé à 0
      * Bruit gaussien 2D ajouté aux features

  * Tête finale :

    ```python
    self.to_rgb = nn.Conv2d(in_ch, 3, kernel_size=1)
    ```

* Style mixing conditionnel (après correction) :

  * Optionnel, activé via :

    ```python
    USE_STYLE_MIXING = True
    STYLE_MIX_PROB   = 0.9
    ```
  * On génère un second bruit `z_mix` pour le même label `y` (mixing **intra-classe**) :

    * `w_mix = mapping(z_mix + embed(y))`
    * Pour certains blocs (selon `mix_prob`), on utilise `w_mix` à la place de `w`
  * But : favoriser la diversité tout en restant cohérent avec la classe cible.

* Clamp de sortie :

  * Pour éviter la saturation complète :

    ```python
    output = output.clamp(-0.9, 0.9)
    ```

---

### 2.3 Discriminateur de base

**Classe : `Discriminator`**

* CNN séquentiel avec Spectral Normalization :

  * Plusieurs blocs `Conv2d → LeakyReLU` avec stride 2 pour downsampling
  * Canaux de base augmentés :

    ```python
    D_BASE_CH = 96  # vs 64 dans les versions plus simples
    ```
* Tête finale :

  * `conv_last` (kernel 4×4, stride 1)
  * `to_logit` (1×1) → logit scalaire `logit_base`
  * Flatten des features en vecteur `feat` de dimension `out_ch`

---

### 2.4 Discriminateur combiné conditionnel + PathoDuet

**Classe : `CombinedCondDiscriminator`**

* Entrées :

  * Image `x`
  * Label `y` (indices de classe)

* Composants :

  1. Tronc CNN : `D_base(x)` → `logit_base, features`
  2. Tête de projection (cGAN) :

     * Embedding de classe
     * Produit scalaire avec les features → `logit_proj`
  3. Tête PathoDuet :

     * `duet_extractor(x)` : modèle PathoDuet pré-entraîné utilisé comme extracteur
     * Logit supplémentaire `logit_duet` (via tête linéaire)

* Logit final :

  ```python
  logit = logit_base + logit_proj + alpha * logit_duet
  ```

  où `alpha` est un coefficient dynamique (voir §4.4).

---

## 3. Pertes & termes de régularisation

### 3.1 Perte adversariale

* Formulation NS-GAN (non-saturante), avec possibilité de :

  * soit logs `softplus` (StyleGAN-like),
  * soit `BCEWithLogits` avec label smoothing (chemin label smoothing activé dans le code actuel).

* Discriminateur :

  * Si `USE_LABEL_SMOOTHING = True` :

    ```python
    LABEL_SMOOTHING_REAL = 0.9
    LABEL_SMOOTHING_FAKE = 0.0

    d_loss_real = BCEWithLogits(real_pred, real_target=0.9)
    d_loss_fake = BCEWithLogits(fake_pred, fake_target=0.0)
    d_loss = d_loss_real + d_loss_fake
    ```
  * Sinon :

    ```python
    d_loss = d_logistic_loss(real_pred, fake_pred)
    ```

* Générateur :

  ```python
  g_adv = g_nonsat_loss(fake_pred)
  ```

### 3.2 Pénalité R1 (gradient penalty sur les vrais)

* Appliquée toutes les `R1_EVERY` itérations (en pratique : `R1_EVERY = 8`) :

  * `R1_GAMMA = 15.0`
  * Calcul :

    ```python
    r1 = 0.5 * R1_GAMMA * r1_penalty(real_aug_r1, real_pred_r1)
    ```
* Permet de régulariser le D et de stabiliser l’entraînement.

### 3.3 Stats d’intensité (RealStatsEMA)

* Classe `RealStatsEMA` :

  * Maintient un EMA de la moyenne et de l’écart-type par canal sur les images **réelles** (en [0,1]).
* Pénalité :

  * On applique les mêmes stats au fake (après clamp et re-scaling en [0,1]) et on pénalise l’écart :

    ```python
    real_stats = RealStatsEMA(m=0.995)
    LAMBDA_STATS = 3e-3
    g_stats = LAMBDA_STATS * real_stats.penalty(fake)
    ```

### 3.4 Perceptual loss PathoDuet

* Utilise PathoDuet comme extracteur de features :

  * `fake_feat = pathoduet(fake)`
  * `real_feat = pathoduet(real)` (en `no_grad`)
* Perte :

  ```python
  LAMBDA_PERCEPTUAL = 0.01  # déjà réduite par rapport aux essais initiaux
  g_perc = LAMBDA_PERCEPTUAL * MSE(fake_feat, real_feat)
  ```
* Garde-fou dynamique en cas de saturation (voir §4.5).

### 3.5 Saturation penalty

* Surveille la valeur moyenne absolue des pixels synthétiques :

  ```python
  fake_mean_abs = fake.abs().mean()
  saturation_penalty = clamp(fake_mean_abs - 0.6, min=0)**2
  LAMBDA_SATURATION = 0.05
  g_sat = LAMBDA_SATURATION * saturation_penalty
  ```

### 3.6 Feature matching

* On réutilise le discriminateur de base comme extracteur :

  * `fake_feat = D.base(fake_aug)`
  * `real_feat = D.base(real_for_feat)`
* Perte :

  ```python
  LAMBDA_FEATURE_MATCHING = 0.1
  g_fm = LAMBDA_FEATURE_MATCHING * MSE(fake_feat, real_feat_detach)
  ```

### 3.7 Mixing regularization (diversité intra-batch)

* Encourage la diversité des images générées au sein d’un batch :

  * On aplatie `fake` en `(B, -1)`, on calcule la variance globale `fake_var` et on pénalise les batchs trop homogènes :

    ```python
    USE_MIXING_REG = True
    MIXING_REG_WEIGHT = 0.1
    mixing_reg_loss = clamp(0.1 - fake_var, min=0)**2
    g_mix = MIXING_REG_WEIGHT * mixing_reg_loss
    ```

### 3.8 Loss totale générateur

Au final, la loss G est de la forme :

```python
g_loss = (
    g_nonsat_loss(fake_pred)
    + LAMBDA_STATS       * real_stats.penalty(fake)
    + LAMBDA_PERCEPTUAL  * perceptual_loss_pathoduet(fake, real)
    + LAMBDA_SATURATION  * saturation_penalty
    + LAMBDA_FEATURE_MATCHING * feat_match_loss
    + MIXING_REG_WEIGHT  * mixing_reg_loss
)
```

---

## 4. Mécanismes de stabilité & garde-fous

### 4.1 ADA – Adaptive Data Augmentation

* Implémentation inspirée de StyleGAN2 adaptée à l’histopathologie :

  ```python
  ADA_STATE = {"p": 0.0, "acc_ema": None}
  ADA_TARGET = 0.6
  ADA_DECAY  = 0.99
  ADA_SPEED  = 0.20
  ADA_MAX_P  = 0.12
  ADA_FREEZE_UNTIL = 500
  ADA_P_CAP_IF_ACC_HI = 0.08  # si acc_ema > 0.9
  ```

* Augmentations :

  * Flip horizontal
  * Rotation légère (± quelques degrés)
  * Translation (quelques pixels)
  * Jitter couleur désactivé (`color=0.0`)

* Mise à jour :

  * On suit `acc_ema` de D sur les vrais, puis :

    ```python
    err = acc_ema - ADA_TARGET
    new_p = clamp(p + 0.001 * err * ADA_SPEED, 0, ADA_MAX_P)
    ```
  * Si `acc_ema > 0.9`, on plafonne `p` à `ADA_P_CAP_IF_ACC_HI`.

### 4.2 R1 + Drift penalty

* R1 déjà décrit (§3.2)
* Drift penalty StyleGAN2-like sur les logits réels :

  ```python
  DRIFT_EPS = 1e-3
  d_loss += DRIFT_EPS * (real_pred ** 2).mean()
  ```

### 4.3 EMA du générateur

* `G_ema` est une copie Exponential Moving Average de `G` :

  * Initialisation :

    ```python
    G_ema = Generator(..., fmap_base=G.const.shape[1], num_classes=num_classes)
    G_ema.load_state_dict(G.state_dict())
    requires_grad(G_ema, False)
    ```
  * Mise à jour à chaque step :

    ```python
    ema_update(G_ema, G, decay=0.9995)
    ```

* `G_ema` est utilisé pour :

  * la génération d’images de monitoring / samples,
  * le calcul FID/LPIPS,
  * la sauvegarde du meilleur modèle (`cgan_best_model.pt`).

### 4.4 Alpha PathoDuet (gate adaptatif)

* Coefficient `alpha` modulant la contribution de la tête PathoDuet :

  ```python
  ACC_GATE_HI  ≈ 0.78   # seuil ON
  ACC_GATE_LO  ≈ 0.70   # seuil OFF
  ALPHA_MAX    ≈ 0.15   # borne supérieure de alpha
  ALPHA_DALPHA = 1e-4   # increment max par step
  ALPHA_TAU    = 400.0  # constante de temps
  ALPHA_FREEZE_UNTIL = 500
  DUET_DWELL_STEPS   = 200  # hysteresis
  ```

* Logiciel :

  * Tant que `step < ALPHA_FREEZE_UNTIL` → `alpha = 0`
  * Ensuite :

    * Si `acc_ema ≥ ACC_GATE_HI` et D stable → gate **ON** et alpha → `ALPHA_MAX`
    * Si `acc_ema ≤ ACC_GATE_LO` → gate **OFF** et alpha → `0`
    * Interpolation lissée (`k = 1 - exp(-1/ALPHA_TAU)`) + clamp par `ALPHA_DALPHA`

* Résultat : PathoDuet n’est pas imposé dès le début, il est injecté progressivement quand D est déjà “compétent”.

### 4.5 Détection automatique de saturation

* Après la mise à jour de G :

  * On mesure `fake_mean_abs = fake.abs().mean().item()`
  * Si `fake_mean_abs > 0.75` :

    * On réduit dynamiquement `LAMBDA_PERCEPTUAL` (multiplié par 0.9, avec borne >= 0.005)
    * On réduit dynamiquement `ALPHA_MAX` (multiplié par 0.95, borne >= ~0.1)
    * On force le gate PathoDuet OFF (`D._gate_on = False`)
    * Message loggué toutes les 50 itérations pour trace

Objectif : empêcher les “numériques” trop agressifs (PathoDuet + perceptuel) de pousser le G vers des images complètement saturées.

### 4.6 Gradient clipping & AMP

* Optimisation :

  * `opt_G = Adam(G.parameters(), lr=4e-4, betas=(0.0, 0.99))`
  * `opt_D = Adam(D.parameters(), lr=7e-5, betas=(0.0, 0.99))`
* Clipping :

  * `torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)`
  * `torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)`
* AMP :

  * AMP supporté via `GradScaler`, mais `USE_AMP = False` dans la config actuelle (stabilité prioritaire).

### 4.7 Scheduler de LR

* Scheduler optionnel :

  ```python
  USE_LR_SCHEDULER = True
  LR_FINAL_RATIO   = 0.01
  TOTAL_STEPS_ESTIMATE = EPOCHS * 250  # approx
  scheduler_G = CosineAnnealingLR(opt_G, T_max=TOTAL_STEPS_ESTIMATE,
                                  eta_min=LR_G * LR_FINAL_RATIO)
  scheduler_D = CosineAnnealingLR(opt_D, T_max=TOTAL_STEPS_ESTIMATE,
                                  eta_min=LR_D * LR_FINAL_RATIO)
  ```
* Mise à jour à chaque step (après `step()` de l’optimizer).

---

## 5. Hyperparamètres d’entraînement

* `EPOCHS = 50` (les runs POC existants s’arrêtent vers ~10–15 epochs)
* `BATCH_SIZE = 8`
* `LR_G = 4e-4`, `LR_D = 7e-5`
* `BETAS = (0.0, 0.99)`
* `SAMPLE_EVERY = 400` steps (grilles d’images)
* `SAVE_EVERY = 800` steps (checkpoints `cgan_stepXXXXXX.pt`)

---

## 6. Suivi & métriques

### 6.1 CSV `metrics_gan.csv`

Colonnes principales :

* `epoch`, `step`
* `d_loss`, `g_loss`
* `real_mu`, `real_min`, `real_max` (après clamp)
* `fake_mu`, `fake_min`, `fake_max` (après clamp)
* `ada_p` (probabilité d’augmentation ADA)
* `acc` (accuracy instantanée de D sur les vrais)
* `acc_ema` (moyenne exponentielle)
* `alpha` (poids courant de la tête PathoDuet)

### 6.2 FID / LPIPS et évaluation downstream

* Un pipeline dédié (cells 10–12) :

  * Recharge `cgan_best_model.pt` (G_ema)
  * Génère un certain nombre d’images synthétiques par classe :

    * `N_SYNTH_PER_CLASS = 200` (typique)
  * Évalue :

    * FID & LPIPS via `FIDLPIPSEvaluator`
    * Performances downstream via le CNN baseline (`cnn_eval.py`)

      * Accuracies, matrices de confusion, etc. sur real vs synth

Les résultats sont stockés dans :

* `outputs/06b_cgan_ia/` (images, CSV)
* `artifacts/` et `metrics/` (métriques globales, FID, etc.)

---

## 7. Résumé “one-liner” du setup

> cGAN conditionnel StyleGAN-like, avec tête de discriminant duale (projection + PathoDuet), adatation ADA, R1, stats d’intensité, perceptual histo-spécifique, feature matching, pénalité de saturation, régularisation de diversité et EMA du générateur, le tout piloté par une série de thresholds et schedulers pour éviter les divergences et le mode collapse.

---

