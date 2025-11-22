# Analyse et Suggestions d'Amélioration - Notebook 05_StyleGAN.ipynb

## 📊 État Actuel

### Architecture Actuelle
- **Generator** : Version simplifiée StyleGAN2 (POC) sans plusieurs composants critiques
- **Discriminator** : Architecture simple avec spectral norm
- **Entraînement** : 2 époques seulement, batch size 4
- **Dataset** : 2700 images (300 par classe × 9 classes)

---

## 🔴 Problèmes Critiques Identifiés

### 1. **Architecture du Generator - Manque de Composants Essentiels**

#### ❌ Problèmes :
- **Pas de blur kernel (upfirdn2d)** : Le commentaire dit "StyleGAN fait un upsample + blur; ici: up simple pour POC"
  - Impact : Artifacts d'aliasing, images floues, perte de détails fins
- **Pas d'injection de bruit** : Aucun bruit ajouté dans les blocs du générateur
  - Impact : Manque de variation stochastique, textures répétitives
- **Pas de style mixing** : Le même w est utilisé pour tous les blocs
  - Impact : Moins de contrôle sur les styles, génération moins variée
- **Architecture trop simple** : Pas de skip connections, pas de modulation fine

#### ✅ Solutions Recommandées :

**A. Implémenter le blur kernel (upfirdn2d)**
```python
# Nécessite une implémentation de upfirdn2d pour l'upsampling avec blur
# Utiliser une implémentation existante (ex: stylegan2-pytorch) ou implémenter
# Le blur kernel réduit les artifacts d'aliasing lors de l'upsampling
```

**B. Ajouter l'injection de bruit**
```python
# Dans chaque bloc du générateur, ajouter :
# noise = torch.randn(batch_size, 1, h, w, device=x.device)
# x = x + noise * noise_weight  # où noise_weight est appris
```

**C. Implémenter le style mixing**
```python
# Utiliser différents w pour différents blocs :
# - w1 pour les blocs basse résolution (structure globale)
# - w2 pour les blocs haute résolution (détails)
# Probabilité de style mixing : ~0.9 pendant l'entraînement
```

---

### 2. **Architecture du Discriminator - Trop Simple**

#### ❌ Problèmes :
- Architecture très basique : juste des convs empilées
- Pas de features multiples pour la régularisation
- Pas de résolution progressive
- Pas de feature matching loss

#### ✅ Solutions Recommandées :

**A. Ajouter Feature Matching Loss**
```python
# Extraire les features intermédiaires du D pour les vraies et fausses images
# Loss = ||D_features(real) - D_features(fake)||²
# Cela guide le G à produire des features similaires aux vraies images
```

**B. Améliorer l'architecture du D**
- Ajouter des résidus (ResBlocks)
- Utiliser des features à plusieurs résolutions
- Ajouter de la normalisation adaptative

---

### 3. **Hyperparamètres d'Entraînement - Insuffisants**

#### ❌ Problèmes :
- **EPOCHS = 2** : Beaucoup trop peu ! StyleGAN nécessite des centaines d'époques
- **BATCH_SIZE = 4** : Très petit, limite la stabilité
- **SAMPLES_PER_CLASS = 300** : Peut être insuffisant pour capturer la diversité
- **Pas de learning rate scheduling** : LR fixe peut limiter la convergence

#### ✅ Solutions Recommandées :

**A. Augmenter drastiquement le nombre d'époques**
```python
EPOCHS = 50-100  # Minimum pour voir des résultats décents
# StyleGAN2 classique nécessite souvent 100-200+ époques
```

**B. Augmenter le batch size si possible**
```python
BATCH_SIZE = 8-16  # Si VRAM le permet
# Plus grand batch = gradients plus stables
```

**C. Ajouter un scheduler de learning rate**
```python
# Cosine annealing ou step decay
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS)
```

**D. Augmenter le dataset si possible**
```python
SAMPLES_PER_CLASS = 500-1000  # Plus de diversité
# Ou utiliser tout le dataset sans limitation
```

---

### 4. **Loss Functions - Manque de Guidance**

#### ❌ Problèmes :
- Pas de **perceptual loss** avec PathoDuet
- Pas de **feature matching loss**
- PathoDuet alpha trop faible (max 0.08)
- Pas de **path length regularization**

#### ✅ Solutions Recommandées :

**A. Ajouter Perceptual Loss avec PathoDuet**
```python
# Utiliser PathoDuet pour guider le générateur
def perceptual_loss(fake, real):
    fake_feat = pathoduet(fake)
    real_feat = pathoduet(real)
    return F.mse_loss(fake_feat, real_feat)

# Ajouter à la loss du générateur :
g_loss = g_nonsat_loss(fake_pred) + lambda_perceptual * perceptual_loss(fake, real)
```

**B. Augmenter le poids de PathoDuet**
```python
ALPHA_MAX = 0.2-0.5  # Au lieu de 0.08
# PathoDuet est spécialisé histopathologie, il devrait avoir plus de poids
```

**C. Ajouter Path Length Regularization**
```python
# Régularise la courbure de l'espace latent
# Aide à avoir un espace latent plus lisse et contrôlable
```

**D. Feature Matching Loss**
```python
# Extraire features intermédiaires du D
# Loss = ||D_intermediate(real) - D_intermediate(fake)||²
```

---

### 5. **Augmentations ADA - Trop Limitées**

#### ❌ Problèmes :
- Augmentations très simples (flip, translation)
- Pas de rotation (importante pour histopathologie)
- Pas de color jitter (désactivé)
- Pas de cutout/erasing

#### ✅ Solutions Recommandées :

**A. Ajouter plus d'augmentations**
```python
def ada_augment_enhanced(x, p, translate=0.04, rotate=0.1, color=0.1, cutout=0.1):
    # Rotation (importante pour histopathologie)
    if torch.rand(1) < p:
        angle = (torch.rand(1) - 0.5) * 2 * rotate * 180
        x = F.affine(x, angle=angle, translate=[0,0], scale=1.0, shear=0)
    
    # Color jitter (variations de teinte H&E)
    if torch.rand(1) < p and color > 0:
        # Jitter spécifique aux couleurs H&E
        ...
    
    # Cutout (simule les artefacts)
    if torch.rand(1) < p and cutout > 0:
        ...
```

**B. Augmenter ADA_MAX_P**
```python
ADA_MAX_P = 0.2-0.4  # Au lieu de 0.08
# Plus d'augmentations = meilleure généralisation
```

---

### 6. **Initialisation et Normalisation**

#### ❌ Problèmes :
- Pas d'initialisation spécifique mentionnée
- Pas de normalisation adaptative dans le générateur

#### ✅ Solutions Recommandées :

**A. Initialisation correcte des poids**
```python
# Initialiser les poids selon la distribution de StyleGAN2
# Utiliser une variance adaptée pour les ModulatedConv
```

**B. Ajouter Layer Normalization**
```python
# Dans les blocs du générateur, ajouter LayerNorm
# Aide à la stabilité de l'entraînement
```

---

### 7. **Progressive Growing (Optionnel mais Recommandé)**

#### ✅ Solution Recommandée :

**Implémenter Progressive Growing**
```python
# Commencer à 4x4, puis augmenter progressivement : 8x8, 16x16, 32x32, 64x64, 128x128, 256x256
# Cela permet un apprentissage plus stable et des images de meilleure qualité
# Transition douce entre résolutions
```

---

### 8. **Utilisation de PathoDuet - Pas Optimale**

#### ❌ Problèmes :
- PathoDuet est gelé (`torch.no_grad()`)
- Alpha trop faible (max 0.08)
- Pas utilisé pour guider directement le générateur

#### ✅ Solutions Recommandées :

**A. Utiliser PathoDuet comme Perceptual Loss**
```python
# Au lieu de juste dans le discriminateur, utiliser PathoDuet pour guider G
# Cela force G à produire des images avec des features histopathologiques correctes
```

**B. Augmenter significativement alpha**
```python
ALPHA_MAX = 0.3-0.5  # PathoDuet est spécialisé, il devrait avoir plus de poids
```

**C. Utiliser PathoDuet dès le début**
```python
ALPHA_FREEZE_UNTIL = 0  # Commencer avec PathoDuet activé
# Ou au moins beaucoup plus tôt (500 steps au lieu de 1500)
```

---

## 🎯 Plan d'Action Prioritaire

### Priorité 1 (Critique - Impact Immédiat) :
1. ✅ **Augmenter EPOCHS à 50-100 minimum**
2. ✅ **Ajouter l'injection de bruit dans le générateur**
3. ✅ **Ajouter Perceptual Loss avec PathoDuet**
4. ✅ **Augmenter ALPHA_MAX à 0.3-0.5**
5. ✅ **Augmenter BATCH_SIZE si VRAM le permet (8-16)**

### Priorité 2 (Important - Amélioration Significative) :
6. ✅ **Implémenter le blur kernel (upfirdn2d)**
7. ✅ **Ajouter Feature Matching Loss**
8. ✅ **Améliorer les augmentations ADA (rotation, color jitter)**
9. ✅ **Ajouter Learning Rate Scheduling**
10. ✅ **Implémenter le style mixing**

### Priorité 3 (Amélioration Continue) :
11. ✅ **Progressive Growing**
12. ✅ **Path Length Regularization**
13. ✅ **Améliorer l'architecture du Discriminator**
14. ✅ **Augmenter SAMPLES_PER_CLASS**

---

## 📝 Détails Techniques par Composant

### Generator - Modifications Nécessaires

```python
class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, img_res=256, fmap_base=256):
        # ... existant ...
        
        # AJOUTER : Noise injection pour chaque bloc
        self.noise_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(len(self.blocks))
        ])
        
    def forward(self, z, style_mixing_prob=0.9):
        w = self.mapping(z)
        
        # AJOUTER : Style mixing
        if self.training and torch.rand(1) < style_mixing_prob:
            w2 = self.mapping(torch.randn_like(z))
            cutoff = torch.randint(1, len(self.blocks), (1,))
            w = torch.cat([w[:cutoff], w2[cutoff:]])
        
        x = self.const.repeat(z.size(0), 1, 1, 1)
        
        for i, (m1, a1, m2, a2) in enumerate(self.blocks):
            # AJOUTER : Noise injection
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), 
                               device=x.device) * self.noise_weights[i]
            x = x + noise
            
            x = m1(x, w); x = a1(x)
            x = m2(x, w); x = a2(x)
        
        img = torch.tanh(self.to_rgb(x))
        return img
```

### Loss Functions - Ajouts

```python
# Perceptual Loss avec PathoDuet
def perceptual_loss_pathoduet(fake, real, pathoduet_model):
    with torch.no_grad():
        real_feat = pathoduet_model(real)
    fake_feat = pathoduet_model(fake)
    return F.mse_loss(fake_feat, real_feat)

# Feature Matching Loss
def feature_matching_loss(real_pred, fake_pred, D_model):
    # Extraire features intermédiaires (nécessite modification du D)
    real_feat = D_model.get_intermediate_features(real)
    fake_feat = D_model.get_intermediate_features(fake)
    return F.mse_loss(fake_feat, real_feat)

# Dans la boucle d'entraînement :
g_loss = (g_nonsat_loss(fake_pred) + 
          LAMBDA_STATS * real_stats.penalty(fake) +
          LAMBDA_PERCEPTUAL * perceptual_loss_pathoduet(fake, real, pathoduet) +
          LAMBDA_FM * feature_matching_loss(real_pred, fake_pred, D))
```

### Augmentations ADA - Améliorations

```python
def ada_augment_histo(x, p, translate=0.04, rotate=0.05, color=0.1):
    """Augmentations adaptées à l'histopathologie."""
    if p <= 0:
        return x
    
    b, c, h, w = x.shape
    
    # Rotation (crucial pour histopathologie)
    if torch.rand(1, device=x.device).item() < p and rotate > 0:
        angle = (torch.rand(1, device=x.device) - 0.5) * 2 * rotate * 180
        # Rotation avec interpolation bilinéaire
        x = F.affine(x, angle=angle.item(), translate=[0,0], 
                    scale=1.0, shear=0, interpolation='bilinear')
    
    # Color jitter spécifique H&E
    if torch.rand(1, device=x.device).item() < p and color > 0:
        # Jitter sur les canaux H (hematoxylin) et E (eosin)
        # Plus subtil que le jitter standard
        ...
    
    # Translation (existant mais améliorer)
    if translate > 0 and torch.rand(1, device=x.device).item() < p:
        max_pix = max(1, int(h * translate))
        dx = torch.randint(-max_pix, max_pix + 1, (1,), device=x.device).item()
        dy = torch.randint(-max_pix, max_pix + 1, (1,), device=x.device).item()
        x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
    
    return x
```

---

## 🔬 Métriques à Surveiller

### Métriques Existantes (Bonnes) :
- ✅ Duet-FID (excellent pour histopathologie)
- ✅ Accuracy du discriminateur
- ✅ Gap moyen entre real/fake

### Métriques à Ajouter :
- 📊 **IS (Inception Score)** : Diversité des images générées
- 📊 **LPIPS** : Distance perceptuelle entre images
- 📊 **Fréquence des styles** : Vérifier la diversité des styles générés
- 📊 **Histogramme des couleurs** : Comparer avec les vraies images H&E
- 📊 **Métriques morphologiques** : Utiliser le SSM du notebook 04 pour comparer les formes

---

## 🎓 Références et Implémentations

### Implémentations StyleGAN2 Complètes :
- **stylegan2-pytorch** : https://github.com/lucidrains/stylegan2-pytorch
- **stylegan2-ada-pytorch** : https://github.com/NVlabs/stylegan2-ada-pytorch (officiel NVIDIA)

### Papers Clés :
- StyleGAN2 (Karras et al., 2020)
- StyleGAN2-ADA (Karras et al., 2020) - Adaptive Discriminator Augmentation
- Training Generative Adversarial Networks with Limited Data (Karras et al., 2020)

---

## ⚠️ Avertissements

1. **Temps d'entraînement** : Avec 50-100 époques, l'entraînement prendra beaucoup plus de temps
2. **VRAM** : Les améliorations peuvent nécessiter plus de mémoire GPU
3. **Hyperparamètres** : Chaque modification nécessite un réglage fin des hyperparamètres
4. **Ordre d'implémentation** : Implémenter les changements progressivement et tester à chaque étape

---

## 📈 Résultats Attendus

Avec ces améliorations, on devrait observer :
- ✅ Images plus nettes et détaillées
- ✅ Meilleure cohérence des textures histopathologiques
- ✅ Diversité accrue dans les images générées
- ✅ Duet-FID en baisse (meilleure qualité)
- ✅ Meilleure séparation des classes (si conditionné)

---

## 🚀 Quick Wins (Améliorations Rapides)

Si vous voulez des résultats rapides sans refonte complète :

1. **Augmenter EPOCHS à 20-30** (5 minutes de modification)
2. **Augmenter ALPHA_MAX à 0.3** (1 minute)
3. **Ajouter Perceptual Loss avec PathoDuet** (15 minutes)
4. **Augmenter BATCH_SIZE à 8** (si VRAM OK, 2 minutes)
5. **Ajouter rotation dans ADA** (10 minutes)

Ces 5 changements devraient déjà améliorer significativement les résultats.

