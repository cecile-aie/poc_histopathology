# 🔍 Tests Critiques Manquants - Notebook 01_test_datagenerator.ipynb

## ✅ Ce qui a été testé (bien couvert)
- Configuration JSON des seuils
- Chargement train/val avec échantillonnage
- Normalisation Vahadane (activation)
- Calibration des métriques qualité
- Visualisation et DataLoader
- Mapping des classes
- Séparation train/val
- Filtrage qualité (taux de passage)
- Modes CNN et GAN (pixel_range, return_labels)
- Filtrage par classes spécifiques

## ⚠️ Aspects CRITIQUES non testés

### 1. 🔄 **Test de `no_repeat_eval=True` en mode validation** ⚠️ CRITIQUE
**Problème potentiel** : La cellule 3 charge `ds_val` avec `no_repeat_eval=True`, mais il n'y a **aucun test** pour vérifier que :
- Les images ne se répètent pas entre les époques en mode validation
- Le comportement est différent entre train (avec remise) et val (sans remise)

**Test suggéré** :
```python
# Test no_repeat_eval en validation
ds_val_test = HistoDataset(
    root_data=ROOT_DATA,
    split="val",
    output_size=IMG_SIZE,
    thresholds_json_path="seuils_par_classe.json",
    vahadane_enable=False,
    samples_per_class_per_epoch=50,
    no_repeat_eval=True  # ← Important
)

# Vérifier que les images ne se répètent pas entre époques
epoch_0_paths = {os.path.basename(p) for _, _, p in [ds_val_test[i] for i in range(len(ds_val_test))]}
ds_val_test.set_epoch(1)
epoch_1_paths = {os.path.basename(p) for _, _, p in [ds_val_test[i] for i in range(len(ds_val_test))]}

overlap = len(epoch_0_paths & epoch_1_paths)
print(f"Chevauchement entre époques: {overlap} (devrait être 0 si no_repeat_eval=True)")
assert overlap == 0, "❌ Les images se répètent entre époques en validation !"
```

### 2. 🔍 **Test de cohérence des indices `_epoch_indices`** ⚠️ IMPORTANT
**Problème potentiel** : Vérifier que les indices dans `_epoch_indices` correspondent bien aux images chargées et que `subsample_dataset` fonctionne correctement.

**Test suggéré** :
```python
# Vérifier que subsample_dataset préserve la cohérence
for idx in ds_train_small.indices[:10]:
    ci, j = ds_train[0]._epoch_indices[idx]  # Index dans le dataset complet
    path_expected = ds_train.paths_by_class[ci][j]
    _, _, path_actual = ds_train_small[0]  # Premier élément du subset
    # Vérifier que les chemins correspondent
```

### 3. 🚨 **Test des cas limites du filtrage qualité** ⚠️ CRITIQUE
**Problème observé** : La cellule 20 montre que certaines classes ont un taux de rejet très élevé :
- MUC: 11/50 OK (78% rejeté)
- DEB: 20/50 OK (60% rejeté)
- NORM: 20/50 OK (60% rejeté)

**Risque** : Si trop d'images sont rejetées, le dataset peut devenir trop petit ou déséquilibré.

**Test suggéré** :
```python
# Test du comportement quand le filtre rejette trop d'images
# Vérifier que le mécanisme de retry (max 5 tentatives) fonctionne
# Vérifier qu'on n'obtient pas d'erreur si toutes les images d'une classe sont rejetées

for ci, cname in ds_train.idx_to_class.items():
    paths = ds_train.paths_by_class[ci]
    rejected_count = 0
    for path in paths[:100]:  # Échantillon
        img = Image.open(path)
        metrics = ds_train.qf.score(img)
        thr = ds_train.class_thresholds.get(cname, {})
        if not ds_train.qf.check(metrics, thr):
            rejected_count += 1
    
    rejection_rate = rejected_count / min(100, len(paths))
    print(f"{cname}: {rejection_rate:.1%} rejeté")
    if rejection_rate > 0.9:
        print(f"⚠️ ATTENTION: {cname} a un taux de rejet > 90% !")
```

### 4. 🔄 **Test de la stabilité de la normalisation Vahadane** ⚠️ IMPORTANT
**Problème potentiel** : Vérifier que la normalisation Vahadane est cohérente entre les époques et ne cause pas de dérive.

**Test suggéré** :
```python
# Test de cohérence de la normalisation
sample_path = random.choice(ds_train.paths_by_class[0])
img_raw = Image.open(sample_path)

# Normaliser plusieurs fois
img_norm_1 = ds_train.stain.normalize(img_raw)
ds_train.set_epoch(1)  # Réinitialiser
img_norm_2 = ds_train.stain.normalize(img_raw)

# Vérifier que les résultats sont identiques (ou très proches)
diff = np.abs(np.array(img_norm_1) - np.array(img_norm_2)).mean()
print(f"Différence moyenne entre normalisations: {diff:.6f}")
assert diff < 1.0, "❌ La normalisation n'est pas stable !"
```

### 5. 📊 **Test de l'équilibrage réel dans les batches** ⚠️ IMPORTANT
**Observation** : La cellule 16 montre une répartition inégale (ADI: 13.8%, MUC: 8.8%), ce qui est normal avec shuffle, mais il faudrait vérifier sur plusieurs époques.

**Test suggéré** :
```python
# Test de l'équilibrage sur plusieurs époques
from collections import Counter

all_counts = Counter()
for epoch in range(5):
    ds_train.set_epoch(epoch)
    loader = DataLoader(ds_train_small, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for i, (_, y, _) in enumerate(loader):
        if i >= 20:  # 20 batches par époque
            break
        all_counts.update(y.cpu().numpy())

# Vérifier que la distribution est équilibrée sur plusieurs époques
total = sum(all_counts.values())
for ci, count in sorted(all_counts.items()):
    class_name = ds_train.idx_to_class[ci]
    proportion = count / total * 100
    expected = 100 / len(ds_train.paths_by_class)  # ~11.1% pour 9 classes
    print(f"{class_name}: {proportion:.1f}% (attendu: ~{expected:.1f}%)")
```

### 6. 🛡️ **Test de robustesse (fichiers corrompus, chemins invalides)** ⚠️ BONUS
**Test suggéré** :
```python
# Vérifier que le dataset gère gracieusement les erreurs
# (déjà partiellement testé dans __getitem__ avec le retry, mais pourrait être plus complet)
```

### 7. 🔢 **Test des ranges de pixels (0_1, -1_1, imagenet)** ⚠️ IMPORTANT
**Observation** : La cellule 26 teste les modes CNN et GAN, mais pas le mode "imagenet".

**Test suggéré** :
```python
# Test du mode imagenet
ds_imagenet = HistoDataset(
    root_data=str(DATA_ROOT),
    split="train",
    output_size=IMAGE_SIZE,
    pixel_range="imagenet",  # ← Test manquant
    samples_per_class_per_epoch=50,
    return_labels=True
)
x, y, _ = ds_imagenet[0]
print(f"ImageNet: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}")
# Vérifier que les valeurs sont dans la plage attendue après normalisation ImageNet
```

## 📋 Priorités

1. **🔴 CRITIQUE** : Test de `no_repeat_eval=True` (aspect fondamental pour la validation)
2. **🔴 CRITIQUE** : Test des cas limites du filtrage qualité (risque de dataset trop petit)
3. **🟡 IMPORTANT** : Test de cohérence des indices après `subsample_dataset`
4. **🟡 IMPORTANT** : Test de stabilité de la normalisation Vahadane
5. **🟡 IMPORTANT** : Test de l'équilibrage sur plusieurs époques
6. **🟢 BONUS** : Test du mode imagenet et robustesse

