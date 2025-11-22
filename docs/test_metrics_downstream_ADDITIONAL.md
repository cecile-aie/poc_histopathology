# Tests supplémentaires pour `cnn_eval.py`

## 📊 Analyse du notebook existant

Le notebook `test_metrics_downstream.ipynb` teste déjà :
- ✅ Import des modules
- ✅ Évaluation depuis CSV avec calibration
- ✅ Inférence complète sur images real/synth
- ✅ PCR (Pair Consistency Rate)
- ✅ Métriques complètes (ECE, Brier, Accuracy, F1)
- ✅ Visualisation d'erreurs

## 🧪 Tests supplémentaires recommandés

### 1. Tests unitaires des fonctions individuelles

#### Test `compute_ece()`
```python
# Test avec probabilités parfaitement calibrées (ECE = 0)
# Test avec probabilités non calibrées (ECE > 0)
# Test avec différents nombres de bins
# Test avec un seul échantillon
# Test avec toutes les prédictions correctes
```

#### Test `fit_temperature()`
```python
# Test que T=1.0 pour un modèle déjà calibré
# Test que T améliore l'ECE
# Test avec différents nombres d'itérations
# Test avec logits très confiants vs peu confiants
```

#### Test `load_temperature()` et `fit_temperature_from_csv()`
```python
# Test chargement température depuis JSON valide
# Test chargement température depuis JSON invalide
# Test chargement température depuis fichier inexistant
# Test fit depuis CSV avec logits valides
# Test fit depuis CSV avec logits invalides
```

#### Test `_resolve_classes()`
```python
# Test avec class_to_idx fourni
# Test sans class_to_idx (fallback générique)
# Test avec class_mappings disponible
# Test avec class_mappings indisponible
```

### 2. Tests de cas limites et robustesse

#### CSV invalides ou incomplets
```python
# CSV vide
# CSV sans colonne 'logits_json'
# CSV sans colonne 'y_true'
# CSV avec logits_json invalide (pas JSON)
# CSV avec logits de longueurs différentes
# CSV avec y_true hors bornes (négatif ou >= n_classes)
```

#### Données invalides
```python
# Logits avec NaN
# Logits avec Inf
# Probabilités qui ne somment pas à 1
# Labels négatifs
# Labels >= n_classes
```

#### Température invalide
```python
# Température = 0
# Température négative
# Température = Inf
# Température = NaN
# JSON avec format invalide
```

### 3. Tests de cohérence

#### Cohérence des métriques
```python
# Vérifier que ECE calibré <= ECE raw (normalement)
# Vérifier que Brier calibré <= Brier raw (normalement)
# Vérifier que accuracy = (y_pred == y_true).mean()
# Vérifier que les probabilités somment à 1
# Vérifier que top1_conf correspond à max(probs)
```

#### Cohérence entre fonctions
```python
# Vérifier que export_predictions() produit un CSV valide pour evaluate_csv_metrics()
# Vérifier que fit_temperature_from_csv() produit le même T que fit_temperature()
# Vérifier que run_eval_split() produit les mêmes résultats que le pipeline manuel
```

### 4. Tests de réutilisation

#### Réutilisation des fichiers
```python
# Charger température depuis JSON et l'appliquer à un nouveau CSV
# Évaluer le même CSV plusieurs fois (doit donner les mêmes résultats)
# Évaluer un CSV avec différentes températures
# Évaluer un CSV sans température puis avec température
```

### 5. Tests de performance

#### Performance des fonctions
```python
# Mesurer le temps d'exécution de compute_ece() sur différents tailles
# Mesurer le temps d'exécution de fit_temperature() sur différents tailles
# Mesurer le temps d'exécution de export_predictions() sur différents batch sizes
# Vérifier que l'AMP accélère l'inférence
```

### 6. Tests d'intégration

#### Pipeline complet
```python
# Test run_eval_split() avec fit_temperature_on_val=True
# Test run_eval_split() avec temp_json fourni
# Test run_eval_split() sans calibration
# Test run_eval_split() avec make_plots=False
# Test run_eval_split() avec normalize_cm=True
```

### 7. Tests de visualisation

#### Génération des figures
```python
# Vérifier que les figures sont créées quand plot=True
# Vérifier que les figures ne sont pas créées quand plot=False
# Vérifier que les chemins des figures sont corrects
# Vérifier que les figures sont lisibles (pas vides)
```

## 📝 Exemple de cellules à ajouter au notebook

### Cellule : Tests unitaires ECE
```python
# Test compute_ece avec probabilités parfaitement calibrées
import torch
from metrics.cnn_eval import compute_ece

# Cas idéal : confiance = accuracy pour chaque bin
n_samples = 1000
n_classes = 9
y_true = torch.randint(0, n_classes, (n_samples,))
# Créer des probabilités parfaitement calibrées
probs = torch.rand(n_samples, n_classes)
probs = probs / probs.sum(dim=1, keepdim=True)
# Ajuster pour que la confiance max = accuracy
for i in range(n_samples):
    pred = probs[i].argmax()
    if pred == y_true[i]:
        probs[i, pred] = 0.9  # Haute confiance pour prédiction correcte
    else:
        probs[i, pred] = 0.1  # Basse confiance pour prédiction incorrecte
    probs[i] = probs[i] / probs[i].sum()

ece = compute_ece(probs, y_true)
print(f"ECE pour probabilités calibrées : {ece:.6f} (devrait être proche de 0)")
assert ece < 0.1, "ECE trop élevé pour probabilités calibrées"
```

### Cellule : Tests de robustesse CSV
```python
# Test avec CSV invalide
import pandas as pd
import json
from pathlib import Path
from metrics.cnn_eval import evaluate_csv_metrics

# Créer un CSV de test avec données invalides
test_csv = Path("/tmp/test_invalid.csv")
df_test = pd.DataFrame({
    "split": ["val"] * 10,
    "image_path": [f"test_{i}.png" for i in range(10)],
    "y_true": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 9 est hors bornes si n_classes=9
    "y_pred": [0] * 10,
    "top1_conf": [0.9] * 10,
    "logits_json": [json.dumps([1.0] * 8) for _ in range(10)]  # 8 classes au lieu de 9
})
df_test.to_csv(test_csv, index=False)

# Tester que la fonction gère les erreurs gracieusement
try:
    report = evaluate_csv_metrics(test_csv, plot=False)
    print("⚠️ La fonction n'a pas détecté l'erreur")
except Exception as e:
    print(f"✅ Erreur détectée correctement : {type(e).__name__}")
```

### Cellule : Test de cohérence température
```python
# Vérifier que fit_temperature() et fit_temperature_from_csv() donnent le même résultat
import torch
import pandas as pd
import json
from pathlib import Path
from metrics.cnn_eval import fit_temperature, fit_temperature_from_csv, export_predictions

# Créer des données de test
n_samples = 100
n_classes = 9
logits = torch.randn(n_samples, n_classes)
y_true = torch.randint(0, n_classes, (n_samples,))

# Fit température directement
scaler1 = fit_temperature(logits, y_true)
T1 = float(torch.exp(scaler1.log_t).item())

# Export vers CSV puis fit depuis CSV
test_csv = Path("/tmp/test_temp.csv")
df = pd.DataFrame({
    "split": ["val"] * n_samples,
    "image_path": [f"test_{i}.png" for i in range(n_samples)],
    "y_true": y_true.tolist(),
    "y_pred": [0] * n_samples,
    "top1_conf": [0.9] * n_samples,
    "logits_json": [json.dumps(logits[i].tolist()) for i in range(n_samples)]
})
df.to_csv(test_csv, index=False)

test_json = Path("/tmp/test_temp.json")
T2 = fit_temperature_from_csv(test_csv, test_json)

# Comparer
diff = abs(T1 - T2)
print(f"T1 (direct) = {T1:.6f}")
print(f"T2 (CSV)    = {T2:.6f}")
print(f"Différence  = {diff:.6f}")
assert diff < 1e-3, f"Températures différentes : {diff}"
print("✅ Les deux méthodes donnent le même résultat")
```

### Cellule : Test de réutilisation
```python
# Vérifier qu'on peut réutiliser un CSV avec différentes températures
from metrics.cnn_eval import evaluate_csv_metrics, load_temperature
import json
from pathlib import Path

# Charger le CSV de référence
val_csv = Path("/workspace/outputs/baseline/mobilenetv2_preds_val.csv")
temp_json = Path("./artifacts/mobilenetv2_temp_scaling.json")

# Évaluer avec la température de référence
report1 = evaluate_csv_metrics(val_csv, temp_json=temp_json, plot=False)

# Créer une température différente
T_original = load_temperature(temp_json)
T_modified = T_original * 1.5  # Augmenter de 50%
temp_modified_json = Path("/tmp/temp_modified.json")
with open(temp_modified_json, "w") as f:
    json.dump({"temperature": T_modified}, f)

# Évaluer avec la température modifiée
report2 = evaluate_csv_metrics(val_csv, temp_json=temp_modified_json, plot=False)

print(f"ECE avec T={T_original:.3f} : {report1.ece_cal:.4f}")
print(f"ECE avec T={T_modified:.3f} : {report2.ece_cal:.4f}")
print("✅ Réutilisation du CSV avec différentes températures fonctionne")
```

### Cellule : Test de performance
```python
# Mesurer les performances des fonctions principales
import time
import torch
from metrics.cnn_eval import compute_ece, fit_temperature

# Test compute_ece
sizes = [100, 1000, 10000]
for n in sizes:
    probs = torch.rand(n, 9)
    probs = probs / probs.sum(dim=1, keepdim=True)
    y_true = torch.randint(0, 9, (n,))
    
    start = time.time()
    ece = compute_ece(probs, y_true)
    elapsed = time.time() - start
    print(f"compute_ece({n} samples): {elapsed*1000:.2f}ms")

# Test fit_temperature
for n in [100, 1000]:
    logits = torch.randn(n, 9)
    y_true = torch.randint(0, 9, (n,))
    
    start = time.time()
    scaler = fit_temperature(logits, y_true, max_iter=100)
    elapsed = time.time() - start
    print(f"fit_temperature({n} samples, 100 iter): {elapsed:.2f}s")
```

## 🎯 Priorités

**Haute priorité** :
1. Tests de robustesse (CSV invalides, NaN/Inf)
2. Tests de cohérence (métriques cohérentes)
3. Tests de réutilisation (fichiers réutilisables)

**Moyenne priorité** :
4. Tests unitaires des fonctions individuelles
5. Tests d'intégration du pipeline complet

**Basse priorité** :
6. Tests de performance
7. Tests de visualisation

