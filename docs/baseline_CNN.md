
---

# 🧠 Baseline Classification CNN – README Technique

Ce document détaille la baseline de **classification histopathologique** utilisée dans le projet : pipeline complet, stratégie d’entraînement, préprocessing, métriques d’évaluation et calibration probabiliste.

---

# 1. 🎯 Objectifs de la baseline

* Fournir une **référence robuste** pour juger l’apport des images synthétiques (cGAN, PixCell).
* Évaluer les performances sur les **9 classes histopathologiques** définies dans `class_mappings.py`.
* Exporter des métriques détaillées pour les analyses downstream.

---

# 2. 📁 Jeu de données et splits

* **Train** : NCT-CRC-HE-100K
* **Validation / Test** : CRC-VAL-HE-7K
* **Aucune fuite** : les images sélectionnées pour la génération synthétique sont explicitement exclues du test.
* Possibilité d'équilibrer le train via un sampler.

Les classes traitées :

```python
{
  "ADI":  "Tissu adipeux",
  "BACK": "Arrière-plan",
  "DEB":  "Débris / nécrose",
  "LYM":  "Lymphocytes",
  "MUC":  "Mucus",
  "MUS":  "Muscle lisse",
  "NORM": "Muqueuse normale",
  "STR":  "Stroma",
  "TUM":  "Épithélium tumoral"
}
```

---

# 3. 🧼 Préprocessing avancé (HistoDataset)

## 3.1 Chargement d’image

* Lecture PIL → conversion RGB.
* Redimensionnement en **256×256**.
* Normalisation **ImageNet** pour MobileNetV2.

## 3.2 Filtre qualité

Implémenté dans `QualityFilter` (features calculées via NumPy/Scipy) :

* **Laplace variance** (netteté)
* **Entropie**
* **Tenengrad** (gradient)
* **White ratio**, **Saturation ratio**
* **Blockiness spatial et DCT**
* **Tissue fraction**

Règles spécifiques :

* `BACK` : toujours accepté.
* `ADI` : seuils permissifs.
* Autres classes : seuils stricts.

## 3.3 Normalisation de coloration (optionnelle)

* Basée sur **Vahadane** via `torch_staintools`.
* Initialisation stable + fallback automatique en cas d’erreur.
* Compatible CPU/GPU.

---

# 4. 🧠 Modèle – MobileNetV2

* Backbone pré-entraîné ImageNet.
* Tête remplacée par un **classifier 9 classes**.
* Activation finale : logits (pas de softmax dans le modèle).

Avantages :

* Très léger, rapide en inférence.
* Bon compromis entre vitesse et expressivité.

---

# 5. ⚙️ Stratégie d’entraînement

## 5.1 Fine-tuning progressif

* **Phase 1** : backbone gelé → stabilisation des gradients.
* **Phase 2** : dégel progressif des derniers blocs pour adapter aux motifs H&E.

## 5.2 Optimisation

* Optimiseur : Adam ou SGD (selon notebook).
* **Scheduler** : décroissance du LR ou ReduceLROnPlateau.
* **Early stopping** sur la perte validation.

## 5.3 DataLoader

* GPU-friendly : `pin_memory=True`, `non_blocking=True`.
* Shuffle systématique.
* Retour `(image, label, path)` pour analyses downstream.

---

# 6. 📊 Pipeline d’évaluation (cnn_eval.py)

Le module `cnn_eval.py` offre un pipeline complet d’évaluation et de calibration.

### 6.1 Export CSV

Pour chaque image :

* chemin
* label réel
* prédiction
* confiance brute
* **logits JSON**
* confiance calibrée (optionnel)

Exemple :

```python
df = export_predictions(
    model, dataloader, device,
    out_csv="preds_val.csv",
    split="val",
    class_to_idx=train_ds.class_to_idx
)
```

### 6.2 Métriques calculées

* **Accuracy (micro)** : performance globale.
* **F1 macro** : indispensable pour dataset déséquilibré.
* **Matrice de confusion**.
* **Reliability diagram** (calibration).
* **ECE** : Expected Calibration Error.
* **Brier score** : qualité probabiliste.

---

# 7. 🎛️ Calibration – Temperature Scaling (T-scaling)

## 7.1 Pourquoi calibrer ?

Les CNN sont souvent **trop confiants**.
La calibration ajuste uniquement les probabilités, pas les prédictions.

## 7.2 Principe

On apprend un scalaire **T** tel que :

[
p_i = \text{softmax}(z_i / T)
]

* Si **T > 1** → modèle *moins confiant*.
* Si **T < 1** → modèle *plus confiant*.
* Ajustement global, simple, très efficace.

## 7.3 Comment T est appris ?

* On récupère les **logits val** via le CSV.
* On minimise la **CrossEntropy** avec LBFGS.
* On sauvegarde T dans un `.json`.

```python
T = fit_temperature_from_csv("preds_val.csv", "temperature.json")
```

## 7.4 Application à l’inférence test

Les logits sont divisés par T avant le softmax.

---

# 8. 🧮 Métrologies clés (ECE, Brier, etc.)

## 8.1 ECE – Expected Calibration Error

* On discretise les prédictions par bins de confiance (10 ou 15 bins).
* Pour chaque bin :

  * **confiance moyenne**
  * **accuracy moyenne**
* ECE = somme pondérée des écarts |acc – conf|.

Interprétation :

* **0%** = calibration parfaite.
* **>5%** = sur-confiance typique des CNN.

## 8.2 Brier score

[
\text{Brier}=\frac{1}{N}\sum(p_{\text{pred}} - y_{\text{true}})^2
]

* Plus bas = mieux.
* Mesure la qualité probabiliste brute.

## 8.3 Accuracy & F1 macro

* Accuracy = performance globale.
* F1 macro = équilibre inter-classes, critique en histopathologie (certaines classes rares).

---

# 9. 🧪 Résultats typiques

(À adapter selon le run.)

* Bonnes performances sur classes fréquentes : **NORM**, **STR**, **TUM**.
* Scores plus faibles sur classes rares : **LYM**, **DEB**, **MUS**.
* **Calibration améliore nettement l’ECE** sans modifier les prédictions.
* Baseline robuste pour comparaison **real vs synth (cGAN, PixCell)**.

---

# 10. 📦 Artefacts générés

* `preds_val.csv`, `preds_test.csv`
* `temperature.json`
* `confusion_matrix.png`
* `reliability_diagram.png`
* `metrics.json`

---

# 11. 🚀 Rôle dans le projet

Cette baseline sert de **référence principale** pour :

* évaluer l’utilité des images synthétiques,
* mesurer la cohérence du classifieur sur données GAN/diffusion,
* produire les analyses downstream (PCR, Consistency, KL, …).

---
