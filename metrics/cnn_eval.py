# p9dg/metrics/cnn_eval.py
# -*- coding: utf-8 -*-
"""
Module d'évaluation pour modèles CNN de classification d'images histologiques.

Ce module fournit un pipeline complet d'évaluation incluant :
- Export des prédictions vers CSV (logits, confidences, chemins d'images)
- Calibration des probabilités via Temperature Scaling
- Calcul de métriques (accuracy, F1-score, ECE, Brier score)
- Génération de visualisations (matrice de confusion, reliability diagram)

Usage typique :
    from p9dg.metrics.cnn_eval import run_eval_split
    
    df, report = run_eval_split(
        model=model,
        dataloader=val_loader,
        device=DEVICE,
        out_dir=OUTPUTS_DIR,
        model_name="mobilenetv2",
        split="val",
        class_to_idx=train_ds.class_to_idx,
        fit_temperature_on_val=True,
        make_plots=True
    )
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Utils: classes & mappings
# ==========================
def _resolve_classes(class_to_idx: Optional[Dict[str, int]], n_classes: int) -> List[str]:
    """Retourne la liste ordonnée des noms de classes par index.
    
    Essaie d'utiliser le mapping du projet (p9dg/utils/class_mappings) pour obtenir
    les noms français des classes. Sinon, utilise un fallback générique.
    
    Args:
        class_to_idx: Dictionnaire mapping nom de classe → index (ex: {"ADI": 0, "TUM": 1})
        n_classes: Nombre total de classes
        
    Returns:
        Liste des noms de classes ordonnée par index (ex: ["Tissu adipeux", "Épithélium adénocarcinomateux"])
    """
    if class_to_idx is not None:
        try:
            # Essaye le mapping projet pour noms FR + couleurs
            from p9dg.utils.class_mappings import make_idx_mappings  # type: ignore
            idx_to_name, _, _ = make_idx_mappings(class_to_idx)
            return [idx_to_name[i] for i in range(len(idx_to_name))]
        except Exception:
            # Fallback: ordonner par index
            idx_to_name = {v: k for k, v in class_to_idx.items()}
            return [idx_to_name[i] for i in range(len(idx_to_name))]
    # Fallback minimal si on n'a pas de mapping
    return [f"class_{i}" for i in range(n_classes)]

# ==========================
# Calibration: ECE + T-scaling
# ==========================
def compute_ece(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15) -> float:
    """Calcule l'Expected Calibration Error (ECE) avec binning uniforme.
    
    L'ECE mesure l'écart entre la confiance moyenne et l'accuracy moyenne dans chaque bin.
    Un modèle bien calibré a un ECE proche de 0.
    
    Args:
        probs: Tensor de probabilités (N, C) où N est le nombre d'échantillons et C le nombre de classes
        y_true: Tensor des labels réels (N,)
        n_bins: Nombre de bins pour le binning uniforme (défaut: 15)
        
    Returns:
        Valeur de l'ECE (float entre 0 et 1, plus bas est mieux)
    """
    probs = probs.to(torch.float32)
    y_true = y_true.to(torch.long)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(y_true)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = accuracies[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_bin - acc_bin) * prop
    return float(ece.item())

class TempScaler(torch.nn.Module):
    """Module PyTorch pour la calibration par Temperature Scaling.
    
    La température T est apprise pour calibrer les probabilités du modèle.
    Les logits sont divisés par T avant l'application du softmax.
    """
    def __init__(self, init_temp: float = 1.0):
        """Initialise le scaler avec une température initiale.
        
        Args:
            init_temp: Température initiale (défaut: 1.0, pas de scaling)
        """
        super().__init__()
        self.log_t = torch.nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applique le scaling de température aux logits.
        
        Args:
            logits: Tensor de logits (N, C)
            
        Returns:
            Logits divisés par la température exp(log_t)
        """
        return logits / torch.exp(self.log_t)

def fit_temperature(logits_val: torch.Tensor, y_val: torch.Tensor,
                    max_iter: int = 1000, lr: float = 0.01) -> TempScaler:
    """Ajuste la température T en minimisant la Cross-Entropy sur le set de validation.
    
    Utilise l'optimiseur LBFGS pour trouver la température optimale qui améliore
    la calibration des probabilités sans changer les prédictions (argmax).
    
    Args:
        logits_val: Logits du modèle sur le set de validation (N, C)
        y_val: Labels réels (N,)
        max_iter: Nombre maximum d'itérations pour LBFGS (défaut: 1000)
        lr: Taux d'apprentissage pour LBFGS (défaut: 0.01)
        
    Returns:
        TempScaler entraîné avec la température optimale
    """
    scaler = TempScaler().to(logits_val.device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = criterion(scaler(logits_val), y_val)
        loss.backward()
        return loss
    opt.step(closure)
    return scaler

def load_temperature(json_path: str | Path) -> Optional[float]:
    """Charge la température depuis un fichier JSON.
    
    Format attendu : {"temperature": <float>}
    
    Args:
        json_path: Chemin vers le fichier JSON contenant la température
        
    Returns:
        Valeur de la température si le fichier existe et est valide, None sinon
    """
    p = Path(json_path)
    if p.exists():
        try:
            with open(p, "r") as f:
                return float(json.load(f).get("temperature", 1.0))
        except Exception:
            return None
    return None

# ==========================
# Export prédictions → CSV
# ==========================
@torch.inference_mode()
def export_predictions(model: torch.nn.Module,
                       dataloader,
                       device: torch.device,
                       out_csv: str | Path,
                       split: str = "val",
                       class_to_idx: Optional[Dict[str, int]] = None,
                       temperature: Optional[float] = None) -> pd.DataFrame:
    """Fait l'inférence sur un dataloader et exporte les résultats vers un CSV.
    
    Le CSV contient pour chaque image :
    - split: Nom du split (train/val/test)
    - image_path: Chemin vers l'image
    - y_true: Label réel (index)
    - y_pred: Prédiction (index)
    - top1_conf: Confiance de la prédiction (probabilité max)
    - logits_json: Logits complets au format JSON
    - top1_conf_cal: Confiance calibrée (si temperature fourni)
    
    Args:
        model: Modèle PyTorch en mode eval()
        dataloader: DataLoader retournant (images, labels, paths)
        device: Device PyTorch (cuda/cpu)
        out_csv: Chemin de sortie pour le CSV
        split: Nom du split pour le CSV (défaut: "val")
        class_to_idx: Mapping classe → index (optionnel, pour documentation)
        temperature: Température pour calibration (optionnel)
        
    Returns:
        DataFrame pandas avec toutes les prédictions
    """
    use_amp = (getattr(device, "type", str(device)) == "cuda")
    rows = []
    for imgs, labels, paths in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
        else:
            logits = model(imgs)

        probs = softmax(logits, dim=1)
        preds = probs.argmax(1)
        top1 = probs.max(1).values

        # température optionnelle
        top1_cal = None
        if temperature is not None:
            probs_cal = torch.softmax(logits / float(temperature), dim=1)
            top1_cal = probs_cal.max(1).values

        for i, p in enumerate(paths):
            row = {
                "split": split,
                "image_path": p,
                "y_true": int(labels[i]),
                "y_pred": int(preds[i]),
                "top1_conf": float(top1[i].cpu()),
                "logits_json": json.dumps(logits[i].detach().cpu().tolist(), separators=(",", ":")),
            }
            if top1_cal is not None:
                row["top1_conf_cal"] = float(top1_cal[i].cpu())
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] export → {out_csv}  ({len(df)} lignes)")
    return df

# ==========================
# Fit T depuis un CSV (val)
# ==========================
def fit_temperature_from_csv(val_csv: str | Path, out_json: str | Path) -> float:
    """Charge les logits depuis un CSV, ajuste la température et la sauvegarde.
    
    Cette fonction permet de réutiliser des prédictions déjà exportées pour
    ajuster la température sans refaire l'inférence complète.
    
    Args:
        val_csv: Chemin vers le CSV contenant les prédictions (doit avoir 'logits_json' et 'y_true')
        out_json: Chemin de sortie pour sauvegarder la température au format JSON
        
    Returns:
        Valeur de la température optimale trouvée
    """
    df = pd.read_csv(val_csv)
    logits = torch.tensor(
        np.stack(df["logits_json"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))),
        dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    y_true = torch.tensor(df["y_true"].values, dtype=torch.long, device=logits.device)
    T = fit_temperature(logits, y_true)
    temp_value = float(torch.exp(T.log_t).item())
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"temperature": temp_value}, f)
    print(f"[OK] Température sauvegardée → {out_json}  (T={temp_value:.3f})")
    return temp_value

# ==========================
# Évaluer un CSV (métriques + figures)
# ==========================
@dataclass
class EvalReport:
    """Rapport d'évaluation contenant toutes les métriques et chemins des figures.
    
    Attributes:
        split: Nom du split évalué (train/val/test)
        n: Nombre d'échantillons évalués
        acc_micro: Accuracy micro (globale)
        f1_macro: F1-score macro (moyenne des F1 par classe)
        ece_raw: Expected Calibration Error avant calibration
        ece_cal: Expected Calibration Error après calibration (None si pas de calibration)
        brier_raw: Brier score avant calibration
        brier_cal: Brier score après calibration (None si pas de calibration)
        classification_report: Rapport de classification textuel (sklearn)
        cm_path: Chemin vers la figure de matrice de confusion (None si plot=False)
        reliability_path: Chemin vers le reliability diagram (None si plot=False)
    """
    split: str
    n: int
    acc_micro: float
    f1_macro: float
    ece_raw: float
    ece_cal: Optional[float]
    brier_raw: float
    brier_cal: Optional[float]
    classification_report: str
    cm_path: Optional[str]
    reliability_path: Optional[str]

def _one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(y.to(torch.long), num_classes=n_classes).to(torch.float32)

def evaluate_csv_metrics(csv_path: str | Path,
                         classes: Optional[List[str]] = None,
                         class_to_idx: Optional[Dict[str, int]] = None,
                         temp_json: Optional[str | Path] = None,
                         out_dir: Optional[str | Path] = None,
                         normalize_cm: bool = False,
                         plot: bool = True,
                         device: Optional[str | torch.device] = None) -> EvalReport:
    """Évalue un CSV de prédictions et génère métriques + visualisations.
    
    Cette fonction permet de réévaluer des prédictions déjà exportées, par exemple
    pour appliquer une température de calibration différente ou régénérer les figures.
    
    Args:
        csv_path: Chemin vers le CSV contenant les prédictions
        classes: Liste des noms de classes (optionnel, sera déduit de class_to_idx si fourni)
        class_to_idx: Mapping classe → index (optionnel, pour résoudre les noms de classes)
        temp_json: Chemin vers le JSON contenant la température pour calibration (optionnel)
        out_dir: Dossier de sortie pour les figures (défaut: même dossier que le CSV)
        normalize_cm: Si True, normalise la matrice de confusion par ligne (défaut: False)
        plot: Si True, génère les figures (défaut: True)
        device: Device à utiliser ("cpu", "cuda", ou torch.device). Si None, utilise CUDA si disponible (défaut: None)
        
    Returns:
        EvalReport contenant toutes les métriques et chemins des figures
    """
    df = pd.read_csv(csv_path)
    split = str(df["split"].iloc[0]) if "split" in df.columns else "unknown"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device)  # Convertir en string si c'est un torch.device

    logits = torch.tensor(
        np.stack(df["logits_json"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))),
        dtype=torch.float32, device=device
    )
    y_true = torch.tensor(df["y_true"].values, dtype=torch.long, device=device)
    probs  = torch.softmax(logits, dim=1)
    y_pred = probs.argmax(1)

    n_classes = probs.shape[1]
    if classes is None:
        classes = _resolve_classes(class_to_idx, n_classes)
    labels = list(range(n_classes))

    # Calibration optionnelle
    T = load_temperature(temp_json) if temp_json else None
    probs_cal = torch.softmax(logits / float(T), dim=1) if T else None

    # Métriques
    acc_micro = float((y_pred == y_true).float().mean().item())
    f1_macro  = float(f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro"))
    ece_raw   = compute_ece(probs, y_true)
    brier_raw = float(torch.mean((probs - _one_hot(y_true, n_classes)).pow(2).sum(dim=1)).item())

    ece_cal = brier_cal = None
    if probs_cal is not None:
        ece_cal   = compute_ece(probs_cal, y_true)
        brier_cal = float(torch.mean((probs_cal - _one_hot(y_true, n_classes)).pow(2).sum(dim=1)).item())

    # Rapport texte
    cls_report = classification_report(
        y_true.cpu().numpy(), y_pred.cpu().numpy(),
        target_names=classes, labels=labels, digits=3, zero_division=0
    )

    # Figures
    cm_path = rel_path = None
    if plot:
        out_dir = Path(out_dir) if out_dir else Path(csv_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # CM
        cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy(), labels=labels, normalize="true" if normalize_cm else None)
        plt.figure(figsize=(8, 6))
        fmt = ".2f" if normalize_cm else "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Prédit"); plt.ylabel("Réel")
        title_norm = "normalisée" if normalize_cm else "absolue"
        plt.title(f"Matrice de confusion ({title_norm}) — {split}")
        plt.tight_layout()
        cm_path = str(out_dir / f"cm_{split}.png")
        plt.savefig(cm_path, dpi=160)
        plt.close()

        # Reliability
        def _reliability_curve(P: torch.Tensor, Y: torch.Tensor, n_bins: int = 15):
            conf, pred = P.max(1)
            acc = pred.eq(Y)
            bins = torch.linspace(0, 1, n_bins + 1, device=P.device)
            xs, ys = [], []
            for i in range(n_bins):
                sel = (conf > bins[i]) & (conf <= bins[i+1])
                if sel.any():
                    xs.append(conf[sel].float().mean().item())
                    ys.append(acc[sel].float().mean().item())
            return np.array(xs), np.array(ys)

        x_raw, y_raw = _reliability_curve(probs, y_true)
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", label="Idéal")
        plt.plot(x_raw, y_raw, marker="o", label=f"Brut (ECE={ece_raw:.3f})")
        if probs_cal is not None:
            x_cal, y_cal = _reliability_curve(probs_cal, y_true)
            plt.plot(x_cal, y_cal, marker="o", label=f"Calibré T={T:.2f} (ECE={ece_cal:.3f})")
        plt.xlabel("Confiance moyenne (bin)")
        plt.ylabel("Accuracy (bin)")
        plt.title(f"Reliability diagram — {split}")
        plt.legend()
        plt.tight_layout()
        rel_path = str(out_dir / f"reliability_{split}.png")
        plt.savefig(rel_path, dpi=160)
        plt.close()

    return EvalReport(
        split=split, n=len(df),
        acc_micro=acc_micro, f1_macro=f1_macro,
        ece_raw=ece_raw, ece_cal=ece_cal,
        brier_raw=brier_raw, brier_cal=brier_cal,
        classification_report=cls_report,
        cm_path=cm_path, reliability_path=rel_path
    )

# ==========================
# Orchestrateur principal
# ==========================
def run_eval_split(model: torch.nn.Module,
                   dataloader,
                   device: torch.device,
                   out_dir: str | Path,
                   model_name: str,
                   split: str,
                   class_to_idx: Optional[Dict[str, int]] = None,
                   fit_temperature_on_val: bool = False,
                   temp_json: Optional[str | Path] = None,
                   export_csv_name: Optional[str] = None,
                   make_plots: bool = True) -> Tuple[pd.DataFrame, EvalReport]:
    """Orchestrateur principal pour l'évaluation complète d'un split.
    
    Cette fonction automatise le pipeline complet d'évaluation :
    1. Export des prédictions vers CSV (avec confiance calibrée si température fournie)
    2. (Optionnel) Ajustement de la température sur le set de validation
    3. Calcul des métriques et génération des visualisations
    
    Les fichiers générés sont :
    - <model_name>_preds_<split>.csv : Prédictions avec logits et confidences
    - <model_name>_temp_scaling.json : Température de calibration (si fit_temperature_on_val=True)
    - cm_<split>.png : Matrice de confusion
    - reliability_<split>.png : Reliability diagram
    
    Args:
        model: Modèle PyTorch en mode eval()
        dataloader: DataLoader retournant (images, labels, paths)
        device: Device PyTorch (cuda/cpu)
        out_dir: Dossier de sortie pour tous les fichiers générés
        model_name: Nom du modèle (utilisé pour nommer les fichiers)
        split: Nom du split (train/val/test)
        class_to_idx: Mapping classe → index (optionnel, pour résoudre les noms de classes)
        fit_temperature_on_val: Si True et split=="val", ajuste la température sur ce split
        temp_json: Chemin vers un JSON de température existant (optionnel)
        export_csv_name: Nom personnalisé pour le CSV (défaut: <model_name>_preds_<split>.csv)
        make_plots: Si True, génère les figures (défaut: True)
        
    Returns:
        Tuple (DataFrame, EvalReport) contenant les prédictions et le rapport d'évaluation
        
    Example:
        >>> from p9dg.metrics.cnn_eval import run_eval_split
        >>> df_val, report_val = run_eval_split(
        ...     model=model,
        ...     dataloader=val_loader,
        ...     device=torch.device("cuda"),
        ...     out_dir="outputs/",
        ...     model_name="mobilenetv2",
        ...     split="val",
        ...     class_to_idx=train_ds.class_to_idx,
        ...     fit_temperature_on_val=True,
        ...     make_plots=True
        ... )
        >>> print(f"Accuracy: {report_val.acc_micro:.4f}")
        >>> print(f"ECE (raw): {report_val.ece_raw:.4f}")
        >>> print(f"ECE (cal): {report_val.ece_cal:.4f}")
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) température connue ?
    T = load_temperature(temp_json) if temp_json else None
    # 2) export CSV
    csv_path = out_dir / (export_csv_name or f"{model_name}_preds_{split}.csv")
    export_predictions(model, dataloader, device, csv_path, split=split,
                       class_to_idx=class_to_idx, temperature=T)

    # 3) fit T sur val si demandé
    if split == "val" and fit_temperature_on_val:
        tj = out_dir / f"{model_name}_temp_scaling.json"
        T = fit_temperature_from_csv(csv_path, tj)
        temp_json = str(tj)  # pour l'étape 4

        # réécrire un CSV avec conf calibrée (optionnel)
        export_predictions(model, dataloader, device, csv_path, split=split,
                           class_to_idx=class_to_idx, temperature=T)

    # 4) évaluer CSV
    report = evaluate_csv_metrics(csv_path, classes=None, class_to_idx=class_to_idx,
                                  temp_json=temp_json, out_dir=out_dir, plot=make_plots)

    # + renvoyer le DataFrame
    df = pd.read_csv(csv_path)
    return df, report
