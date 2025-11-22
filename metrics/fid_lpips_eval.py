"""
FID + LPIPS multi-classes evaluator
-----------------------------------
Permet de calculer les métriques de similarité entre un dossier réel et un dossier synthétique.

Modes :
- Multi-classe (par défaut) : toutes les sous-classes détectées sont évaluées.
- Mono-classe ou sous-ensemble : une ou plusieurs classes spécifiques peuvent être fournies.

Utilisation :
    from metrics.fid_lpips_eval import FIDLPIPSEvaluator
    evaluator = FIDLPIPSEvaluator(gen_root="outputs/preprocessing/normalized_tst", classes="TUM")
    evaluator.run()
"""

from pathlib import Path
import numpy as np
import random, shutil, tempfile, warnings, os
import pandas as pd
from tqdm import tqdm
import torch
from torch_fidelity import calculate_metrics
import lpips
from torchvision import transforms
from PIL import Image

# --- Utils pour FID_UNI ---
from typing import Callable, Optional, List
def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    """Racine de matrice pour PSD via décomp. propres (fallback si SciPy absent)."""
    try:
        import scipy.linalg as la  # type: ignore
        return la.sqrtm(A).real
    except Exception:
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 0, None)
        return (V * np.sqrt(w)) @ V.T

def _frechet_from_feats(F1: np.ndarray, F2: np.ndarray) -> float:
    """Fréchet distance entre deux nuages d'embeddings (Nxd, Mxd)."""
    mu1, mu2 = F1.mean(0), F2.mean(0)
    c1, c2 = np.cov(F1, rowvar=False), np.cov(F2, rowvar=False)
    diff = mu1 - mu2
    covmean = _sqrtm_psd(c1 @ c2)
    return float(diff @ diff + np.trace(c1 + c2 - 2.0 * covmean))

# ===================================================================
# 🧮 Classe principale
# ===================================================================
class FIDLPIPSEvaluator:
    def __init__(
        self,
        real_root: str | Path = "/workspace/data/NCT-CRC-HE-100K",
        gen_root: str | Path | None = None,
        save_dir: str | Path | None = None,
        max_images: int = 400,
        seed: int = 42,
        classes: str | list[str] | None = None,
        device: str | None = None,
        uni_embed: Optional[Callable[[str], np.ndarray]] = None,  # callable: path -> (d,)
        fid_uni_max: Optional[int] = None,                        # None => utilise max_images
        lpips_pairs: int = 50,                                    # nb max de paires LPIPS par classe
    ):
        self.real_root = Path(real_root).resolve()
        self.gen_root = Path(gen_root).resolve() if gen_root else None
        self.save_dir = Path(save_dir).resolve() if save_dir else Path("./artifacts")
        self.max_images = int(max_images)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.uni_embed = uni_embed
        self.fid_uni_max = fid_uni_max
        self.lpips_pairs = int(lpips_pairs)

        # --- Validation minimale des chemins ---
        if not self.real_root.exists():
            raise FileNotFoundError(f"Chemin réel introuvable : {self.real_root}")
        if not self.gen_root or not self.gen_root.exists():
            raise FileNotFoundError("⚠️ Dossier synthétique introuvable ou non spécifié.")

        # --- Création du dossier de sauvegarde ---
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- Vérification des sous-dossiers disponibles ---
        real_classes = sorted([d.name for d in self.real_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        gen_classes  = sorted([d.name for d in self.gen_root.iterdir()  if d.is_dir() and not d.name.startswith(".")])

        # --- Gestion du paramètre `classes` (intersection par défaut) ---
        if classes is None:
            selected = sorted(set(real_classes) & set(gen_classes))
        elif isinstance(classes, str):
            selected = [classes]
        else:
            selected = list(classes)

        # Vérification stricte des classes demandées
        missing_real = [c for c in selected if c not in real_classes]
        missing_gen  = [c for c in selected if c not in gen_classes]
        if missing_real:
            raise ValueError(f"⚠️ Classes absentes du dossier réel : {missing_real}")
        if missing_gen:
            raise ValueError(f"⚠️ Classes absentes du dossier synthétique : {missing_gen}")

        if len(selected) == 0:
            raise ValueError("Aucune classe commune entre real_root et gen_root.")

        self.class_labels = selected

        # --- Nom dynamique du rapport ---
        self.save_name = f"{self.real_root.name}_{self.gen_root.name}.csv"
        if len(self.class_labels) != len(real_classes):
            cls_suffix = "_".join(self.class_labels)
            self.save_name = f"{self.real_root.name}_{self.gen_root.name}_{cls_suffix}.csv"
        self.save_path = self.save_dir / self.save_name

        # --- Masquage des warnings externes ---
        warnings.filterwarnings("ignore", category=FutureWarning, module="lpips.lpips")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity.datasets")
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

        # --- Initialisation LPIPS ---
        self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        print(f"✅ Initialisé : {self.real_root.name} vs {self.gen_root.name}")
        print(f"📂 Classes évaluées : {', '.join(self.class_labels)}")
        print(f"💾 Rapport : {self.save_path}")

    # -------------------------------------------------------------------
    # 🔧 Prépare un sous-ensemble d'images (converties en PNG) pour FID
    # -------------------------------------------------------------------
    def _prepare_subset_for_fid(self, src_dir: Path) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"fid_subset_{src_dir.name}_"))
        all_files = [f for f in src_dir.iterdir()
                     if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")]
        if not all_files:
            raise ValueError(f"Aucune image trouvée dans {src_dir}")
        random.seed(self.seed)
        subset = random.sample(all_files, min(self.max_images, len(all_files)))
        for file in subset:
            dst = tmp_dir / (file.stem + ".png")
            try:
                Image.open(file).convert("RGB").save(dst)
            except Exception as e:
                print(f"⚠️ Erreur conversion {file.name}: {e}")
        return tmp_dir

    # -------------------------------------------------------------------
    # 🎨 LPIPS moyen entre deux dossiers d'images (mêmes index)
    # -------------------------------------------------------------------
    def _compute_lpips_mean(self, real_dir: Path, gen_dir: Path, max_pairs: int = 50) -> float:
        reals = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(".png")])
        gens  = sorted([f for f in os.listdir(gen_dir)  if f.lower().endswith(".png")])
        n = min(len(reals), len(gens), max_pairs)
        if n == 0:
            return float("nan")
        pairs = list(zip(reals[:n], gens[:n]))
        scores = []
        for r, g in pairs:
            img1 = self.to_tensor(Image.open(real_dir / r).convert("RGB")).unsqueeze(0)
            img2 = self.to_tensor(Image.open(gen_dir / g).convert("RGB")).unsqueeze(0)
            if self.device == "cuda":
                img1, img2 = img1.cuda(), img2.cuda()
            with torch.no_grad():
                d = self.lpips_model(img1, img2)
            scores.append(float(d.item()))
        return float(sum(scores) / len(scores))

    # -------------------------------------------------------------------
    # 🎨 FID dans l'espace UNI (optionnel)
    # -------------------------------------------------------------------
    def _compute_fid_uni(self, png_dir_real: Path, png_dir_gen: Path) -> float | None:
        if self.uni_embed is None:
            return None
        reals = sorted([p for p in png_dir_real.iterdir() if p.suffix.lower() == ".png"])
        gens  = sorted([p for p in png_dir_gen.iterdir()  if p.suffix.lower() == ".png"])
        if not reals or not gens:
            return None
        k = min(len(reals), len(gens), self.fid_uni_max or self.max_images)
        if k < 2:
            return float("nan")
        reals, gens = reals[:k], gens[:k]
        R = np.stack([self.uni_embed(str(p)) for p in reals], axis=0).astype(np.float64)
        G = np.stack([self.uni_embed(str(p)) for p in gens],  axis=0).astype(np.float64)
        return _frechet_from_feats(R, G)

    # -------------------------------------------------------------------
    # 🚀 Boucle principale d’évaluation multi-classes
    # -------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        rows = []
        for c in tqdm(self.class_labels, desc="Évaluation multi-classes"):
            real_sub = gen_sub = None
            try:
                real_dir = self.real_root / c
                gen_dir  = self.gen_root  / c

                # Sous-ensembles convertis en PNG pour FID/KID/PRC & LPIPS
                real_sub = self._prepare_subset_for_fid(real_dir)
                gen_sub  = self._prepare_subset_for_fid(gen_dir)

                # Comptages & tolérance petits volumes
                n_real = len([p for p in real_sub.iterdir() if p.suffix.lower()==".png"])
                n_synth= len([p for p in gen_sub.iterdir()  if p.suffix.lower()==".png"])
                k = min(n_real, n_synth)
                if k < 2:
                    lp = self._compute_lpips_mean(real_sub, gen_sub, max_pairs=min(self.lpips_pairs, k))
                    rows.append({
                        "class": c, "n_real": n_real, "n_synth": n_synth,
                        "FID": float("nan"), "KID_mean": float("nan"), "KID_std": float("nan"),
                        "LPIPS": lp, "FID_UNI": float("nan"),
                    })
                    continue

                # KID adapté aux petits volumes
                kid_ss = min(k, 50)

                # FID/KID (PRC désactivé en mode dossiers : n'a de sens qu'en mode pairé)
                metrics = calculate_metrics(
                    input1=str(real_sub),
                    input2=str(gen_sub),
                    cuda=(self.device == "cuda"),
                    isc=False,
                    fid=True,
                    kid=True,
                    prc=False,  # PRC nécessite des paires alignées, pas adapté au mode dossiers
                    kid_subset_size=kid_ss,
                    verbose=False,
                )
                fid       = metrics.get("frechet_inception_distance", None)
                kid_mean  = metrics.get("kernel_inception_distance_mean", None)
                kid_std   = metrics.get("kernel_inception_distance_std", None)

                # LPIPS (même sous-ensemble)
                lp = self._compute_lpips_mean(real_sub, gen_sub, max_pairs=min(self.lpips_pairs, k))

                # FID_UNI (optionnel)
                fid_uni = self._compute_fid_uni(real_sub, gen_sub)

                rows.append({
                    "class": c, "n_real": n_real, "n_synth": n_synth,
                    "FID": fid, "KID_mean": kid_mean, "KID_std": kid_std,
                    "LPIPS": lp, "FID_UNI": fid_uni,
                })

            except Exception as e:
                print(f"❌ Erreur sur {c}: {e}")
                rows.append({
                    "class": c, "n_real": 0, "n_synth": 0,
                    "FID": float("nan"), "KID_mean": float("nan"), "KID_std": float("nan"),
                    "LPIPS": float("nan"), "FID_UNI": float("nan"),
                })
            finally:
                # Nettoyage des sous-ensembles temporaires
                if real_sub and Path(real_sub).exists():
                    shutil.rmtree(real_sub, ignore_errors=True)
                if gen_sub and Path(gen_sub).exists():
                    shutil.rmtree(gen_sub,  ignore_errors=True)

        # DataFrame & sauvegarde
        df = pd.DataFrame(rows)
        if "class" in df.columns:
            df = df.sort_values("class").reset_index(drop=True)
        out_cols = ["class","n_real","n_synth","FID","KID_mean","KID_std","LPIPS","FID_UNI"]
        for col in out_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[out_cols]
        df.to_csv(self.save_path, index=False)
        print(f"\n✅ Rapport enregistré : {self.save_path}")
        return df

# ===================================================================
# 🚀 Exécution directe (CLI)
#   Exemples :
#   1) Mode dossiers (réel vs synthé)
#      python fid_lpips_eval.py \
#        --name real_vs_pixcell_kid_prc_uni \
#        --real-root ./outputs_pixcell_synth/train/real \
#        --gen-root  ./outputs_pixcell_synth/train/synth \
#        --max-images-per-class 50 --lpips-pairs 50 \
#        --use-uni --fid-uni-max 50 --seed 42
#
#   2) Mode CSV pairé
#      python fid_lpips_eval.py \
#        --name paired_pixcell_kid_prc_uni \
#        --pairs-csv ./artifacts/pixcell_metadata.csv \
#        --lpips-pairs 50 --use-uni --fid-uni-max 50 --seed 42
# ===================================================================
if __name__ == "__main__":
    import argparse, os

    def _parse_classes(s: str | None):
        if not s:
            return None
        return [c.strip() for c in s.split(",") if c.strip()]

    # --- Embedder UNI-2h (identique à celui des notebooks 07/08) ---
    def make_uni2h_embedder_timm(device=None, hf_token=None):
        import torch, timm, numpy as np
        from PIL import Image
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

        timm_kwargs = dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, embed_dim=1536,
            mlp_ratio=2.66667*2, init_values=1e-5, num_classes=0, no_embed_class=True,
            reg_tokens=8, dynamic_img_size=True, mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).eval().to(device)
        cfg = resolve_data_config(uni.pretrained_cfg, model=uni)
        tfm = create_transform(**cfg)

        @torch.no_grad()
        def uni_embed(path: str) -> np.ndarray:
            x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            feat = uni(x)
            if feat.ndim == 4:
                feat = feat.mean((-2, -1))
            return feat.squeeze(0).float().cpu().numpy().astype(np.float64)
        return uni_embed

    parser = argparse.ArgumentParser(description="FID/KID/PRC + LPIPS (+ FID_UNI) pour réel vs synthé (dossiers) ou CSV pairé.")
    parser.add_argument("--name", required=True, help="Nom de l'expérience (préfixe du CSV).")
    parser.add_argument("--save-dir", default="./artifacts", help="Dossier de sortie des CSV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lpips-pairs", type=int, default=50, help="Nombre max de paires LPIPS par classe.")

    # Modes exclusifs : dossiers OU CSV pairé
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs-csv", help="CSV PixCell pour l'évaluation pairée.")
    group.add_argument("--gen-root", help="Racine des images synthétiques (mode dossiers).")

    # Args mode dossiers
    parser.add_argument("--real-root", help="Racine des images réelles (mode dossiers).")
    parser.add_argument("--classes", help="Liste de classes séparées par des virgules (sinon intersection real∩gen).")
    parser.add_argument("--drop-back-variant", action="store_true", default=False,
                        help="Ignore les sous-dossiers BACK si présent (géré dans run_eval_experiment).")
    parser.add_argument("--max-images-per-class", type=int, default=400, help="Plafond d'images par classe.")

    # Args mode CSV pairé
    parser.add_argument("--real-col", default="ref_path")
    parser.add_argument("--gen-col",  default="out_path")
    parser.add_argument("--class-col", default="ref_label")

    # FID_UNI (optionnel)
    parser.add_argument("--use-uni", action="store_true", help="Active FID_UNI (encodeur UNI-2h).")
    parser.add_argument("--fid-uni-max", type=int, default=None, help="Plafond d'images pour FID_UNI.")
    parser.add_argument("--hf-token", default=None, help="HF token (sinon lit HF_TOKEN / HUGGINGFACE_HUB_TOKEN).")
    
    # PRC (optionnel, uniquement en mode pairé)
    parser.add_argument("--compute-prc", action="store_true", default=True,
                        help="Active PRC (Precision/Recall) en mode pairé (désactivé par défaut en mode dossiers).")
    parser.add_argument("--no-prc", dest="compute_prc", action="store_false",
                        help="Désactive PRC même en mode pairé.")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Optionnel : init embedder UNI
    uni_embed = None
    if args.use_uni:
        uni_embed = make_uni2h_embedder_timm(hf_token=args.hf_token)

    # Dispatch selon le mode
    if args.pairs_csv:
        # --- Mode CSV pairé ---
        df = run_eval_paired_experiment(
            name=args.name,
            pairs_csv=args.pairs_csv,
            save_dir=args.save_dir,
            real_col=args.real_col,
            gen_col=args.gen_col,
            class_col=args.class_col if args.class_col.lower() != "none" else None,
            lpips_pairs=args.lpips_pairs,
            seed=args.seed,
            uni_embed=uni_embed,
            fid_uni_max=args.fid_uni_max,
            compute_prc=args.compute_prc,
        )
    else:
        # --- Mode dossiers ---
        if not args.real_root:
            parser.error("--real-root est requis en mode dossiers.")
        df = run_eval_experiment(
            name=args.name,
            real_root=args.real_root,
            gen_root=args.gen_root,
            classes=_parse_classes(args.classes),
            drop_back_variant=args.drop_back_variant,
            max_images_per_class=args.max_images_per_class,
            lpips_pairs=args.lpips_pairs,
            seed=args.seed,
            save_dir=args.save_dir,
            uni_embed=uni_embed,
            fid_uni_max=args.fid_uni_max,
        )
    # Affiche un aperçu
    try:
        print(df.head())
    except Exception:
        pass

# ===================================================================
# 🧩 Runner générique pour expérimentations
# ===================================================================

def run_eval_experiment(
    name: str,
    real_root: str | Path,
    gen_root: str | Path,
    classes: list[str] | None = None,
    drop_back_variant: bool = True,
    max_images_per_class: int = 400,
    lpips_pairs: int = 50,
    seed: int = 42,
    save_dir: str | Path = "./artifacts",
    uni_embed: Optional[Callable[[str], np.ndarray]] = None,
    fid_uni_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lance une évaluation FID/KID/PRC + LPIPS (+ FID_UNI optionnel) en
    échantillonnant au plus `max_images_per_class` images par classe.

    Ex:
        run_eval_experiment(
            name="real_vs_pixcell_kid_prc_uni",
            real_root="./outputs_pixcell_synth/train/real",
            gen_root="./outputs_pixcell_synth/train/synth",
            seed=42, uni_embed=uni_embed, fid_uni_max=50,
            max_images_per_class=50, lpips_pairs=50
        )
    """
    real_root = Path(real_root)
    gen_root = Path(gen_root)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -- Filtre des classes (intersection real ∩ gen), avec option BACK
    def _filter_classes(root: Path) -> list[str]:
        cls = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if drop_back_variant:
            cls = [c for c in cls if "BACK" not in c.upper()]
        return sorted(cls)

    if classes is None:
        classes = sorted(set(_filter_classes(real_root)) & set(_filter_classes(gen_root)))

    print(f"🎯 Expérience : {name}")
    print(f"📁 real_root = {real_root}")
    print(f"📁 gen_root  = {gen_root}")
    print(f"📦 classes évaluées : {classes}")
    print(f"🎲 seed = {seed}")

    # -- Cas sans classe commune : on écrit un CSV vide (entête standard) et on sort proprement
    if not classes:
        cols = ["class","n_real","n_synth","FID","KID_mean","KID_std","LPIPS","FID_UNI"]
        empty = pd.DataFrame(columns=cols)
        out_path = save_dir / f"{name}_metrics.csv"
        empty.to_csv(out_path, index=False)
        print(f"⚠️ Aucune classe commune — CSV vide écrit dans {out_path}")
        return empty

    # -- Initialisation de l'évaluator (attend lpips_pairs dans son __init__)
    evaluator = FIDLPIPSEvaluator(
        real_root=real_root,
        gen_root=gen_root,
        save_dir=save_dir,
        max_images=max_images_per_class,
        seed=seed,
        classes=classes,
        uni_embed=uni_embed,
        fid_uni_max=fid_uni_max,
        lpips_pairs=lpips_pairs,
    )

    # -- Exécution
    df = evaluator.run()
    df["experiment"] = name
    df["max_images_per_class"] = max_images_per_class
    df["lpips_pairs"] = lpips_pairs
    # Precision/Recall ne sont plus calculées dans FIDLPIPSEvaluator (mode dossiers)

    # -- Sauvegarde finale
    out_path = save_dir / f"{name}_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"💾 Résultats sauvegardés → {out_path}")

    return df


# ===================================================================
# 🧩 Runner pour évaluation FID/LPIPS à partir d’un CSV de paires réelles–synthétiques
#  pixcell_metadata_*.csv
# ===================================================================

def run_eval_paired_experiment(
    name: str,
    pairs_csv: str | Path,
    save_dir: str | Path = "./artifacts",
    real_col: str = "ref_path",
    gen_col: str = "out_path",
    class_col: str | None = "ref_label",   # None => regroupe tout en 'ALL'
    lpips_pairs: int | None = None,        # None = toutes les paires ; sinon échantillon aléatoire par classe
    seed: int = 42,
    uni_embed: Optional[Callable[[str], np.ndarray]] = None,
    fid_uni_max: Optional[int] = None,     # plafond images pour FID_UNI
    compute_prc: bool = True,              # PRC a du sens en mode pairé (paires alignées)
) -> pd.DataFrame:
    """
    Évalue FID/KID/PRC + LPIPS (+ FID_UNI optionnel) à partir d'un CSV PixCell contenant
    des paires réel/généré. Le FID est calculé par classe sur des sous-dossiers alignés.
    
    PRC (Precision/Recall) n'a de sens qu'en mode pairé car il nécessite des paires alignées
    entre réel et synthétique. En mode dossiers, les images ne sont pas appariées.
    """
    pairs_csv = Path(pairs_csv)
    df_pairs = pd.read_csv(pairs_csv)
    print(f"📑 CSV : {pairs_csv}  ({len(df_pairs)} lignes)")
    print(f"🔍 Colonnes détectées : {list(df_pairs.columns)}")

    # validations colonnes
    for col in (real_col, gen_col):
        if col not in df_pairs.columns:
            raise KeyError(f"Colonne manquante dans le CSV : '{col}'")
    if class_col is not None and class_col not in df_pairs.columns:
        print(f"⚠️ Colonne de classe '{class_col}' absente → regroupement en 'ALL'.")
        class_col = None

    # classes à traiter
    if class_col is None:
        df_pairs["__class__"] = "ALL"
        class_col = "__class__"
    classes = sorted(df_pairs[class_col].unique())

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # modèles / utils
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device)
    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    random.seed(seed)
    print(f"🎯 Expérience pairée : {name}  |  Device: {device}")
    print(f"📦 Classes : {classes}")

    results = []

    for c in tqdm(classes, desc="Évaluation pairée"):
        sub = df_pairs[df_pairs[class_col] == c]
        n_total = len(sub)

        # Échantillonnage optionnel (cohérent LPIPS/FID)
        if lpips_pairs is not None and lpips_pairs < n_total:
            sub = sub.sample(lpips_pairs, random_state=seed)
        n_pairs = len(sub)

        # Dossiers temporaires alignés pour FID/KID/PRC
        real_tmp = Path(tempfile.mkdtemp(prefix=f"real_{c}_"))
        gen_tmp  = Path(tempfile.mkdtemp(prefix=f"gen_{c}_"))

        try:
            lpips_scores = []
            used_pairs = 0
            uid = 0  # pour des noms .png uniques

            for _, row in sub.iterrows():
                r_path = Path(str(row[real_col]))
                g_path = Path(str(row[gen_col]))
                try:
                    if not (r_path.exists() and g_path.exists()):
                        continue

                    # --- LPIPS ---
                    img1 = to_tensor(Image.open(r_path).convert("RGB")).unsqueeze(0)
                    img2 = to_tensor(Image.open(g_path).convert("RGB")).unsqueeze(0)
                    if device == "cuda":
                        img1, img2 = img1.cuda(), img2.cuda()
                    with torch.no_grad():
                        d = lpips_model(img1, img2)
                    lpips_scores.append(float(d.item()))

                    # --- FID: convertit/copie en PNG dans des dossiers parallèles ---
                    uid += 1
                    r_dst = real_tmp / f"{uid:06d}_real.png"
                    g_dst = gen_tmp  / f"{uid:06d}_gen.png"
                    Image.open(r_path).convert("RGB").save(r_dst)
                    Image.open(g_path).convert("RGB").save(g_dst)

                    used_pairs += 1

                except Exception as e:
                    print(f"⚠️ paire ignorée ({r_path.name if 'r_path' in locals() else '?'}) : {e}")

            # Comptage effectif et tolérance petits volumes
            reals_png = [p for p in real_tmp.iterdir() if p.suffix.lower() == ".png"]
            gens_png  = [p for p in gen_tmp.iterdir()  if p.suffix.lower() == ".png"]
            k = min(len(reals_png), len(gens_png))

            if k < 2:
                fid = kid_mean = kid_std = float("nan")
                precision = recall = float("nan") if compute_prc else None
                # FID_UNI
                fid_uni = float("nan")
                # LPIPS
                lp = float(np.mean(lpips_scores)) if len(lpips_scores) else float("nan")

                result_row = {
                    "class": c, "n_pairs": used_pairs,
                    "FID": fid, "KID_mean": kid_mean, "KID_std": kid_std,
                    "LPIPS": lp, "FID_UNI": fid_uni
                }
                if compute_prc:
                    result_row["Precision"] = precision
                    result_row["Recall"] = recall
                results.append(result_row)
                continue

            # KID adapté aux petits volumes
            kid_ss = min(k, 50)

            # --- FID/KID/PRC sur les sous-dossiers temporaires alignés ---
            # PRC activé uniquement si demandé (a du sens en mode pairé avec paires alignées)
            fid_metrics = calculate_metrics(
                input1=str(real_tmp),
                input2=str(gen_tmp),
                cuda=(device == "cuda"),
                isc=False,
                fid=True,
                kid=True,
                prc=compute_prc,  # PRC nécessite des paires alignées (mode pairé uniquement)
                kid_subset_size=kid_ss,     # fix petits volumes
                verbose=False,
            )
            fid = float(fid_metrics.get("frechet_inception_distance", float("nan")))
            kid_mean = fid_metrics.get("kernel_inception_distance_mean", float("nan"))
            kid_std  = fid_metrics.get("kernel_inception_distance_std", float("nan"))
            precision = fid_metrics.get("precision", float("nan")) if compute_prc else None
            recall    = fid_metrics.get("recall", float("nan")) if compute_prc else None

            # --- FID_UNI (optionnel)
            def _fid_uni_from_dirs(rdir: Path, gdir: Path) -> float | None:
                if uni_embed is None:
                    return None
                reals = sorted([p for p in rdir.iterdir() if p.suffix.lower()==".png"])
                gens  = sorted([p for p in gdir.iterdir() if p.suffix.lower()==".png"])
                kk = min(len(reals), len(gens))
                if kk < 2:
                    return float("nan")
                if fid_uni_max is not None:
                    kk = min(kk, int(fid_uni_max))
                reals, gens = reals[:kk], gens[:kk]
                R = np.stack([uni_embed(str(p)) for p in reals], axis=0).astype(np.float64)
                G = np.stack([uni_embed(str(p)) for p in gens],  axis=0).astype(np.float64)
                return _frechet_from_feats(R, G)

            fid_uni = _fid_uni_from_dirs(real_tmp, gen_tmp)
            lp  = float(np.mean(lpips_scores)) if len(lpips_scores) else float("nan")

            result_row = {
                "class": c, "n_pairs": used_pairs,
                "FID": fid, "KID_mean": kid_mean, "KID_std": kid_std,
                "LPIPS": lp, "FID_UNI": fid_uni
            }
            if compute_prc:
                result_row["Precision"] = precision
                result_row["Recall"] = recall
            results.append(result_row)
        finally:
            # Nettoyage des dossiers temporaires
            shutil.rmtree(real_tmp, ignore_errors=True)
            shutil.rmtree(gen_tmp,  ignore_errors=True)

    df = pd.DataFrame(results)
    # Ne supprime Precision/Recall que si elles n'ont pas été calculées
    if not compute_prc:
        df = df.drop(columns=["Precision","Recall"], errors="ignore")
    out_path = save_dir / f"{name}_paired_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Rapport pairé enregistré : {out_path}")
    return df





""" 
| Fonctionnalité | Description                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| `real_root`     | Chemin du dataset réel (**par défaut :** `/workspace/data/NCT-CRC-HE-100K`)                                 |
| `gen_root`      | Chemin du dataset synthétique (**obligatoire**)                                                             |
| `save_dir`      | Dossier où sauvegarder le CSV (créé s’il n’existe pas si non fourni)                                        |
| `save_name`     | Généré automatiquement : `nom_reel_nom_synth.csv`, suffixé par les classes sélectionnées si partiel         |
| `max_images`    | Nombre maximum d’images utilisées pour le calcul du FID (échantillonnées aléatoirement)                    |
| `seed`          | Graine aléatoire pour assurer la reproductibilité des sous-échantillons et du LPIPS                        |
| `classes`       | Liste ou nom de classe(s) à évaluer (par défaut `None` = toutes les classes)                                |
| Vérification    | Compare les sous-dossiers entre `real_root` et `gen_root` et valide les classes spécifiées                 |
| Mode mono/multi | Permet d’évaluer une ou plusieurs classes ; contrôle strict des noms de dossiers correspondants             |
| `device`        | Détection automatique de `"cuda"` ou `"cpu"` selon disponibilité                                            |
| Warnings        | Tous neutralisés pour un run propre (LPIPS, TorchFidelity, Torchvision)                                     |
"""


""" 🧪 Exemple d’appel depuis un notebook (monoclasse)
from metrics.fid_lpips_eval import FIDLPIPSEvaluator
from pathlib import Path

evaluator = FIDLPIPSEvaluator(
    gen_root="../outputs/preprocessing/normalized_tst",
    classes="TUM"
)
df = evaluator.run()
display(df)

Appel multi-classes
evaluator = FIDLPIPSEvaluator(
    gen_root="../outputs/preprocessing/normalized_tst",
    classes=["TUM", "LYM", "MUC"],
    seed=123
)
df = evaluator.run()
"""