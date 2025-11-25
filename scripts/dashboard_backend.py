"""
Backend pour le dashboard Streamlit - Génération et évaluation d'images synthétiques
"""
from __future__ import annotations
import os
import sys
import random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

# Ajout des chemins nécessaires
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
if str(PROJECT_ROOT / "p9dg") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "p9dg"))
if str(PROJECT_ROOT / "metrics") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "metrics"))

from p9dg.histo_dataset import HistoDataset
from p9dg.utils.class_mappings import class_labels, class_colors, make_idx_mappings
from metrics.cnn_eval import run_eval_split, export_predictions
from metrics.fid_lpips_eval import FIDLPIPSEvaluator

# Configuration globale
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_TEST_PER_CLASS = 100  # Fixé dans le code
FID_REF_IMAGES_PER_CLASS = 200  # Subset immuable pour FID/LPIPS
IMAGE_SIZE = 256

# Chemins
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data")).resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", PROJECT_ROOT / "configs")).resolve()


@dataclass
class GeneratedImageInfo:
    """Information sur une image générée"""
    path: str
    class_name: str
    class_id: int
    generator_type: str
    experiment_id: str
    index: int
    ref_path: Optional[str] = None  # chemin de l'image réelle de référence (pour PixCell)


@dataclass
class RealImagePool:
    """Pool d'images réelles par classe"""
    pool: Dict[str, List[str]]  # {class_name: [list_of_paths]}
    excluded_from_test: Dict[str, List[str]]  # Images exclues du test set


# ==========================
# Utilitaires
# ==========================

def set_seed(seed: int = SEED):
    """Fixer la graine pour reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Récupère le mapping des classes depuis class_mappings.py"""
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_labels.keys()))}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


# ==========================
# Pool d'images réelles
# ==========================

def sample_real_images_per_class(
    selected_classes: List[str],
    n_per_class: int,
    data_root: Path = DATA_ROOT,
    seed: int = SEED
) -> RealImagePool:
    """
    Échantillonne des images réelles depuis CRC-VAL-HE-7K pour chaque classe.
    
    Args:
        selected_classes: Liste des noms de classes (ex: ["TUM", "NORM"])
        n_per_class: Nombre d'images par classe (1-20)
        data_root: Racine des données
        seed: Graine pour reproductibilité
        
    Returns:
        RealImagePool avec pool et excluded_from_test
    """
    set_seed(seed)
    data_dir = data_root / "CRC-VAL-HE-7K"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {data_dir}")
    
    pool = {}
    excluded = {}
    
    for class_name in selected_classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️ Classe {class_name} introuvable dans {data_dir}")
            pool[class_name] = []
            excluded[class_name] = []
            continue
        
        # Lister toutes les images
        all_images = []
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                all_images.append(str(p))
        
        if len(all_images) == 0:
            print(f"⚠️ Aucune image trouvée pour {class_name}")
            pool[class_name] = []
            excluded[class_name] = []
            continue
        
        # Échantillonner
        n_sample = min(n_per_class, len(all_images))
        sampled = random.sample(all_images, n_sample)
        
        pool[class_name] = sampled
        excluded[class_name] = sampled  # Ces images sont exclues du test
    
    return RealImagePool(pool=pool, excluded_from_test=excluded)


# ==========================
# Chargement des modèles
# ==========================

@torch.no_grad()
def load_cgan_model(model_path: Path, device: torch.device = DEVICE) -> nn.Module:
    """
    Charge le modèle cGAN depuis models/cgan_best_model.pt
    
    Note: L'architecture doit correspondre à celle du notebook 06b_cGAN_IA.ipynb
    """
    import torch.nn.functional as F
    import torch.nn.utils as nn_utils
    
    # Architecture Generator depuis le notebook
    class MappingNetwork(nn.Module):
        def __init__(self, z_dim=512, w_dim=512, n_layers=8):
            super().__init__()
            layers = []
            dim = z_dim
            for _ in range(n_layers):
                layers += [nn.Linear(dim, w_dim), nn.LeakyReLU(0.2, inplace=True)]
                dim = w_dim
            self.mapping = nn.Sequential(*layers)
        def forward(self, z):
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            return self.mapping(z)

    class ModulatedConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel, style_dim, demod=True, up=False):
            super().__init__()
            self.up = up
            self.demod = demod
            self.eps = 1e-8
            self.pad = kernel // 2
            self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
            self.style = nn.Linear(style_dim, in_ch)
        def forward(self, x, s):
            if self.up:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            b, c, h, w = x.shape
            w1 = self.style(s).view(b, 1, c, 1, 1)
            w2 = self.weight * (w1 + 1)
            if self.demod:
                d = torch.rsqrt((w2 ** 2).sum([2, 3, 4]) + self.eps)
                w2 = w2 * d.view(b, -1, 1, 1, 1)
            x = x.view(1, -1, h, w)
            w2 = w2.view(b * w2.size(1), w2.size(2), w2.size(3), w2.size(4))
            out = F.conv2d(x, w2, padding=self.pad, groups=b)
            return out.view(b, -1, out.shape[-2], out.shape[-1])

    class Generator(nn.Module):
        def __init__(self, z_dim=512, w_dim=512, img_res=256, fmap_base=256, num_classes=9):
            super().__init__()
            self.z_dim = z_dim
            self.w_dim = w_dim
            self.num_blocks = 0
            self.embed = nn.Embedding(num_classes, z_dim)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.mapping = MappingNetwork(z_dim, w_dim)
            self.const = nn.Parameter(torch.randn(1, fmap_base, 4, 4))
            self.blocks = nn.ModuleList()
            in_ch = fmap_base
            res = 4
            while res < img_res:
                out_ch = max(fmap_base // max(res // 8, 1), 64)
                noise_weight = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
                self.register_parameter(f'noise_{self.num_blocks}', noise_weight)
                self.blocks.append(nn.ModuleList([
                    ModulatedConv2d(in_ch, out_ch, 3, w_dim, up=True), nn.LeakyReLU(0.2, inplace=True),
                    ModulatedConv2d(out_ch, out_ch, 3, w_dim, up=False), nn.LeakyReLU(0.2, inplace=True),
                ]))
                in_ch = out_ch
                res *= 2
                self.num_blocks += 1
            self.to_rgb = nn.Conv2d(in_ch, 3, 1)
            nn.init.xavier_uniform_(self.to_rgb.weight, gain=0.8)
            if self.to_rgb.bias is not None:
                nn.init.constant_(self.to_rgb.bias, 0.0)
        
        def forward(self, z, y):
            zc = z + self.embed(y)
            w = self.mapping(zc)
            x = self.const.repeat(z.size(0), 1, 1, 1)
            for i, (m1, a1, m2, a2) in enumerate(self.blocks):
                x = m1(x, w)
                if self.training:
                    noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
                    noise_weight = getattr(self, f'noise_{i}')
                    if noise_weight.shape[1] != x.shape[1]:
                        noise_weight_adj = noise_weight[:, :1, :, :]
                    else:
                        noise_weight_adj = noise_weight
                    x = x + noise * noise_weight_adj
                x = a1(x)
                x = m2(x, w)
                if self.training:
                    noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
                    noise_weight = getattr(self, f'noise_{i}')
                    if noise_weight.shape[1] != x.shape[1]:
                        noise_weight_adj = noise_weight[:, :1, :, :]
                    else:
                        noise_weight_adj = noise_weight
                    x = x + noise * noise_weight_adj
                x = a2(x)
            output = torch.tanh(self.to_rgb(x))
            output = output.clamp(-0.9, 0.9)
            return output
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extraire les paramètres
    if "G_ema" in checkpoint:
        state_dict = checkpoint["G_ema"]
        num_classes = checkpoint.get("num_classes", 9)
        # Extraire fmap_base depuis const
        const_key = [k for k in state_dict.keys() if "const" in k][0]
        fmap_base = state_dict[const_key].shape[1]
    elif "G" in checkpoint:
        state_dict = checkpoint["G"]
        num_classes = checkpoint.get("num_classes", 9)
        const_key = [k for k in state_dict.keys() if "const" in k][0]
        fmap_base = state_dict[const_key].shape[1]
    else:
        raise ValueError(f"Format de checkpoint non reconnu: {model_path}")
    
    # Créer le modèle
    Z_DIM = 512
    model = Generator(
        z_dim=Z_DIM, w_dim=Z_DIM, img_res=IMAGE_SIZE,
        fmap_base=fmap_base, num_classes=num_classes
    ).to(device)
    
    # Charger les poids
    model.load_state_dict(state_dict)
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Mode train pour injection de bruit (comme dans le notebook)
    model.train()
    
    return model


@torch.no_grad()
def load_pixcell_model(model_path: Path, device: torch.device = DEVICE):
    """
    Charge le modèle PixCell depuis models/pixcell256_reference.pt
    
    Le fichier .pt contient un dictionnaire avec les métadonnées et chemins locaux
    pour recharger PixCell et UNI2-h sans dépendre de Hugging Face à chaque fois.
    
    Note: Utilise DiffusionPipeline avec UNI-2h embeddings
    Retourne un tuple (pipe, uni_model, uni_transform) pour la génération
    """
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    import types
    
    # Workaround pour transformers.modeling_layers manquant
    # Dans transformers 4.45.2, modeling_layers n'existe pas.
    # diffusers importe peft (même sans LoRA), et peft essaie d'importer
    # transformers.modeling_layers.GradientCheckpointingLayer lors de son initialisation.
    # On crée un module factice avant l'import de diffusers pour satisfaire peft.
    import sys
    import types
    import transformers
    
    if 'transformers.modeling_layers' not in sys.modules:
        # Classe factice (on n'utilise pas LoRA, donc pas besoin d'implémentation réelle)
        class GradientCheckpointingLayer:
            """Classe factice pour compatibilité avec peft"""
            pass
        
        # Créer et enregistrer le module factice
        modeling_layers_module = types.ModuleType('transformers.modeling_layers')
        modeling_layers_module.GradientCheckpointingLayer = GradientCheckpointingLayer
        sys.modules['transformers.modeling_layers'] = modeling_layers_module
        transformers.modeling_layers = modeling_layers_module
    
    # Import diffusers (peft sera satisfait par le module factice)
    from diffusers import DiffusionPipeline
    
    # Filtre des warnings liés aux attributs de config non attendus
    # (ces warnings sont inoffensifs - les attributs sont simplement ignorés)
    import warnings
    warnings.filterwarnings("ignore", message=".*config attributes.*were passed.*but are not expected.*")
    warnings.filterwarnings("ignore", message=".*Keyword arguments.*are not expected.*and will be ignored.*")
    
    # Charger le modèle de référence
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle de référence introuvable: {model_path}")
    
    print(f"📂 Chargement du modèle de référence depuis {model_path}...")
    ref = torch.load(model_path, map_location="cpu", weights_only=False)
    
    if not isinstance(ref, dict) or "pixcell" not in ref:
        raise ValueError(
            f"Format de modèle invalide. Attendu: dict avec clés 'pixcell', 'uni2h', etc. "
            f"Reçu: {type(ref)}"
        )
    
    # 1) Recharger UNI2-h
    print("🔄 Rechargement de UNI2-h...")
    uni2h_config = ref["uni2h"]
    device_str = uni2h_config.get("device", "cuda" if device.type == "cuda" else "cpu")
    
    # Vérifier HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN non trouvé dans les variables d'environnement. "
            "Le modèle UNI2-h est gated et nécessite un token Hugging Face."
        )
    
    timm_kwargs = uni2h_config["timm_kwargs"].copy()
    
    # Convertir les strings en types réels
    if timm_kwargs.get("mlp_layer") == "timm.layers.SwiGLUPacked":
        timm_kwargs["mlp_layer"] = timm.layers.SwiGLUPacked
    if timm_kwargs.get("act_layer") == "torch.nn.SiLU":
        timm_kwargs["act_layer"] = torch.nn.SiLU
    
    # Charger UNI2-h (timm utilise automatiquement HF_TOKEN depuis os.environ)
    uni_model = timm.create_model(
        uni2h_config["repo_id"],
        pretrained=True,
        **timm_kwargs
    ).eval().to(device)
    
    cfg = resolve_data_config(uni_model.pretrained_cfg, model=uni_model)
    uni_transform = create_transform(**cfg)
    print(f"✅ UNI2-h rechargé: {next(uni_model.parameters()).dtype} | device: {device}")
    
    # 2) Recharger PixCell Pipeline depuis les chemins locaux
    print("🔄 Rechargement de PixCell Pipeline...")
    pixcell_config = ref["pixcell"]
    pixcell_dir = Path(pixcell_config["pixcell_dir"])
    custom_pipe_dir = Path(pixcell_config["custom_pipeline_dir"])
    
    # Résoudre les chemins relatifs si nécessaire
    if not pixcell_dir.is_absolute():
        pixcell_dir = PROJECT_ROOT / pixcell_dir
    if not custom_pipe_dir.is_absolute():
        custom_pipe_dir = PROJECT_ROOT / custom_pipe_dir
    
    if not pixcell_dir.exists():
        raise FileNotFoundError(f"PixCell checkpoint introuvable: {pixcell_dir}")
    if not custom_pipe_dir.exists():
        raise FileNotFoundError(f"Custom pipeline introuvable: {custom_pipe_dir}")
    
    load_config = pixcell_config["load_config"]
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[load_config["dtype"]]
    
    # Charger depuis les chemins locaux (évite le téléchargement depuis HF)
    try:
        pipe = DiffusionPipeline.from_pretrained(
            str(pixcell_dir),
            custom_pipeline=str(custom_pipe_dir),
            trust_remote_code=load_config.get("trust_remote_code", True),
            torch_dtype=torch_dtype,
        ).to(device)
    except Exception as e:
        import traceback
        error_msg = f"Erreur chargement PixCell depuis {pixcell_dir}: {e}\n"
        error_msg += "Traceback complet:\n"
        error_msg += "".join(traceback.format_exc())
        raise RuntimeError(error_msg) from e
    
    # Appliquer les optimisations VRAM
    opt = ref.get("optimizations", {})
    if opt.get("attention_slicing", True):
        pipe.enable_attention_slicing()
    if opt.get("vae_slicing", True):
        pipe.vae.enable_slicing()
    if opt.get("vae_tiling", False):
        pipe.vae.enable_tiling()
    
    print(f"✅ PixCell Pipeline rechargée: {type(pipe).__name__}")
    
    # 3) Appliquer le monkey-patch
    print("🔄 Application du monkey-patch...")
    try:
        adaln = pipe.transformer.adaln_single
        emb_mod = adaln.emb
        orig_forward = emb_mod.forward
        
        # Exécuter le code du patch depuis les métadonnées
        patch_code = ref["monkey_patch"]["code"]
        # Créer un contexte avec orig_forward disponible
        exec_globals = {"torch": torch, "orig_forward": orig_forward, "getattr": getattr}
        exec(patch_code, exec_globals)
        _patched_forward = exec_globals["_patched_forward"]
        
        # Appliquer le patch
        emb_mod.forward = types.MethodType(_patched_forward, emb_mod)
        print(f"✅ Monkey-patch appliqué: {ref['monkey_patch']['target']}")
    except Exception as e:
        print(f"⚠️ Monkey-patch PixArt échoué (peut fonctionner quand même): {e}")
        import traceback
        traceback.print_exc()
    
    return pipe, uni_model, uni_transform


@torch.no_grad()
def load_mobilenet_cnn(model_path: Path, device: torch.device = DEVICE) -> nn.Module:
    """
    Charge le modèle MobileNetV2 depuis models/mobilenetv2_best.pt
    
    Note: Architecture doit correspondre à 02_baseline_cnn.ipynb
    """
    from torchvision import models
    
    # Reconstruire l'architecture (9 classes)
    num_classes = len(class_labels)
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    
    # Ajuster la dernière couche (comme dans le notebook)
    in_features = model.classifier[1].in_features if hasattr(model.classifier[1], "in_features") else model.last_channel
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Charger les poids
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Nettoyer les clés (enlever 'module.' si présent)
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '').replace('model.', '')
        clean_state_dict[clean_key] = v
    
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


# ==========================
# Génération d'images
# ==========================

def generate_with_cgan(
    model: nn.Module,
    class_name: str,
    class_id: int,
    n_images: int,
    output_dir: Path,
    experiment_id: str,
    device: torch.device = DEVICE,
    seed: int = SEED
) -> List[GeneratedImageInfo]:
    """
    Génère des images avec le cGAN conditionnel.
    
    Args:
        model: Modèle cGAN chargé (G_ema)
        class_name: Nom de la classe (ex: "TUM")
        class_id: ID de la classe
        n_images: Nombre d'images à générer
        output_dir: Dossier de sortie
        experiment_id: ID de l'expérience
        device: Device PyTorch
        seed: Graine pour reproductibilité
        
    Returns:
        Liste de GeneratedImageInfo
    """
    set_seed(seed)
    import torchvision.utils as vutils
    
    generated = []
    class_dir = output_dir / experiment_id / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    Z_DIM = 512
    BATCH_SIZE = 32  # Génération par batch pour efficacité
    
    model.train()  # Mode train pour injection de bruit (comme dans le notebook)
    
    with torch.no_grad():
        for i in range(0, n_images, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, n_images - i)
            
            # Générer le batch
            z = torch.randn(batch_size, Z_DIM, device=device)
            y = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
            
            fake_images = model(z, y).clamp(-1, 1)
            
            # Normaliser à [0, 1] pour sauvegarde
            fake_normalized = (fake_images + 1) * 0.5
            
            # Sauvegarder chaque image
            for j in range(batch_size):
                img_idx = i + j
                img_path = class_dir / f"sample_{img_idx:05d}.png"
                
                # Sauvegarder avec vutils.save_image (comme dans le notebook)
                vutils.save_image(
                    fake_normalized[j:j+1],
                    str(img_path),
                    normalize=False
                )
                
                generated.append(GeneratedImageInfo(
                    path=str(img_path),
                    class_name=class_name,
                    class_id=class_id,
                    generator_type="cgan",
                    experiment_id=experiment_id,
                    index=img_idx
                ))
    
    return generated


def generate_with_pixcell(
    pipe_tuple,
    class_name: str,
    class_id: int,
    n_images: int,
    output_dir: Path,
    experiment_id: str,
    real_pool: RealImagePool,
    device: torch.device = DEVICE,
    seed: int = SEED
) -> List[GeneratedImageInfo]:
    """
    Génère des images avec PixCell (diffusion conditionnelle).
    
    Args:
        pipe_tuple: Tuple (pipe, uni_model, uni_transform) depuis load_pixcell_model
        class_name: Nom de la classe
        class_id: ID de la classe
        n_images: Nombre d'images à générer
        output_dir: Dossier de sortie
        experiment_id: ID de l'expérience
        real_pool: Pool d'images réelles pour conditionnement
        device: Device PyTorch
        seed: Graine pour reproductibilité
        
    Returns:
        Liste de GeneratedImageInfo
    """
    set_seed(seed)
    
    pipe, uni_model, uni_transform = pipe_tuple
    generated = []
    class_dir = output_dir / experiment_id / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Récupérer les images de référence pour cette classe
    ref_images = real_pool.pool.get(class_name, [])
    if len(ref_images) == 0:
        print(f"⚠️ Aucune image de référence pour {class_name}")
        return generated
    
    # Paramètres de génération (fixes pour POC)
    GUIDANCE_SCALE = 2.0
    STEPS = 20
    
    @torch.inference_mode()
    def uni_embed_from_image(pil_img, device=device):
        """Encode une PIL.Image en embedding UNI-2h"""
        x = uni_transform(pil_img).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            e = uni_model(x)
            if e.ndim == 4:
                e = e.mean((-2, -1))
        return e.squeeze(0).float()
    
    @torch.inference_mode()
    def run_pixcell256(emb, uncond, guidance_scale=2.0, steps=20, n=1, seed=42):
        """Génère une image avec PixCell"""
        set_seed(seed)
        g = torch.Generator(device=emb.device).manual_seed(seed)
        
        # added_cond_kwargs pour PixArt
        added_cond = {
            "resolution": torch.tensor([256], device=emb.device, dtype=torch.long),
            "aspect_ratio": torch.tensor([1000], device=emb.device, dtype=torch.long),  # 1.0 * 1000
        }
        
        with torch.autocast(device_type="cuda", enabled=(emb.device.type == "cuda")):
            result = pipe(
                uni_embeds=emb,
                negative_uni_embeds=uncond,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                num_images_per_prompt=n,
                generator=g,
                height=256,
                width=256,
                added_cond_kwargs=added_cond,
            )
        return result.images
    
    # Générer les images
    for i in range(n_images):
        try:
            # Sélectionner une image de référence (cyclique)
            ref_path = ref_images[i % len(ref_images)]
            ref_img = Image.open(ref_path).convert("RGB")
            
            # Encoder en UNI-2h
            emb = uni_embed_from_image(ref_img, device=device)
            emb = emb.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)  # (1, 1, 1536)
            
            # Embedding négatif (unconditional)
            B = emb.shape[0]
            uncond = pipe.get_unconditional_embedding(B).to(device=emb.device, dtype=emb.dtype)
            
            # Générer
            gen_seed = seed + i
            outs = run_pixcell256(emb, uncond, guidance_scale=GUIDANCE_SCALE, steps=STEPS, n=1, seed=gen_seed)
            
            # Sauvegarder
            img_path = class_dir / f"sample_{i:05d}.png"
            outs[0].save(img_path)
            
            generated.append(GeneratedImageInfo(
                path=str(img_path),
                class_name=class_name,
                class_id=class_id,
                generator_type="pixcell",
                experiment_id=experiment_id,
                index=i,
                ref_path=ref_path,  # lien explicite vers l'image réelle conditionnante
            ))
            
        except Exception as e:
            print(f"⚠️ Erreur génération PixCell pour {class_name} image {i}: {e}")
            continue
    
    return generated


# ==========================
# Construction du test set
# ==========================

def build_test_set(
    selected_classes: List[str],
    mix_ratio: int,  # 0-100 (% synthétique)
    real_pool: RealImagePool,
    generated_index: Dict[str, List[GeneratedImageInfo]],
    n_test_per_class: int = N_TEST_PER_CLASS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Construit un DataFrame avec le test set (mélange réel/synthétique).
    
    Args:
        selected_classes: Classes sélectionnées
        mix_ratio: Pourcentage d'images synthétiques (0-100)
        real_pool: Pool d'images réelles
        generated_index: Index des images générées {class_name: [GeneratedImageInfo]}
        n_test_per_class: Nombre d'images de test par classe
        
    Args:
        selected_classes: Classes sélectionnées
        mix_ratio: Pourcentage d'images synthétiques (0-100)
        real_pool: Pool d'images réelles
        generated_index: Index des images générées {class_name: [GeneratedImageInfo]}
        n_test_per_class: Nombre d'images de test par classe
        seed: Graine pour la sélection aléatoire des images réelles

    Returns:
        DataFrame avec colonnes: image_path, class_name, class_id, source_type, generator_type
    """
    # Graine fixée pour que, à pool réel et images synthétiques identiques,
    # la construction du test set soit reproductible (pour un mix_ratio donné).
    set_seed(seed)
    rows = []
    
    # On construit le test set sur TOUTES les classes, pas seulement celles enrichies.
    # - Classes enrichies (selected_classes) : mix réel/synth défini par le slider.
    # - Autres classes : uniquement des images réelles (n_test_per_class).
    class_to_idx, _ = get_class_mapping()
    all_classes = sorted(class_to_idx.keys())
    
    for class_name in all_classes:
        is_enriched = class_name in selected_classes
        
        if is_enriched:
            n_synth = int(n_test_per_class * mix_ratio / 100)
        else:
            n_synth = 0
        
        n_real = n_test_per_class - n_synth
        
        # Images synthétiques (uniquement pour les classes enrichies)
        if is_enriched and n_synth > 0 and class_name in generated_index:
            synth_images = generated_index[class_name]
            n_available_synth = len(synth_images)
            n_synth = min(n_synth, n_available_synth)
            
            for img_info in synth_images[:n_synth]:
                rows.append({
                    "image_path": img_info.path,
                    "class_name": class_name,
                    "class_id": img_info.class_id,
                    "source_type": "synth",
                    "generator_type": img_info.generator_type
                })
        
        # Images réelles (en excluant celles du pool si présent)
        class_dir = DATA_ROOT / "CRC-VAL-HE-7K" / class_name
        all_real = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            all_real.extend(class_dir.glob(ext))
        
        excluded_list = []
        if real_pool is not None and real_pool.excluded_from_test is not None:
            excluded_list = real_pool.excluded_from_test.get(class_name, [])
        excluded = set(excluded_list)
        
        available_real = [str(p) for p in all_real if str(p) not in excluded]
        
        if len(available_real) > 0:
            n_real = min(n_real, len(available_real))
            sampled_real = random.sample(available_real, n_real)
            
            for img_path in sampled_real:
                rows.append({
                    "image_path": str(img_path),
                    "class_name": class_name,
                    "class_id": class_to_idx[class_name],
                    "source_type": "real",
                    "generator_type": None
                })
    
    return pd.DataFrame(rows)


# ==========================
# Évaluation CNN
# ==========================

def evaluate_cnn_on_index(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device = DEVICE,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Évalue le modèle CNN sur le test set.
    
    Args:
        model: Modèle MobileNetV2 chargé
        test_df: DataFrame du test set
        device: Device PyTorch
        batch_size: Taille des batches
        
    Returns:
        Dict avec accuracy, f1_macro, confusion_matrix, etc.
    """
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from torch.nn.functional import softmax
    
    # Dataset personnalisé depuis le DataFrame
    class DataFrameDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform or transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row["image_path"]
            label = int(row["class_id"])
            
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            
            return img, label, img_path
    
    # Créer le dataset et dataloader
    transform_eval = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DataFrameDataset(test_df, transform=transform_eval)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    # Évaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(imgs)
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculer les métriques
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    
    # Matrice de confusion
    class_to_idx, _ = get_class_mapping()
    classes = sorted(class_to_idx.keys())
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    
    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm,
        "n_classes": len(test_df["class_name"].unique()),
        # Liste ordonnée des codes de classes (ADI, BACK, ..., TUM) pour l'affichage
        "classes": classes
    }


# ==========================
# Métriques FID/LPIPS
# ==========================

def compute_fid_lpips(
    generator_type: str,
    selected_classes: List[str],
    experiment_id: str,
    generated_index: Dict[str, List[GeneratedImageInfo]],
    n_ref_images: int = FID_REF_IMAGES_PER_CLASS,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Calcule FID et LPIPS pour chaque classe (mode unpaired).
    
    Args:
        generator_type: Type de générateur ("cgan" ou "pixcell")
        selected_classes: Classes sélectionnées
        experiment_id: ID de l'expérience
        generated_index: Index des images générées
        n_ref_images: Nombre d'images de référence par classe (200)
        seed: Graine pour reproductibilité
        
    Returns:
        DataFrame avec colonnes: class, FID, LPIPS, FID_global, LPIPS_global
    """
    set_seed(seed)
    
    # Utiliser run_eval_experiment qui gère mieux les dossiers
    from metrics.fid_lpips_eval import run_eval_experiment
    
    # Créer un dossier temporaire pour les images synthétiques organisées par classe
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmp_base:
        tmp_base_path = Path(tmp_base)
        tmp_gen_root = tmp_base_path / "synth"
        tmp_gen_root.mkdir()
        
        # Dossier temporaire pour les résultats (run_eval_experiment nécessite un save_dir valide)
        tmp_save_dir = tmp_base_path / "results"
        tmp_save_dir.mkdir()
        
        # Organiser les images synthétiques par classe
        for class_name in selected_classes:
            if class_name not in generated_index:
                print(f"⚠️ Aucune image générée pour la classe {class_name}")
                continue
            
            class_dir = tmp_gen_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            n_copied = 0
            for img_info in generated_index[class_name]:
                try:
                    # Vérifier que l'image existe
                    if not Path(img_info.path).exists():
                        print(f"⚠️ Image introuvable: {img_info.path}")
                        continue
                    
                    # Copier l'image avec un nom simple
                    dst = class_dir / Path(img_info.path).name
                    shutil.copy(img_info.path, dst)
                    n_copied += 1
                except Exception as e:
                    print(f"⚠️ Erreur copie image {img_info.path}: {e}")
            
            if n_copied == 0:
                print(f"⚠️ Aucune image copiée pour la classe {class_name}")
        
        # Vérifier qu'il y a au moins une classe avec des images
        if not any((tmp_gen_root / cls).exists() and any((tmp_gen_root / cls).iterdir()) 
                   for cls in selected_classes):
            print("⚠️ Aucune image synthétique disponible pour le calcul FID/LPIPS")
            return pd.DataFrame({
                "class": selected_classes,
                "FID": [np.nan] * len(selected_classes),
                "LPIPS": [np.nan] * len(selected_classes),
                "FID_global": [np.nan] * len(selected_classes),
                "LPIPS_global": [np.nan] * len(selected_classes)
            })
        
        # Dossier réel (CRC-VAL-HE-7K)
        real_root = DATA_ROOT / "CRC-VAL-HE-7K"
        
        if not real_root.exists():
            raise FileNotFoundError(f"Dossier réel introuvable: {real_root}")
        
        # Vérifier que les classes existent dans le dossier réel
        real_classes = [d.name for d in real_root.iterdir() 
                       if d.is_dir() and not d.name.startswith(".")]
        missing_classes = [c for c in selected_classes if c not in real_classes]
        if missing_classes:
            print(f"⚠️ Classes absentes du dossier réel: {missing_classes}")
        
        # Calculer FID/LPIPS avec run_eval_experiment
        try:
            df_results = run_eval_experiment(
                name=f"{generator_type}_{experiment_id}",
                real_root=str(real_root),
                gen_root=str(tmp_gen_root),
                classes=selected_classes,
                max_images_per_class=n_ref_images,
                lpips_pairs=50,
                seed=seed,
                save_dir=str(tmp_save_dir),  # Dossier temporaire valide
                drop_back_variant=True,  # Exclure BACK comme dans le notebook
            )
            
            # Calculer les scores globaux (moyenne simple) - répéter pour chaque ligne
            if len(df_results) > 0 and "FID" in df_results.columns and "LPIPS" in df_results.columns:
                # Filtrer les NaN pour le calcul global
                valid_fid = df_results["FID"].dropna()
                valid_lpips = df_results["LPIPS"].dropna()
                
                if len(valid_fid) > 0:
                    fid_global = valid_fid.mean()
                else:
                    fid_global = np.nan
                
                if len(valid_lpips) > 0:
                    lpips_global = valid_lpips.mean()
                else:
                    lpips_global = np.nan
                
                # Répéter les valeurs globales pour chaque ligne (même longueur)
                df_results["FID_global"] = fid_global
                df_results["LPIPS_global"] = lpips_global
            else:
                print("⚠️ DataFrame de résultats vide ou colonnes manquantes")
                df_results["FID_global"] = np.nan
                df_results["LPIPS_global"] = np.nan
            
            return df_results
            
        except Exception as e:
            print(f"⚠️ Erreur calcul FID/LPIPS: {e}")
            import traceback
            traceback.print_exc()
            
            # Retourner un DataFrame vide avec les colonnes attendues
            # Répéter les valeurs globales pour chaque ligne
            return pd.DataFrame({
                "class": selected_classes,
                "FID": [np.nan] * len(selected_classes),
                "LPIPS": [np.nan] * len(selected_classes),
                "FID_global": [np.nan] * len(selected_classes),
                "LPIPS_global": [np.nan] * len(selected_classes)
            })

