# streamlit_app_EC2.py
# Full Streamlit dashboard with S3 integration

import os
import sys
from pathlib import Path
import time
import io
from datetime import datetime
from typing import Dict, List

import streamlit as st
from PIL import Image
import boto3
import s3fs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================
# GLOBAL CONFIGURATION
# =============================================================
BUCKET = "p9-histo-data"
s3 = boto3.client("s3")
fs = s3fs.S3FileSystem(anon=False)

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_PREFIX = "CRC-VAL-HE-7K"

from dashboard_backend import (
    get_class_mapping,
    sample_real_images_per_class,
    load_cgan_model,
    load_pixcell_model,
    load_mobilenet_cnn,
    generate_with_cgan,
    generate_with_pixcell,
    build_test_set,
    evaluate_cnn_on_index,
    compute_fid_lpips,
    GeneratedImageInfo,
    RealImagePool,
    DEVICE,
    N_TEST_PER_CLASS,
    IMAGE_SIZE
)
from p9dg.utils.class_mappings import class_labels

# =============================================================
# UTILITY FUNCTIONS
# =============================================================

def ensure_model(local_path: Path, s3_key: str):
    """
    Download model from S3 if missing.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path
    fs.get(f"{BUCKET}/models/{s3_key}", str(local_path))
    return local_path


def to_s3_path(local_path: str) -> str:
    p = Path(local_path)
    parts = p.parts

    if "CRC-VAL-HE-7K" in parts:
        idx = parts.index("CRC-VAL-HE-7K")
        rel = Path(*parts[idx:])
        return f"{BUCKET}/CRC-VAL-HE-7K/{rel.relative_to('CRC-VAL-HE-7K')}"

    if "outputs" in parts:
        idx = parts.index("outputs")
        rel = Path(*parts[idx:])
        return f"{BUCKET}/outputs/{rel.relative_to('outputs')}"

    return f"{BUCKET}/{p.name}"


def load_image_s3(s3_path: str):
    with fs.open(s3_path, "rb") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def save_image_s3(pil_img: Image.Image, s3_path: str):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    with fs.open(s3_path, "wb") as f:
        f.write(buf.read())


def upload_file(local_path: Path, s3_path: str):
    fs.put(str(local_path), s3_path)


def sample_real_images_per_class_s3(
    selected_classes: List[str],
    n_per_class: int,
    seed: int = 42
) -> RealImagePool:
    """
    Échantillonne des images réelles depuis S3 pour chaque classe.
    Version adaptée pour EC2 qui utilise S3 au lieu de chemins locaux.
    """
    import random
    random.seed(seed)
    
    pool = {}
    excluded = {}
    
    for class_name in selected_classes:
        s3_class_prefix = f"{BUCKET}/CRC-VAL-HE-7K/{class_name}/"
        
        # Lister les images dans S3
        try:
            all_images_s3 = []
            # Utiliser s3fs pour lister les fichiers
            if fs.exists(s3_class_prefix):
                # Lister tous les fichiers dans le dossier S3
                for path in fs.ls(s3_class_prefix, detail=False):
                    if any(path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]):
                        all_images_s3.append(path)
            
            if len(all_images_s3) == 0:
                st.warning(f"⚠️ Aucune image trouvée pour {class_name} dans S3")
                pool[class_name] = []
                excluded[class_name] = []
                continue
            
            # Échantillonner
            n_sample = min(n_per_class, len(all_images_s3))
            sampled = random.sample(all_images_s3, n_sample)
            
            pool[class_name] = sampled
            excluded[class_name] = sampled
            
        except Exception as e:
            st.error(f"Erreur lors de la lecture de S3 pour {class_name}: {e}")
            pool[class_name] = []
            excluded[class_name] = []
    
    return RealImagePool(pool=pool, excluded_from_test=excluded)

# =============================================================
# STREAMLIT PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="GAN vs Diffusion - EC2 Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 GAN vs Diffusion (EC2 Mode)")
st.markdown("### Fully Cloud-Based Dashboard (S3 Storage)")

# =============================================================
# LOAD MODELS FROM S3
# =============================================================
@st.cache_resource
def load_cgan():
    path = ensure_model(MODELS_DIR / "cgan_best_model.pt", "cgan_best_model.pt")
    return load_cgan_model(path, DEVICE)

@st.cache_resource
def load_pix():
    path = ensure_model(MODELS_DIR / "pixcell256_reference.pt", "pixcell256_reference.pt")
    return load_pixcell_model(path, DEVICE)

@st.cache_resource
def load_cnn():
    path = ensure_model(MODELS_DIR / "mobilenetv2_best.pt", "mobilenetv2_best.pt")
    return load_mobilenet_cnn(path, DEVICE)

# =============================================================
# SIDEBAR - CLASS SELECTION
# =============================================================
st.sidebar.header("Configuration")
all_classes = sorted(class_labels.keys())
classes_selected = st.sidebar.multiselect(
    "Classes:", all_classes, default=["NORM", "STR", "TUM"]
)
n_real = st.sidebar.slider("Real images per class", 1, 20, 5)
generator_choice = st.sidebar.radio("Generator:", ["cgan", "pixcell"])

if st.sidebar.button("Build real pool"):
    pool = sample_real_images_per_class_s3(classes_selected, n_real)
    st.session_state["real_pool"] = pool

# =============================================================
# MAIN - GENERATION SECTION
# =============================================================
st.header("Synthetic Generation")
if st.button("Generate Synthetic Images", type="primary"):
    if "real_pool" not in st.session_state:
        st.error("Build real pool first")
    else:
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["experiment_id"] = exp_id
        model = load_cgan() if generator_choice == "cgan" else load_pix()

        result_index = {}
        for c in classes_selected:
            class_to_idx, _ = get_class_mapping()
            cid = class_to_idx[c]

            output_dir = OUTPUTS_DIR / "synth" / generator_choice
            output_dir.mkdir(parents=True, exist_ok=True)

            if generator_choice == "cgan":
                gens = generate_with_cgan(model, c, cid, N_TEST_PER_CLASS, output_dir, exp_id)
            else:
                gens = generate_with_pixcell(model, c, cid, N_TEST_PER_CLASS, output_dir, exp_id, st.session_state["real_pool"])

            # Upload to S3
            for g in gens:
                s3_path = to_s3_path(g.path)
                upload_file(Path(g.path), s3_path)
                g.path = s3_path

            result_index[c] = gens

        st.session_state["generated_index"] = result_index
        st.success("Generation complete and uploaded to S3")

# =============================================================
# GALLERY
# =============================================================
st.header("Gallery: Real vs Synthetic")
if "real_pool" in st.session_state and "generated_index" in st.session_state:
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.subheader("Real sample")
        c = st.selectbox("Class", classes_selected)
        real_paths = st.session_state["real_pool"].pool[c]
        if real_paths:
            # Les chemins sont déjà des chemins S3
            s3_path = real_paths[0] if real_paths[0].startswith(f"{BUCKET}/") else f"{BUCKET}/{real_paths[0]}"
            img = load_image_s3(s3_path)
            st.image(img, caption="Real Example")

    with gcol2:
        st.subheader("Synthetic sample")
        gens = st.session_state["generated_index"][c]
        if gens:
            img = load_image_s3(gens[0].path)
            st.image(img, caption="Synthetic Example")

# =============================================================
# CNN EVALUATION
# =============================================================
st.header("CNN Evaluation")
mix = st.slider("% Synthetic in test set", 0, 100, 0)
if st.button("Run CNN Evaluation"):
    model = load_cnn()
    
    # Note: build_test_set utilise DATA_ROOT pour trouver les images réelles
    # On doit télécharger les images nécessaires depuis S3 vers un dossier temporaire
    import tempfile
    import shutil
    import os
    
    tmp_data_root = Path("/tmp/crc_val_he_7k")
    if tmp_data_root.exists():
        shutil.rmtree(tmp_data_root)
    tmp_data_root.mkdir(parents=True, exist_ok=True)
    
    # Télécharger toutes les images nécessaires depuis S3 pour chaque classe
    # (build_test_set a besoin de toutes les images disponibles pour le sampling)
    st.info("Téléchargement des images depuis S3...")
    progress_bar = st.progress(0)
    
    all_classes = sorted(class_labels.keys())
    for idx, class_name in enumerate(all_classes):
        class_dir = tmp_data_root / "CRC-VAL-HE-7K" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Lister et télécharger les images depuis S3
        s3_class_prefix = f"{BUCKET}/CRC-VAL-HE-7K/{class_name}/"
        try:
            if fs.exists(s3_class_prefix):
                s3_files = [f for f in fs.ls(s3_class_prefix, detail=False) 
                           if any(f.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])]
                # Limiter à un nombre raisonnable pour éviter de tout télécharger
                # On télécharge seulement ce qui est nécessaire pour le test set
                max_needed = N_TEST_PER_CLASS * 2  # Marge de sécurité
                for s3_file in s3_files[:max_needed]:
                    local_path = class_dir / Path(s3_file).name
                    if not local_path.exists():
                        fs.get(s3_file, str(local_path))
        except Exception as e:
            st.warning(f"Erreur téléchargement {class_name}: {e}")
        
        progress_bar.progress((idx + 1) / len(all_classes))
    
    # Créer un pool adapté avec chemins locaux
    real_pool_local = RealImagePool(pool={}, excluded_from_test={})
    for class_name in classes_selected:
        s3_paths = st.session_state["real_pool"].pool.get(class_name, [])
        local_paths = []
        for s3_path in s3_paths:
            local_path = tmp_data_root / "CRC-VAL-HE-7K" / class_name / Path(s3_path).name
            if not local_path.exists():
                fs.get(s3_path, str(local_path))
            local_paths.append(str(local_path))
        
        real_pool_local.pool[class_name] = local_paths
        real_pool_local.excluded_from_test[class_name] = local_paths
    
    # Modifier temporairement DATA_ROOT pour build_test_set
    original_data_root = os.getenv("DATA_ROOT")
    os.environ["DATA_ROOT"] = str(tmp_data_root)
    
    try:
        # Construire le test set
        df = build_test_set(classes_selected, mix, real_pool_local, st.session_state["generated_index"])
        
        # Télécharger toutes les images du test set depuis S3 si nécessaire
        tmp_paths = []
        for p in df["image_path"]:
            if p.startswith(f"{BUCKET}/") or p.startswith("s3://"):
                # C'est un chemin S3, télécharger
                local_tmp = Path("/tmp/cnn_eval") / Path(p).name
                local_tmp.parent.mkdir(parents=True, exist_ok=True)
                fs.get(p, str(local_tmp))
                tmp_paths.append(str(local_tmp))
            else:
                # C'est déjà un chemin local
                tmp_paths.append(p)
        df["image_path"] = tmp_paths

        res = evaluate_cnn_on_index(model, df)
        st.json(res)
    finally:
        # Restaurer DATA_ROOT
        if original_data_root:
            os.environ["DATA_ROOT"] = original_data_root
        elif "DATA_ROOT" in os.environ:
            del os.environ["DATA_ROOT"]

# =============================================================
# FID / LPIPS
# =============================================================
st.header("FID / LPIPS Evaluation")
if st.button("Compute FID / LPIPS"):
    exp = st.session_state.get("experiment_id", "exp")
    gen_index = st.session_state["generated_index"]

    # FID/LPIPS requires local directory => download first
    import shutil
    import os
    
    local_tmp_root = Path("/tmp/synth_eval")
    if local_tmp_root.exists():
        shutil.rmtree(local_tmp_root)
    local_tmp_root.mkdir(parents=True, exist_ok=True)

    # Télécharger les images synthétiques depuis S3
    st.info("Téléchargement des images synthétiques depuis S3...")
    for c, lst in gen_index.items():
        d = local_tmp_root / "synth" / c
        d.mkdir(parents=True, exist_ok=True)
        for g in lst:
            local_p = d / Path(g.path).name
            if not local_p.exists():
                fs.get(g.path, str(local_p))
    
    # Télécharger les images réelles depuis S3 pour FID/LPIPS
    st.info("Téléchargement des images réelles depuis S3...")
    tmp_data_root = Path("/tmp/fid_lpips_real")
    if tmp_data_root.exists():
        shutil.rmtree(tmp_data_root)
    tmp_data_root.mkdir(parents=True, exist_ok=True)
    
    real_root = tmp_data_root / "CRC-VAL-HE-7K"
    real_root.mkdir(parents=True, exist_ok=True)
    
    # Télécharger les images réelles nécessaires pour chaque classe
    FID_REF_IMAGES_PER_CLASS = 200  # Comme dans dashboard_backend
    for class_name in classes_selected:
        class_dir = real_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        s3_class_prefix = f"{BUCKET}/CRC-VAL-HE-7K/{class_name}/"
        try:
            if fs.exists(s3_class_prefix):
                s3_files = [f for f in fs.ls(s3_class_prefix, detail=False) 
                           if any(f.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])]
                # Télécharger jusqu'à FID_REF_IMAGES_PER_CLASS images
                for s3_file in s3_files[:FID_REF_IMAGES_PER_CLASS]:
                    local_path = class_dir / Path(s3_file).name
                    if not local_path.exists():
                        fs.get(s3_file, str(local_path))
        except Exception as e:
            st.warning(f"Erreur téléchargement {class_name}: {e}")
    
    # Modifier temporairement DATA_ROOT pour compute_fid_lpips
    original_data_root = os.getenv("DATA_ROOT")
    os.environ["DATA_ROOT"] = str(tmp_data_root)
    
    # Modifier temporairement les chemins dans gen_index pour pointer vers les fichiers locaux
    gen_index_local = {}
    for c, lst in gen_index.items():
        gen_index_local[c] = []
        for g in lst:
            local_p = local_tmp_root / "synth" / c / Path(g.path).name
            # Créer une copie de GeneratedImageInfo avec le chemin local
            from dataclasses import replace
            gen_info_local = replace(g, path=str(local_p))
            gen_index_local[c].append(gen_info_local)
    
    try:
        df = compute_fid_lpips(generator_choice, classes_selected, exp, gen_index_local)
        st.dataframe(df)
    finally:
        # Restaurer DATA_ROOT
        if original_data_root:
            os.environ["DATA_ROOT"] = original_data_root
        elif "DATA_ROOT" in os.environ:
            del os.environ["DATA_ROOT"]
