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
    pool = sample_real_images_per_class(classes_selected, n_real)
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
            img = load_image_s3(to_s3_path(real_paths[0]))
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
    df = build_test_set(classes_selected, mix, st.session_state["real_pool"], st.session_state["generated_index"])
    # Convert paths to S3
    df["image_path"] = df["image_path"].apply(to_s3_path)

    # Download temporarily for evaluation
    tmp_paths = []
    for p in df["image_path"]:
        local_tmp = Path("/tmp") / Path(p).name
        fs.get(p, str(local_tmp))
        tmp_paths.append(str(local_tmp))
    df["image_path"] = tmp_paths

    res = evaluate_cnn_on_index(model, df)
    st.json(res)

# =============================================================
# FID / LPIPS
# =============================================================
st.header("FID / LPIPS Evaluation")
if st.button("Compute FID / LPIPS"):
    exp = st.session_state.get("experiment_id", "exp")
    gen_index = st.session_state["generated_index"]

    # FID/LPIPS requires local directory => download first
    local_tmp_root = Path("/tmp/synth_eval")
    if local_tmp_root.exists():
        import shutil
        shutil.rmtree(local_tmp_root)
    local_tmp_root.mkdir(parents=True, exist_ok=True)

    for c, lst in gen_index.items():
        d = local_tmp_root / c
        d.mkdir(parents=True, exist_ok=True)
        for g in lst:
            local_p = d / Path(g.path).name
            fs.get(g.path, str(local_p))

    df = compute_fid_lpips(generator_choice, classes_selected, exp, gen_index)
    st.dataframe(df)
