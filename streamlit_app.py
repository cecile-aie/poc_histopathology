"""
Dashboard Streamlit - GAN vs Diffusion models in histopathology
Are synthetic images helping classification?
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="GAN vs Diffusion - Histopathology",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ajout des chemins
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "p9dg") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "p9dg"))
if str(PROJECT_ROOT / "metrics") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "metrics"))

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
    FID_REF_IMAGES_PER_CLASS,
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    DATA_ROOT,
)
from p9dg.utils.class_mappings import class_labels, class_colors

# ==========================
# Configuration WCAG
# ==========================
# Couleurs avec contraste suffisant (WCAG AA minimum)
COLORS = {
    "primary": "#1f77b4",      # Bleu (contraste > 4.5:1)
    "secondary": "#ff7f0e",    # Orange
    "success": "#2ca02c",     # Vert
    "danger": "#d62728",      # Rouge
    "text": "#212529",         # Texte sombre
    "bg": "#ffffff",           # Fond blanc
    "border": "#dee2e6",       # Bordure grise
}

# Tailles de police accessibles (minimum 16px pour le texte principal)
FONT_SIZES = {
    "title": "2.5rem",      # ~40px
    "subtitle": "1.5rem",   # ~24px
    "heading": "1.25rem",   # ~20px
    "body": "1rem",         # 16px
    "small": "0.875rem",    # 14px
}

# ==========================
# Initialisation session state
# ==========================
if "real_pool" not in st.session_state:
    st.session_state["real_pool"] = None
if "generated_index" not in st.session_state:
    st.session_state["generated_index"] = {}  # {class_name: [GeneratedImageInfo]}
if "experiment_id" not in st.session_state:
    st.session_state["experiment_id"] = None
if "cnn_results" not in st.session_state:
    st.session_state["cnn_results"] = None
if "cnn_results_prev" not in st.session_state:
    st.session_state["cnn_results_prev"] = None
if "fid_lpips_results" not in st.session_state:
    st.session_state["fid_lpips_results"] = None
if "fid_lpips_results_prev" not in st.session_state:
    st.session_state["fid_lpips_results_prev"] = None

# Contrôle du défilement des images synthétiques (onglet Real vs Synthetic)
if "synth_slideshow_running" not in st.session_state:
    st.session_state["synth_slideshow_running"] = False
if "synth_slideshow_idx" not in st.session_state:
    st.session_state["synth_slideshow_idx"] = 0
if "synth_slideshow_class" not in st.session_state:
    st.session_state["synth_slideshow_class"] = None
if "prev_real_pair_idx" not in st.session_state:
    st.session_state["prev_real_pair_idx"] = None
if "last_gen_time" not in st.session_state:
    # Temps total (en secondes) de la dernière génération d'images
    st.session_state["last_gen_time"] = None
if "prev_selected_class_gallery" not in st.session_state:
    # Suivre la classe précédente pour détecter les changements
    st.session_state["prev_selected_class_gallery"] = None
if "real_pair_idx" not in st.session_state:
    # Index de l'image réelle dans la comparaison
    st.session_state["real_pair_idx"] = 1
if "synth_manual_idx" not in st.session_state:
    # Index manuel de l'image synthétique
    st.session_state["synth_manual_idx"] = 1

# ==========================
# Cache des modèles (lazy loading)
# ==========================
@st.cache_resource
def load_cgan_cached():
    """Charge le modèle cGAN avec cache Streamlit"""
    try:
        return load_cgan_model(MODELS_DIR / "cgan_best_model.pt", DEVICE)
    except Exception as e:
        st.error(f"Erreur chargement cGAN: {e}")
        return None

@st.cache_resource
def load_pixcell_cached():
    """Charge le modèle PixCell avec cache Streamlit"""
    try:
        # load_pixcell_model retourne un tuple (pipe, uni_model, uni_transform)
        return load_pixcell_model(MODELS_DIR / "pixcell256_reference.pt", DEVICE)
    except Exception as e:
        st.error(f"Erreur chargement PixCell: {e}")
        return None

@st.cache_resource
def load_mobilenet_cached():
    """Charge le modèle MobileNetV2 avec cache Streamlit"""
    try:
        return load_mobilenet_cnn(MODELS_DIR / "mobilenetv2_best.pt", DEVICE)
    except Exception as e:
        st.error(f"Erreur chargement MobileNetV2: {e}")
        return None

# ==========================
# Header
# ==========================
st.title("🔬 GAN vs Diffusion models in histopathology")
st.markdown("### Are synthetic images helping classification?")

# ==========================
# Colonne 1: Configuration & Génération
# ==========================
col1, col2, col3 = st.columns([1, 1.5, 1], gap="large")

with col1:
    st.markdown("## Setup & Generation")
    
    # 1. Sélection des classes
    st.markdown("### 1. Choose classes")
    all_classes = sorted(class_labels.keys())
    
    # Checkbox "All classes"
    select_all = st.checkbox("All classes", key="select_all_classes")
    
    if select_all:
        selected_classes = all_classes
    else:
        # Par défaut, on préfère les trois classes NORM, STR, TUM si elles existent
        preferred_default = [c for c in ["NORM", "STR", "TUM"] if c in all_classes]
        if len(preferred_default) == 3:
            default_classes = preferred_default
        else:
            default_classes = all_classes[:3] if len(all_classes) >= 3 else all_classes
        
        selected_classes = st.multiselect(
            "Select classes:",
            options=all_classes,
            default=default_classes,
            key="selected_classes"
        )
    
    st.info(f"**Classes selected:** {len(selected_classes)} / {len(all_classes)}")
    
    # 2. Images réelles
    st.markdown("### 2. Real images")
    n_real_per_class = st.slider(
        "Real images per class",
        min_value=1,
        max_value=20,
        value=5,
        key="n_real_per_class"
    )
    
    # Bouton pour construire le pool
    if st.button("Build real images pool", key="build_pool", use_container_width=True):
        with st.spinner("Building real images pool..."):
            try:
                pool = sample_real_images_per_class(
                    selected_classes=selected_classes,
                    n_per_class=n_real_per_class
                )
                st.session_state["real_pool"] = pool
                total_real = sum(len(paths) for paths in pool.pool.values())
                st.success(f"✅ Pool total: {n_real_per_class} × {len(selected_classes)} = {total_real} real images")
                
                # Nouvelle expérience : vider les images synthétiques et réinitialiser les index
                st.session_state["generated_index"] = {}
                st.session_state["experiment_id"] = None
                st.session_state["real_pair_idx"] = 1
                st.session_state["synth_manual_idx"] = 1
                st.session_state["synth_slideshow_idx"] = 0
                st.session_state["synth_slideshow_running"] = False
                st.session_state["prev_real_pair_idx"] = None
                st.session_state["prev_selected_class_gallery"] = None
                st.session_state["synth_slideshow_class"] = None
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    # Afficher info pool
    if st.session_state["real_pool"] is not None:
        pool = st.session_state["real_pool"]
        total = sum(len(paths) for paths in pool.pool.values())
        st.info(f"Current pool: {total} images across {len(pool.pool)} classes")
    
    # 3. Génération synthétique
    st.markdown("### 3. Synthetic generation")
    
    generator_type = st.radio(
        "Generator type:",
        options=["cgan", "pixcell"],
        format_func=lambda x: "cGAN" if x == "cgan" else "Diffusion (PixCell)",
        key="generator_type"
    )
    
    # Info: toujours 100 images par classe
    st.info(f"**Will generate:** {N_TEST_PER_CLASS} synthetic images per class (fixed)")
    
    # Avertissement sur le temps de génération
    if generator_type == "pixcell":
        st.warning(
            "⏱️ **Note:** PixCell uses an iterative generative diffusion process (~20-28 steps per image). "
            "Images generation will take much longer than cGAN but image quality should improve."
        )
    
    # Bouton de génération
    if st.button("Generate images", key="generate", use_container_width=True, type="primary"):
        # Vérifications préalables
        if len(selected_classes) == 0:
            st.error("⚠️ Please select at least one class")
        elif st.session_state["real_pool"] is None:
            st.error("⚠️ Please build the real images pool first")
        else:
            # Générer experiment_id
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["experiment_id"] = experiment_id
            
            # Charger le modèle nécessaire uniquement
            if generator_type == "cgan":
                model = load_cgan_cached()
            else:  # pixcell
                model = load_pixcell_cached()
            
            if model is None:
                st.error(f"⚠️ Model {generator_type} not loaded")
            else:
                import time
                start_time = time.time()
                with st.spinner(f"Generating {N_TEST_PER_CLASS} images per class with {generator_type}..."):
                    generated_index = {}
                    class_to_idx, _ = get_class_mapping()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    timer_placeholder = st.empty()
                    
                    total_classes = len(selected_classes)
                    for idx, class_name in enumerate(selected_classes):
                        status_text.text(f"Generating for {class_name}... ({idx+1}/{total_classes})")
                        
                        # Mise à jour du chronomètre en temps quasi réel
                        elapsed_loop = time.time() - start_time
                        total_secs = int(round(elapsed_loop))
                        if elapsed_loop > 0 and total_secs == 0:
                            total_secs = 1
                        mins_loop = total_secs // 60
                        secs_loop = total_secs % 60
                        timer_placeholder.text(f"Generation time: {mins_loop:02d}:{secs_loop:02d}")
                        
                        try:
                            class_id = class_to_idx[class_name]
                            output_dir = OUTPUTS_DIR / "synth" / generator_type
                            
                            if generator_type == "cgan":
                                generated = generate_with_cgan(
                                    model=model,
                                    class_name=class_name,
                                    class_id=class_id,
                                    n_images=N_TEST_PER_CLASS,
                                    output_dir=output_dir,
                                    experiment_id=experiment_id
                                )
                            else:  # pixcell
                                generated = generate_with_pixcell(
                                    pipe_tuple=model,
                                    class_name=class_name,
                                    class_id=class_id,
                                    n_images=N_TEST_PER_CLASS,
                                    output_dir=output_dir,
                                    experiment_id=experiment_id,
                                    real_pool=st.session_state["real_pool"]
                                )
                            
                            generated_index[class_name] = generated
                            
                        except Exception as e:
                            st.warning(f"⚠️ Error generating for {class_name}: {e}")
                            generated_index[class_name] = []
                        
                        progress_bar.progress((idx + 1) / total_classes)
                    
                    st.session_state["generated_index"] = generated_index
                    
                    # Réinitialiser les index après génération de nouvelles images
                    st.session_state["real_pair_idx"] = 1
                    st.session_state["synth_manual_idx"] = 1
                    st.session_state["synth_slideshow_idx"] = 0
                    st.session_state["synth_slideshow_running"] = False
                    st.session_state["prev_real_pair_idx"] = None
                    st.session_state["prev_selected_class_gallery"] = None
                    st.session_state["synth_slideshow_class"] = None
                
                # Afficher le nombre d'images générées par classe
                total_generated = sum(len(imgs) for imgs in generated_index.values())
                st.success(f"✅ Images generated for {len(selected_classes)} classes. Total: {total_generated} images")
                
                # Affichage du temps total de génération (chronomètre final, persistant)
                elapsed = time.time() - start_time
                st.session_state["last_gen_time"] = elapsed
                total_secs = int(round(elapsed))
                if elapsed > 0 and total_secs == 0:
                    total_secs = 1
                mins = total_secs // 60
                secs = total_secs % 60
                # Réutiliser le placeholder du timer pour afficher le temps final
                timer_placeholder.markdown(f"**Total generation time (last run): {mins:02d}:{secs:02d}**")
                
                # Afficher le détail par classe
                for class_name, imgs in generated_index.items():
                    st.text(f"  • {class_name}: {len(imgs)} images")

# ==========================
# Colonne 2: Galerie
# ==========================
with col2:
    st.markdown("## Gallery (real vs synth)")
    
    # Sélection de classe (galerie) – radio horizontal pour rappeler les "chips" rouges de la colonne 1
    if len(selected_classes) > 0:
        selected_class_gallery = st.radio(
            "Class:",
            options=selected_classes,
            key="selected_class_gallery",
            horizontal=True
        )
        
        # Détecter le changement de classe et réinitialiser les index
        if st.session_state["prev_selected_class_gallery"] != selected_class_gallery:
            st.session_state["prev_selected_class_gallery"] = selected_class_gallery
            # Réinitialiser les index à 1 quand la classe change
            st.session_state["real_pair_idx"] = 1
            st.session_state["synth_manual_idx"] = 1
            st.session_state["synth_slideshow_idx"] = 0
            st.session_state["synth_slideshow_running"] = False
            st.session_state["prev_real_pair_idx"] = None
            st.session_state["synth_slideshow_class"] = selected_class_gallery
        
        # Compteurs
        n_real_gallery = 0
        n_synth_gallery = 0
        
        if st.session_state["real_pool"] is not None:
            pool = st.session_state["real_pool"]
            n_real_gallery = len(pool.pool.get(selected_class_gallery, []))
        
        if selected_class_gallery in st.session_state["generated_index"]:
            n_synth_gallery = len(st.session_state["generated_index"][selected_class_gallery])
        
        st.caption(f"{n_real_gallery} real | {n_synth_gallery} synthetic")
        
        # Onglets
        tab1, tab2 = st.tabs(["Class preview", "Real vs Synth"])
        
        with tab1:
            # 1) Images réelles en premier (référence)
            st.markdown("### Real images (reference)")
            if st.session_state["real_pool"] is not None:
                pool = st.session_state["real_pool"]
                real_images = pool.pool.get(selected_class_gallery, [])
                
                if len(real_images) > 0:
                    # Afficher un échantillon horizontal
                    n_show = min(5, len(real_images))
                    cols = st.columns(n_show)
                    for idx, img_path in enumerate(real_images[:n_show]):
                        try:
                            img = Image.open(img_path)
                            cols[idx].image(img, caption=f"Real {idx+1}")
                        except Exception as e:
                            cols[idx].error(f"Error: {e}")
                else:
                    st.info("No real images in pool for this class")
            
            # 2) Puis images synthétiques en dessous
            st.markdown("---")
            st.markdown(f"### Synthetic images: {selected_class_gallery}")
            
            if selected_class_gallery in st.session_state["generated_index"]:
                synth_images = st.session_state["generated_index"][selected_class_gallery]
                
                if len(synth_images) > 0:
                    # Grille 2x4 (8 images)
                    n_per_page = 8
                    n_pages = (len(synth_images) + n_per_page - 1) // n_per_page
                    
                    if n_pages > 1:
                        page = st.number_input(
                            "Page:",
                            min_value=1,
                            max_value=n_pages,
                            value=1,
                            key="synth_page"
                        )
                    else:
                        page = 1
                    
                    start_idx = (page - 1) * n_per_page
                    end_idx = min(start_idx + n_per_page, len(synth_images))
                    
                    # Afficher la grille
                    for row in range(2):
                        cols = st.columns(4)
                        for col_idx in range(4):
                            img_idx = start_idx + row * 4 + col_idx
                            if img_idx < end_idx:
                                img_info = synth_images[img_idx]
                                try:
                                    img = Image.open(img_info.path)
                                    cols[col_idx].image(img, caption=f"Sample {img_idx+1}")
                                except Exception as e:
                                    cols[col_idx].error(f"Error loading image: {e}")
                else:
                    st.info("No synthetic images generated yet for this class")
            else:
                st.info("No synthetic images generated yet for this class")
        
        with tab2:
            st.markdown("### Real vs Synthetic comparison")
            generator_type = st.session_state.get("generator_type", "cgan")
            
            # Paires réel/synthétique + défilement des images synthétiques
            if (st.session_state["real_pool"] is not None and 
                selected_class_gallery in st.session_state["generated_index"]):
                
                pool = st.session_state["real_pool"]
                real_images = pool.pool.get(selected_class_gallery, [])
                synth_images = st.session_state["generated_index"][selected_class_gallery]
                
                if len(real_images) > 0 and len(synth_images) > 0:
                    # Réinitialiser le slideshow si la classe change
                    if st.session_state["synth_slideshow_class"] != selected_class_gallery:
                        st.session_state["synth_slideshow_class"] = selected_class_gallery
                        st.session_state["synth_slideshow_idx"] = 0
                        st.session_state["synth_slideshow_running"] = False
                    
                    col_left, col_right = st.columns(2)
                    
                    # Affichage image réelle
                    real_width = None
                    real_idx = 0
                    real_img_path = None
                    with col_left:
                        try:
                            # Choix de l'image réelle de référence (au-dessus de l'image)
                            current_real_idx = st.session_state.get("real_pair_idx", 1)
                            if current_real_idx > len(real_images) or current_real_idx < 1:
                                current_real_idx = 1
                                st.session_state["real_pair_idx"] = 1
                            real_idx_raw = st.number_input(
                                "Real image index:",
                                min_value=1,
                                max_value=len(real_images),
                                value=1,
                                key="real_pair_idx"
                            )
                            real_idx = real_idx_raw - 1

                            # Si l'index réel a changé, revenir au premier index synthétique
                            if (
                                st.session_state["prev_real_pair_idx"] is None
                                or st.session_state["prev_real_pair_idx"] != real_idx
                            ):
                                st.session_state["synth_slideshow_idx"] = 0
                                st.session_state["synth_slideshow_running"] = False
                                st.session_state["synth_manual_idx"] = 1
                            st.session_state["prev_real_pair_idx"] = real_idx

                            real_img_path = real_images[real_idx]
                            real_img = Image.open(real_img_path)
                            real_width, _ = real_img.size
                            st.image(real_img, caption="Real", width=real_width)
                        except Exception as e:
                            st.error(f"Error loading real image: {e}")
                            real_img_path = None
                    
                    # Construction du sous-ensemble synthétique
                    real_img_path = real_img_path or (real_images[real_idx] if real_images else None)
                    if real_img_path is not None:
                        if generator_type == "pixcell":
                            filtered_synth_images = [
                                img for img in synth_images
                                if getattr(img, "generator_type", None) == "pixcell"
                                and getattr(img, "ref_path", None) == real_img_path
                            ]
                            if not filtered_synth_images:
                                filtered_synth_images = [
                                    img for img in synth_images
                                    if getattr(img, "generator_type", None) == "pixcell"
                                ]
                        else:
                            filtered_synth_images = [
                                img for img in synth_images
                                if getattr(img, "generator_type", None) == "cgan"
                            ] or synth_images
                    else:
                        filtered_synth_images = []

                    if not filtered_synth_images:
                        col_right.info("No synthetic images available for this selection.")
                    else:
                        n_synth = len(filtered_synth_images)
                        st.session_state["synth_slideshow_idx"] %= n_synth
                        synth_idx = st.session_state["synth_slideshow_idx"]
                        synth_info = None
                        
                        with col_right:
                            try:
                                if st.session_state["synth_slideshow_running"]:
                                    st.session_state["synth_manual_idx"] = synth_idx + 1
                                default_synth_idx = int(st.session_state.get("synth_manual_idx", synth_idx + 1))
                                default_synth_idx = max(1, min(default_synth_idx, n_synth))
                                
                                manual_idx_raw = st.number_input(
                                    "Synthetic index:",
                                    min_value=1,
                                    max_value=n_synth,
                                    value=default_synth_idx,
                                    key="synth_manual_idx"
                                )
                                manual_idx = manual_idx_raw - 1

                                if not st.session_state["synth_slideshow_running"]:
                                    st.session_state["synth_slideshow_idx"] = manual_idx
                                    synth_idx = manual_idx
                                else:
                                    synth_idx = st.session_state["synth_slideshow_idx"]
                                
                                synth_info = filtered_synth_images[synth_idx]
                                synth_img = Image.open(synth_info.path)
                                st.image(
                                    synth_img,
                                    caption=f"Synthetic #{synth_idx+1}",
                                    width=real_width
                                )
                                if generator_type == "pixcell" and getattr(synth_info, "ref_path", None):
                                    st.caption(f"Source real: {Path(synth_info.ref_path).name}")
                                
                                col_btn_left, col_btn_center, col_btn_right = st.columns([0.5, 4, 0.5])
                                with col_btn_center:
                                    if st.button(
                                        "▶  Start / Stop ",
                                        key="toggle_synth_slideshow",
                                        use_container_width=True
                                    ):
                                        st.session_state["synth_slideshow_running"] = not st.session_state["synth_slideshow_running"]

                            except Exception as e:
                                st.error(f"Error loading synthetic image: {e}")
                        
                        if synth_info is not None:
                            st.caption(
                                f"Real index: {real_idx+1} / {len(real_images)} — "
                                f"Synth index: {synth_idx+1} / {n_synth} — "
                                f"class: {selected_class_gallery} — "
                                f"generator: {getattr(synth_info, 'generator_type', 'N/A')}"
                            )

                        if st.session_state["synth_slideshow_running"]:
                            import time
                            time.sleep(2.0)
                            next_idx = (st.session_state["synth_slideshow_idx"] + 1) % n_synth
                            st.session_state["synth_slideshow_idx"] = next_idx
                            st.rerun()
                else:
                    st.info("Need both real and synthetic images for comparison")
            else:
                st.info("Please generate images and build the real pool first")

            # ------------------------------
            # Image quality metrics (FID / LPIPS)
            # ------------------------------
            st.markdown("---")
            st.markdown("#### Image quality metrics (FID / LPIPS)")
            
            if st.button("Compute FID / LPIPS", key="compute_fid", use_container_width=True):
                if len(selected_classes) == 0:
                    st.error("⚠️ Please select classes (left column)")
                elif len(st.session_state["generated_index"]) == 0:
                    st.error("⚠️ Please generate synthetic images first")
                elif st.session_state["experiment_id"] is None:
                    st.error("⚠️ No experiment ID found")
                else:
                    with st.spinner("Computing FID/LPIPS (this may take a while)..."):
                        try:
                            generator_type = st.session_state.get("generator_type", "cgan")
                            
                            results_df = compute_fid_lpips(
                                generator_type=generator_type,
                                selected_classes=selected_classes,
                                experiment_id=st.session_state["experiment_id"],
                                generated_index=st.session_state["generated_index"]
                            )
                            
                            # Sauvegarder l'ancien résultat avant d'écraser
                            st.session_state["fid_lpips_results_prev"] = st.session_state.get("fid_lpips_results")
                            st.session_state["fid_lpips_results"] = results_df
                            
                            st.success("✅ FID/LPIPS computation complete")
                        except Exception as e:
                            st.error(f"Erreur calcul FID/LPIPS: {e}")
            
            # Afficher le dernier résultat FID/LPIPS (stable vis-à-vis des sliders/boutons)
            if st.session_state["fid_lpips_results"] is not None:
                results_df = st.session_state["fid_lpips_results"]
                
                if len(results_df) > 0:
                    st.markdown("##### Last FID/LPIPS scores")
                    col_fid, col_lpips = st.columns(2)
                    col_fid.metric("FID global", f"{results_df['FID_global'].iloc[0]:.2f}")
                    col_lpips.metric("LPIPS global", f"{results_df['LPIPS_global'].iloc[0]:.3f}")
                    
                    st.markdown("##### Per class")
                    st.dataframe(results_df[["class", "FID", "LPIPS"]])
            
            # Afficher résultats précédents FID/LPIPS
            if st.session_state["fid_lpips_results_prev"] is not None:
                prev_df = st.session_state["fid_lpips_results_prev"]
                if len(prev_df) > 0:
                    st.markdown("---")
                    st.markdown("##### Previous FID/LPIPS results")
                    st.dataframe(prev_df[["class", "FID", "LPIPS", "FID_global", "LPIPS_global"]])
    else:
        st.info("Please select classes in the left column")

# ==========================
# Colonne 3: Métriques
# ==========================
with col3:
    st.markdown("## Metrics & Insights")
    
    # 4. Test mix & CNN evaluation
    st.markdown("### 4. Test mix & CNN evaluation")
    
    mix_ratio = st.select_slider(
        "Proportion synthetic in test:",
        options=[0, 20, 40, 60, 80, 100],
        value=0,
        key="mix_ratio"
    )
    
    n_synth_test = int(N_TEST_PER_CLASS * mix_ratio / 100)
    n_real_test = N_TEST_PER_CLASS - n_synth_test
    
    if len(selected_classes) > 0:
        st.caption(
            f"For enriched classes ({len(selected_classes)}): {N_TEST_PER_CLASS} test images → "
            f"{n_synth_test} synth + {n_real_test} real. "
            f"Other classes: {N_TEST_PER_CLASS} real images only."
        )
    else:
        st.caption(f"Per class: {N_TEST_PER_CLASS} real images (no enriched class selected).")
    
    if st.button("🔍 Evaluate CNN classification", key="eval_cnn", use_container_width=True, type="primary"):
        if len(selected_classes) == 0:
            st.error("⚠️ Please select classes")
        elif st.session_state["real_pool"] is None:
            st.error("⚠️ Please build the real images pool first")
        elif len(st.session_state["generated_index"]) == 0:
            st.error("⚠️ Please generate synthetic images first")
        else:
            with st.spinner("Evaluating CNN..."):
                try:
                    mobilenet = load_mobilenet_cached()
                    
                    if mobilenet is None:
                        st.error("⚠️ MobileNetV2 model not loaded")
                    else:
                        # Construire le test set
                        test_df = build_test_set(
                            selected_classes=selected_classes,
                            mix_ratio=mix_ratio,
                            real_pool=st.session_state["real_pool"],
                            generated_index=st.session_state["generated_index"]
                        )
                        
                        # Évaluer
                        results = evaluate_cnn_on_index(
                            model=mobilenet,
                            test_df=test_df
                        )
                        # Mémoriser les classes enrichies pour l'affichage (couleur différente)
                        results["enriched_classes"] = list(selected_classes)
                        
                        # Sauvegarder ancien résultat avant d'écraser
                        st.session_state["cnn_results_prev"] = st.session_state.get("cnn_results")
                        st.session_state["cnn_results"] = results
                        
                        st.success("✅ Evaluation complete")
                
                except Exception as e:
                    st.error(f"Erreur évaluation CNN: {e}")
    
    # Afficher le dernier résultat courant de CNN (indépendant des clics / sliders)
    if st.session_state["cnn_results"] is not None:
        current = st.session_state["cnn_results"]
        st.markdown("#### Last CNN evaluation")
        
        # Calcul du taux de faux négatifs pour TUM
        fn_tum_str = "N/A"
        cm = current.get("confusion_matrix")
        classes_for_ticks = current.get("classes")
        if cm is not None and classes_for_ticks is not None and "TUM" in classes_for_ticks:
            import numpy as np
            tum_idx = classes_for_ticks.index("TUM")
            tum_row = np.array(cm[tum_idx])
            tp = tum_row[tum_idx]
            fn = tum_row.sum() - tp
            denom = tp + fn
            if denom > 0:
                fn_rate_tum = fn / denom
                fn_tum_str = f"{fn_rate_tum * 100:.0f}%"
        
        col_acc, col_f1, col_fn = st.columns(3)
        col_acc.metric("Accuracy", f"{current['accuracy']:.3f}")
        col_f1.metric("F1 macro", f"{current['f1_macro']:.3f}")
        col_fn.metric("FN rate TUM", fn_tum_str)
        
        if current.get("confusion_matrix") is not None:
            st.markdown("##### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                current["confusion_matrix"],
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax
            )
            # Utiliser les codes de classes abrégés (e.g. NORM, STR, TUM)
            classes_for_ticks = current.get("classes", None)
            if classes_for_ticks is not None:
                ax.set_xticklabels(classes_for_ticks, rotation=45, ha="right")
                ax.set_yticklabels(classes_for_ticks, rotation=0)
                
                # Mettre en évidence les classes enrichies (texte dans une autre couleur)
                enriched = set(current.get("enriched_classes", []))
                if enriched:
                    for tick_label, cls in zip(ax.get_xticklabels(), classes_for_ticks):
                        if cls in enriched:
                            tick_label.set_color(COLORS["secondary"])
                    for tick_label, cls in zip(ax.get_yticklabels(), classes_for_ticks):
                        if cls in enriched:
                            tick_label.set_color(COLORS["secondary"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    
    # Afficher résultats précédents (récap lisible plutôt que JSON brut)
    if st.session_state["cnn_results_prev"] is not None:
        results = st.session_state["cnn_results_prev"]
        st.markdown("---")
        st.markdown("#### Previous CNN results")
        
        # KPI cards (taux de faux négatifs TUM sur la précédente évaluation)
        fn_tum_prev_str = "N/A"
        cm_prev = results.get("confusion_matrix")
        classes_for_ticks_prev = results.get("classes")
        if cm_prev is not None and classes_for_ticks_prev is not None and "TUM" in classes_for_ticks_prev:
            import numpy as np
            tum_idx_prev = classes_for_ticks_prev.index("TUM")
            tum_row_prev = np.array(cm_prev[tum_idx_prev])
            tp_prev = tum_row_prev[tum_idx_prev]
            fn_prev = tum_row_prev.sum() - tp_prev
            denom_prev = tp_prev + fn_prev
            if denom_prev > 0:
                fn_rate_tum_prev = fn_prev / denom_prev
                fn_tum_prev_str = f"{fn_rate_tum_prev * 100:.0f}%"
        
        col_acc_prev, col_f1_prev, col_fn_prev = st.columns(3)
        col_acc_prev.metric("Accuracy", f"{results['accuracy']:.3f}")
        col_f1_prev.metric("F1 macro", f"{results['f1_macro']:.3f}")
        col_fn_prev.metric("FN rate TUM", fn_tum_prev_str)
        
        # Matrice de confusion
        if results.get("confusion_matrix") is not None:
            st.markdown("##### Confusion Matrix (previous)")
            fig_prev, ax_prev = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                results["confusion_matrix"],
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax_prev
            )
            classes_for_ticks_prev = results.get("classes", None)
            if classes_for_ticks_prev is not None:
                ax_prev.set_xticklabels(classes_for_ticks_prev, rotation=45, ha="right")
                ax_prev.set_yticklabels(classes_for_ticks_prev, rotation=0)
                
                enriched_prev = set(results.get("enriched_classes", []))
                if enriched_prev:
                    for tick_label, cls in zip(ax_prev.get_xticklabels(), classes_for_ticks_prev):
                        if cls in enriched_prev:
                            tick_label.set_color(COLORS["secondary"])
                    for tick_label, cls in zip(ax_prev.get_yticklabels(), classes_for_ticks_prev):
                        if cls in enriched_prev:
                            tick_label.set_color(COLORS["secondary"])
            ax_prev.set_xlabel("Predicted")
            ax_prev.set_ylabel("Actual")
            st.pyplot(fig_prev)
    
    # (FID/LPIPS section déplacée dans l'onglet 'Real vs Synthetic comparison')

# ==========================
# Footer avec info accessibilité
# ==========================
st.markdown("---")
st.caption(
    "🔍 **Accessibility:** This dashboard follows WCAG guidelines. "
    "Use Tab to navigate, Enter/Space to activate buttons. "
    f"Device: {DEVICE.type.upper()}"
)

