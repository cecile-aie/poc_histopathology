# ============================================================
# 🧠 Base image : PyTorch 2.4.0 + CUDA 12.4 + cuDNN 9
# ============================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# ⚙️ Empêche les dialogues interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# ============================================================
# 📦 Système : outils et dépendances standards
#  + git-lfs (poids Hugging Face)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-opencv \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# ============================================================
# 🌍 Variables d'environnement du projet
# (ajout HF_HOME pour le cache HF propre)
# ============================================================
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PROJECT_ROOT=/workspace \
    CONFIG_DIR=/workspace/config \
    DATASET_DIR=/data \
    TORCH_HOME=/workspace/.torch \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR ${PROJECT_ROOT}

# ============================================================
# 🧰 Mise à jour pip et outils de build
# ============================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================
# ⚙️ Numpy d'abord (pour le build de Pyradiomics)
# ============================================================
RUN pip install --no-cache-dir numpy==1.26.4 "cython<3.0" "pydicom>=2.2.0"

# ============================================================
# 📚 Librairies scientifiques principales
# ============================================================
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn==1.6.1 \
    scikit-image \
    scipy==1.14.1 \
    tqdm \
    jupyterlab \
    fastparquet \
    plotly \
    SimpleITK \
    PyWavelets \
    umap-learn \
    scikit-posthocs \
    statsmodels \
    nvidia-ml-py3

# ============================================================
# 🔬 Pyradiomics (build C++ → sans isolation)
# ============================================================
RUN pip install --no-cache-dir pyradiomics==3.0.1 --no-build-isolation

# ============================================================
# 🔥 Torch utils + vision
# ============================================================
RUN pip install --no-cache-dir \
    torchvision==0.19.0 \
    torchaudio==2.4.0

# ============================================================
# ✅ timm (vision models utilities)
# ============================================================
RUN pip install --no-cache-dir timm

# TensorBoard
RUN pip install --no-cache-dir tensorboard
EXPOSE 6006

# ============================================================
# 🎨 Histopathology preprocessing (Torch-StainTools)
# ============================================================
RUN pip install --no-cache-dir \
    torch-staintools==1.0.4 \
    opencv-python-headless==4.10.0.84

# ============================================================
# 🔧 Évaluation GAN : FID / LPIPS / IS / etc.
#  + clean-fid (pour FID “propre”)
# ============================================================
RUN pip install --no-cache-dir \
    torchmetrics \
    torch-fidelity \
    lpips \
    clean-fid==0.1.35

# ============================================================
# 🧩 Widgets, Pillow, etc.
# ============================================================
RUN pip install --no-cache-dir \
    ipywidgets \
    pillow

# ============================================================
# 📊 Streamlit (dashboard web)
# ============================================================
RUN pip install --no-cache-dir streamlit
EXPOSE 8501

# ============================================================
# 🧪 PixCell stack (Diffusers + HF)
#  - versions stables et compatibles PyTorch 2.4
#  - xformers laissé en commentaire (prudence avec CUDA 12.4)
# ============================================================
RUN pip install --no-cache-dir \
    diffusers==0.31.0 \
    transformers==4.45.2 \
    huggingface_hub==0.26.1 \
    accelerate==1.0.1 \
    einops==0.8.0 \
    safetensors==0.4.5

# Optionnel (⚠️ seulement si wheel dispo pour ta version CUDA/PyTorch)
# RUN pip install --no-cache-dir xformers==0.0.28.post3

# ============================================================
# 🎬 Video & LoRA add-ons (ajout tardif pour rebuild rapide)
# ============================================================
# ffmpeg système (robuste, évite le téléchargement à l'exécution)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# imageio-ffmpeg (wrapper Python) + (optionnel) imageio
# peft pour LoRA / Adapters HF
RUN pip install --no-cache-dir \
    imageio-ffmpeg \
    imageio \
    peft


# ============================================================
# 🎯 Default command
# ============================================================
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
