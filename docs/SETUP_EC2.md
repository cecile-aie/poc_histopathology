# Guide de configuration pour déploiement EC2

Ce guide explique comment configurer le dashboard Streamlit sur une instance EC2 AWS.

## Prérequis

- Instance EC2 avec Docker et Docker Compose installés
- Accès SSH à l'instance
- Bucket S3 `p9-histo-data` créé avec les modèles et données

## Structure S3 requise

Votre bucket S3 doit contenir :

```
p9-histo-data/
├── models/
│   ├── cgan_best_model.pt
│   ├── pixcell256_reference.pt
│   └── mobilenetv2_best.pt
├── CRC-VAL-HE-7K/
│   ├── ADI/
│   ├── BACK/
│   ├── DEB/
│   ├── LYM/
│   ├── MUC/
│   ├── MUS/
│   ├── NORM/
│   ├── STR/
│   └── TUM/
└── outputs/  (créé automatiquement)
```

## Configuration

### 1. Cloner le repository

```bash
git clone <votre-repo-url>
cd oc_aie_p9
```

### 2. Créer le fichier .env

```bash
cp .env.example .env
nano .env  # ou votre éditeur préféré
```

Remplissez les valeurs :
- `HF_TOKEN` : Votre token Hugging Face (obligatoire pour PixCell)
- `AWS_DEFAULT_REGION` : Votre région AWS (défaut: eu-west-3)

### 3. Configuration AWS

**Option A : Rôle IAM (recommandé)** ✅

1. Créez un rôle IAM avec les permissions S3 :
   - `s3:GetObject`
   - `s3:PutObject`
   - `s3:ListBucket`

2. Attachez ce rôle à votre instance EC2

3. **Vous n'avez PAS besoin** de définir `AWS_ACCESS_KEY_ID` et `AWS_SECRET_ACCESS_KEY` dans `.env`
   - boto3/s3fs récupérera automatiquement les credentials depuis le rôle IAM

**Option B : Credentials explicites** ⚠️

Si vous ne pouvez pas utiliser un rôle IAM, ajoutez dans `.env` :
```bash
AWS_ACCESS_KEY_ID=votre_access_key
AWS_SECRET_ACCESS_KEY=votre_secret_key
```

⚠️ **Moins sécurisé** : les credentials seront dans le fichier .env

### 4. Vérifier les permissions

Testez l'accès S3 depuis l'instance :

```bash
# Installer AWS CLI si nécessaire
sudo apt-get update
sudo apt-get install -y awscli

# Tester l'accès (avec rôle IAM ou credentials)
aws s3 ls s3://p9-histo-data/models/
```

### 5. Lancer le dashboard

```bash
docker-compose -f docker-compose.dashboard.cpu.yml up --build
```

Le dashboard sera accessible sur `http://<ip-ec2>:8501`

## Dépannage

### Erreur "HF_TOKEN non trouvé"
- Vérifiez que `HF_TOKEN` est défini dans `.env`
- Vérifiez que le fichier `.env` est bien chargé par docker-compose

### Erreur d'accès S3
- Vérifiez les permissions IAM de l'instance
- Vérifiez que le bucket existe et est accessible
- Testez avec `aws s3 ls s3://p9-histo-data/`

### Le dashboard ne démarre pas
- Vérifiez les logs : `docker-compose -f docker-compose.dashboard.cpu.yml logs`
- Vérifiez que le port 8501 n'est pas déjà utilisé
- Vérifiez que Docker a assez de mémoire/disque

## Sécurité

- ✅ Le fichier `.env` est dans `.gitignore` (ne sera pas commité)
- ✅ Utilisez un rôle IAM plutôt que des credentials hardcodés
- ✅ Limitez les permissions IAM au strict nécessaire (bucket spécifique)
- ✅ Utilisez un Security Group qui limite l'accès au port 8501

## Notes

- Les modèles sont téléchargés depuis S3 au premier chargement
- Les images générées sont uploadées automatiquement vers S3
- Les images sont mises en cache localement dans `/tmp` pour améliorer les performances

