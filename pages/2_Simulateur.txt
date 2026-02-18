import streamlit as st
import joblib
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv

# --- 1. CONFIGURATION DE LA PAGE ---
load_dotenv()

st.set_page_config(
    page_title="Simulateur Prédictif",
    page_icon="🔮",
    layout="centered"
)

# --- 2. FONCTION DE CHARGEMENT DU MODÈLE (CACHE) ---
@st.cache_resource(show_spinner="Réveil de l'IA en cours...")
def charger_modele_s3():
    """
    Cette fonction va chercher le fichier .joblib sur S3
    et le charge dans la mémoire de l'application.
    """
    bucket_name = os.getenv('BUCKET_NAME')
    # Attention : Doit être le nom EXACT que tu as utilisé dans le notebook hier
    model_key = "model_bank_marketing_v1.joblib"
    
    try:
        # Connexion AWS
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('ACCESS_KEY'),
            aws_secret_access_key=os.getenv('SECRET_KEY'),
            region_name="eu-west-3"
        )
        
        # Téléchargement en mémoire vive (RAM) sans écrire sur le disque
        response = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_bytes = BytesIO(response['Body'].read())
        
        # Reconstitution du cerveau (Dé-sérialisation)
        model_charge = joblib.load(model_bytes)
        return model_charge

    except Exception as e:
        st.error(f"❌ Erreur critique : Impossible de charger le modèle S3. Détails : {e}")
        return None

# --- 3. INITIALISATION ---
# On appelle la fonction une seule fois au lancement
model = charger_modele_s3()

# Petit test visuel pour toi (tu pourras l'enlever après)
if model:
    st.success("✅ Système IA connecté et prêt à prédire.")