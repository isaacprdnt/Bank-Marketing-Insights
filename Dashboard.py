import streamlit as st
import pandas as pd
import boto3
import os
from io import StringIO
from dotenv import load_dotenv

# --- 1. Chargement des accès (Fichier .env) ---
load_dotenv()

# --- 2. Configuration de l'onglet ---
st.set_page_config(
    page_title="Bank Marketing Analysis",
    page_icon="🏦",
    layout="wide"
)

# --- 3. Création du "moteur" de lecture S3 ---
# Le @st.cache_data sert à ne pas retélécharger le fichier à chaque clic (gain de temps)
@st.cache_data
def charger_data_s3(nom_du_fichier):
    # On se connecte avec tes identifiants du .env
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('ACCESS_KEY'),
        aws_secret_access_key=os.getenv('SECRET_KEY'),
        region_name="eu-west-3"
    )
    
    # On récupère le fichier dans le Bucket
    reponse = s3_client.get_object(
        Bucket=os.getenv('BUCKET_NAME'), 
        Key="bank_marketing_cleaned_v1.csv"
    )
    
    # On transforme le contenu en tableau (DataFrame)
    contenu = reponse['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(contenu), sep=';')
    return df

# --- 4. Affichage sur la page ---
st.title("🏦 Dashboard Pilotage S3")

# On essaie de charger le fichier
try:
    with st.spinner('Connexion à AWS S3 en cours...'):
        # Remplace par le nom exact de ton fichier sur S3
        df = charger_data_s3("bank_marketing_cleaned_v1.csv")
        
    st.success("✅ Données chargées depuis le Cloud !")
    
    # On affiche les 5 premières lignes pour vérifier
    st.write("Aperçu des données :")
    st.dataframe(df.head())

except Exception as e:
    st.error(f"❌ Erreur de connexion : {e}")

