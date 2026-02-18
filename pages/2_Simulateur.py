import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("Informations Client")

age = st.sidebar.slider("Âge", 18, 95, 35)
solde_bancaire = st.sidebar.number_input("Solde Bancaire", -20000, 200000, 1000)
day = st.sidebar.slider("Jour du mois", 1, 31, 15)
duration = st.sidebar.slider("Durée appel (sec)", 0, 5000, 300)
campaign = st.sidebar.slider("Nb contacts campagne", 1, 20, 1)
previous = st.sidebar.slider("Nb contacts précédents", 0, 20, 0)

metier = st.sidebar.selectbox("Métier",
    ['admin.', 'technician', 'services', 'management', 'retired',
     'blue-collar', 'unemployed', 'entrepreneur', 'housemaid',
     'student', 'self-employed', 'unknown'])

statut_matrimonial = st.sidebar.selectbox("Statut Matrimonial",
    ['married', 'single', 'divorced'])

niveau_etudes = st.sidebar.selectbox("Niveau d'Études",
    ['primary', 'secondary', 'tertiary', 'unknown'])

defaut_credit = st.sidebar.selectbox("Défaut Crédit", ['yes','no'])
pret_immo = st.sidebar.selectbox("Prêt Immobilier", ['yes','no'])
pret_conso = st.sidebar.selectbox("Prêt Conso", ['yes','no'])

mois = st.sidebar.selectbox("Mois",
    ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])

resultat_precedent = st.sidebar.selectbox("Résultat campagne précédente",
    ['success','failure','other','unknown'])

segment_contact = st.sidebar.selectbox("Segment Contact",
    ['cellular','telephone','unknown'])
# --- 5. PREDICTION ---
if st.sidebar.button("🎯 Lancer la prédiction"):

    # Création du DataFrame utilisateur
    input_data = pd.DataFrame([{
        'age': age,
        'solde_bancaire': solde_bancaire,
        'day': day,
        'duration': duration,
        'campaign': campaign,
        'previous': previous,
        'defaut_credit': 1 if defaut_credit == 'yes' else 0,
        'pret_immo': 1 if pret_immo == 'yes' else 0,
        'pret_conso': 1 if pret_conso == 'yes' else 0,
        'metier': metier,
        'statut_matrimonial': statut_matrimonial,
        'niveau_etudes': niveau_etudes,
        'mois': mois,
        'resultat_precedent': resultat_precedent,
        'segment_contact': segment_contact
    }])

    # One-hot encoding des colonnes catégorielles
    categorical_cols = ['metier','statut_matrimonial','niveau_etudes','mois',
                        'resultat_precedent','segment_contact']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)

    # --- Colonnes exactes utilisées par le modèle (47 features) ---
    model_columns = [
    'age', 'solde_bancaire', 'day', 'duration', 'campaign', 'previous',
    'defaut_credit', 'pret_immo', 'pret_conso',
    # Métier (11 colonnes au lieu de 12 - 'admin.' est souvent la base supprimée)
    'metier_blue-collar', 'metier_entrepreneur', 'metier_housemaid', 'metier_management',
    'metier_retired', 'metier_self-employed', 'metier_services', 'metier_student', 
    'metier_technician', 'metier_unemployed', 'metier_unknown',
    # Statut matrimonial (2 colonnes au lieu de 3)
    'statut_matrimonial_married', 'statut_matrimonial_single',
    # Niveau études (3 colonnes au lieu de 4)
    'niveau_etudes_secondary', 'niveau_etudes_tertiary', 'niveau_etudes_unknown',
    # Mois (11 colonnes au lieu de 12)
    'mois_aug', 'mois_dec', 'mois_feb', 'mois_jan', 'mois_jul', 'mois_jun', 
    'mois_mar', 'mois_may', 'mois_nov', 'mois_oct', 'mois_sep',
    # Résultat précédent (3 colonnes au lieu de 4)
    'resultat_precedent_other', 'resultat_precedent_success', 'resultat_precedent_unknown',
    # Segment contact (2 colonnes au lieu de 3)
    'segment_contact_telephone', 'segment_contact_unknown'
]

    # Ajouter les colonnes manquantes
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Supprimer les colonnes en trop
    for col in input_data_encoded.columns:
        if col not in model_columns:
            input_data_encoded.drop(col, axis=1, inplace=True)

    # Réordonner les colonnes
    input_data_encoded = input_data_encoded[model_columns]

    # Prédiction
    proba = model.predict_proba(input_data_encoded)[0][1]
    score = round(proba * 100, 2)

    # --- AFFICHAGE ---
    st.markdown("## Résultat")
    st.success(f"Probabilité de souscription : {score}%")

    # --- JAUGE ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "%"},
        title={'text': "Score IA"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "green"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- FEUX TRICOLORES ---
    st.markdown("## 🚦 Priorité Commerciale")
    if score < 30:
        st.error("🔴 FEU ROUGE - Priorité Basse")
        st.markdown("Ne pas abandonner, allouer peu de ressources.")
    elif score <= 60:
        st.warning("🟠 FEU ORANGE - Priorité Moyenne")
        st.markdown("Client à potentiel, renforcer l’argumentaire.")
    else:
        st.success("🟢 FEU VERT - Priorité Haute")
        st.markdown("Opportunité immédiate. Conclure rapidement.")