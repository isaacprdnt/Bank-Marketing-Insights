import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
import os
from io import StringIO
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()

st.set_page_config(
    page_title="Bank Marketing Insights",
    page_icon="🏦",
    layout="wide"
)

# --- 2. FONCTIONS BACKEND (S3) ---
@st.cache_data
def charger_data_s3(nom_du_fichier):
    # Connexion S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('ACCESS_KEY'),
        aws_secret_access_key=os.getenv('SECRET_KEY'),
        region_name="eu-west-3"
    )
    reponse = s3_client.get_object(Bucket=os.getenv('BUCKET_NAME'), Key=nom_du_fichier)
    contenu = reponse['Body'].read().decode('utf-8')
    
    # Lecture avec le bon séparateur (point-virgule)
    return pd.read_csv(StringIO(contenu), sep=';')

# --- 3. CHARGEMENT ET CALCULS ---
try:
    with st.spinner('Chargement des données...'):
        df = charger_data_s3("bank_marketing_cleaned_v1.csv")
        
        # Définition de la cible
        COLONNE_CIBLE = 'souscription'
        
        # Calcul du taux de conversion
        conversion_rate = (df[COLONNE_CIBLE].value_counts(normalize=True).get('yes', 0)) * 100

except Exception as e:
    st.error(f"Erreur technique : {e}")
    st.stop()

# --- 4. INTERFACE UTILISATEUR (Standard Streamlit) ---

# Sidebar simple
st.sidebar.title("Navigation")
st.sidebar.success("Données connectées S3")
st.sidebar.markdown("---")
st.sidebar.info("Projet Bank Marketing")

# Titre Principal
st.title("🏦 Optimisation des Campagnes Marketing")
st.markdown("### Analyse de la performance et ciblage prédictif")
st.markdown("---")

# --- SECTION KPI MACRO ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Volume Clients", value=f"{df.shape[0]:,}".replace(",", " "))
with col2:
    st.metric(label="Taux de Conversion", value=f"{conversion_rate:.2f} %")
with col3:
    st.metric(label="Âge Moyen", value=f"{df['age'].mean():.0f} ans")
with col4:
    st.metric(label="Durée Moyenne", value=f"{df['duration'].mean()/60:.1f} min")

st.markdown("---")

# --- SECTION 1 : APERÇU DES DONNÉES ---
with st.expander("👁️ Afficher un aperçu des données brutes"):
    st.dataframe(df.head())

# --- SECTION 2 : LE PROBLÈME (Déséquilibre) ---
st.header("1. Analyse de la Cible (Target)")

c1, c2 = st.columns([2, 1])

with c1:
    # Graphique standard avec Seaborn (Palette 'Set2' classique)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=COLONNE_CIBLE, data=df, order=['no', 'yes'], palette="Set2")
    
    ax.set_title("Distribution des Souscriptions (Oui/Non)")
    ax.set_ylabel("Nombre de clients")
    ax.set_xlabel("Résultat de la campagne")
    
    # Fond blanc simple pour le graph
    sns.despine()
    st.pyplot(fig)

with c2:
    # Composant natif st.warning pour l'alerte
    st.warning("⚠️ **ALERTE DATA : DÉSÉQUILIBRE**")
    st.markdown("""
    On constate une majorité écrasante de refus (**'no'**).
    
    **Impact pour le Machine Learning :**
    * L'Accuracy (Précision globale) sera trompeuse.
    * Un modèle qui prédit "Non" tout le temps aura ~88% de réussite.
    
    👉 **Action :** Nous utiliserons le **F1-Score** ou le **Recall** pour évaluer la performance réelle.
    """)
    
st.markdown("---")

# --- SECTION 3 : PROFIL CLIENT (WHO) ---
st.header("2. PROFILING : QUI EST LE CLIENT IDÉAL ?")

st.markdown("""
Ici, nous comparons le **Volume** (qui on appelle le plus) à la **Performance** (qui signe vraiment).
L'objectif est d'identifier les segments sous-exploités.
""")

# 1. PRÉPARATION DES DONNÉES (Calculs des Taux de Conversion)
# On crée un petit tableau récapitulatif par Métier
df_job = df.groupby('metier').agg(
    Volume=('souscription', 'count'),
    Conversion_Rate=('souscription', lambda x: (x == 'yes').mean() * 100)
).sort_values(by='Conversion_Rate', ascending=False).reset_index()

# 2. VISUALISATION MÉTIER (Volume vs Performance)
st.subheader("A. Analyse par Métier (Job)")

col_job1, col_job2 = st.columns(2)

with col_job1:
    st.markdown("**Volume d'appels par métier**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # On trie par volume pour voir les "gros" segments
    order_vol = df['metier'].value_counts().index
    sns.countplot(y='metier', data=df, order=order_vol, palette="Blues_r")
    ax1.set_xlabel("Nombre d'appels")
    ax1.set_ylabel("")
    sns.despine()
    st.pyplot(fig1)

with col_job2:
    st.markdown("**Taux de Conversion (%) par métier**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    # On garde le même ordre que le taux de conversion calculé plus haut
    sns.barplot(y='metier', x='Conversion_Rate', data=df_job, palette="Greens_r")
    ax2.set_xlabel("Taux de réussite (%)")
    ax2.set_ylabel("")
    # On ajoute une ligne verticale pour la moyenne globale
    ax2.axvline(conversion_rate, color='red', linestyle='--', label=f'Moyenne ({conversion_rate:.1f}%)')
    ax2.legend()
    sns.despine()
    st.pyplot(fig2)

# Petit commentaire Business automatique
top_job = df_job.iloc[0]['metier']
top_perf = df_job.iloc[0]['Conversion_Rate']
flop_job = df_job.iloc[-1]['metier']

st.info(f"""
**Observation Business :** Le métier **{top_job}** est le plus performant avec **{top_perf:.1f}%** de réussite.
A l'inverse, **{flop_job}** convertit mal, malgré un volume souvent élevé.

👉 *Stratégie : Réallouer les efforts des profils à faible rendement vers les profils performants.*
""")


# 3. VISUALISATION AGE & STATUT
st.subheader("B. Analyse Démographique")

col_age1, col_age2 = st.columns(2)

with col_age1:
    st.markdown("**Performance par Tranche d'Âge**")
    # On calcule le taux par age_group (si la colonne existe grâce à ton fichier)
    if 'age_group' in df.columns:
        df_age = df.groupby('age_group')['souscription'].apply(lambda x: (x=='yes').mean() * 100).reset_index()
        
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='age_group', y='souscription', data=df_age, palette="Purples")
        ax3.set_ylabel("Taux de Conversion (%)")
        ax3.set_xlabel("Groupe d'Âge")
        sns.despine()
        st.pyplot(fig3)
    else:
        st.warning("Colonne 'age_group' introuvable.")

with col_age2:
    st.markdown("**Performance par Statut Matrimonial**")
    # Calcul par statut
    df_statut = df.groupby('statut_matrimonial')['souscription'].apply(lambda x: (x=='yes').mean() * 100).reset_index()
    
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.barplot(x='statut_matrimonial', y='souscription', data=df_statut, palette="Oranges")
    ax4.set_ylabel("Taux de Conversion (%)")
    ax4.set_xlabel("Statut")
    sns.despine()
    st.pyplot(fig4)
    
st.markdown("---")

# --- SECTION 4 : STRATÉGIE TEMPORELLE (WHEN) ---
st.header("3. TIMING : QUAND LANCER LES CAMPAGNES ?")

st.markdown("""
Analyse de la **Saisonnalité** (Mois) et de la **Pression Marketing** (Nombre de contacts).
Le but est d'optimiser le planning des équipes.
""")

# 1. ANALYSE MENSUELLE (Saisonnalité)
st.subheader("A. Le Paradoxe du Mois de Mai (Volume vs Performance)")

# On définit l'ordre chronologique des mois (sinon c'est alphabétique)
ordre_mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Préparation des données
df_mois = df.groupby('mois').agg(
    Volume=('souscription', 'count'),
    Taux_Conversion=('souscription', lambda x: (x == 'yes').mean() * 100)
).reindex(ordre_mois).dropna() # On réindexe pour avoir l'ordre Jan->Dec

col_mois1, col_mois2 = st.columns([2, 1])

with col_mois1:
    # Graphique à double axe (Combo Chart)
    fig5, ax1 = plt.subplots(figsize=(10, 5))

    # Axe 1 (Gauche) : Barres pour le Volume
    sns.barplot(x=df_mois.index, y='Volume', data=df_mois, color='lightgrey', alpha=0.6, ax=ax1, label='Volume Appels')
    ax1.set_ylabel("Volume d'appels (Barres)", color='grey')
    
    # Axe 2 (Droit) : Ligne pour la Performance
    ax2 = ax1.twinx()
    sns.lineplot(x=df_mois.index, y='Taux_Conversion', data=df_mois, color='red', marker='o', linewidth=3, ax=ax2, label='Taux de Réussite')
    ax2.set_ylabel("Taux de Conversion % (Ligne Rouge)", color='red')
    
    # Titre et affichage
    plt.title("Volume vs Performance par Mois")
    st.pyplot(fig5)

with col_mois2:
    st.info("""
    📉 **Analyse :**
    Regardez le mois de **Mai (may)** : C'est le pic d'appels (barre grise géante), mais le taux de réussite (ligne rouge) s'effondre.
    
    ✅ **Opportunité :**
    Les mois de **Mars, Septembre, Octobre** ont peu d'appels mais d'excellents taux de conversion.
    """)


# 2. ANALYSE DE LA PRESSION (Nombre d'appels)
st.subheader("B. Acharnement vs Efficacité (Combien d'appels ?)")

# On calcule le taux de réussite par nombre d'appels (campaign)
# On limite l'affichage aux 10 premiers appels pour la lisibilité
df_campaign = df.groupby('campaign')['souscription'].apply(lambda x: (x=='yes').mean() * 100).reset_index()
df_campaign = df_campaign[df_campaign['campaign'] <= 10] 

col_cam1, col_cam2 = st.columns([2, 1])

with col_cam1:
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    sns.lineplot(x='campaign', y='souscription', data=df_campaign, marker='o', color='purple')
    
    # Zone de danger (après 3 appels)
    plt.axvline(x=3, color='red', linestyle='--', alpha=0.5)
    plt.text(3.2, df_campaign['souscription'].max(), 'Zone de Harcèlement', color='red')
    
    ax6.set_title("Chute de la conversion après X appels")
    ax6.set_xlabel("Nombre de contacts durant cette campagne")
    ax6.set_ylabel("Probabilité de succès (%)")
    ax6.set_xticks(range(1, 11)) # Force les entiers 1, 2, 3...
    sns.despine()
    st.pyplot(fig6)

with col_cam2:
    st.warning("""
    ⚠️ **Stop ou Encore ?**
    
    La courbe montre clairement qu'après **3 appels**, la probabilité de vente devient quasi-nulle.
    
    Continuer à appeler au-delà de 3 fois coûte de l'argent (temps agent) et risque d'énerver le client.
    """)

st.markdown("---")

# --- SECTION 5 : ANALYSE CROISÉE (THE SNIPER VIEW) ---
st.header("4. CIBLAGE CHIRURGICAL : QUI & QUAND ?")

st.markdown("""
Cette carte de chaleur (Heatmap) croise le **Métier** et le **Mois**.
Les zones **vertes/foncées** indiquent les meilleures opportunités de vente.
""")

# 1. PRÉPARATION DES DONNÉES PIVOT
# On transforme Oui/Non en 1/0 pour pouvoir calculer la moyenne
df['target_num'] = df['souscription'].apply(lambda x: 1 if x == 'yes' else 0)

# On crée la table pivot : Métier en ligne, Mois en colonne, Taux de réussite en valeur
pivot_table = df.pivot_table(
    values='target_num',
    index='metier',
    columns='mois',
    aggfunc='mean'
) * 100 # Pour avoir des pourcentages

# On remet les mois dans l'ordre chronologique (défini dans la partie précédente)
# On vérifie que 'ordre_mois' existe, sinon on le redéfinit
ordre_mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
pivot_table = pivot_table.reindex(columns=ordre_mois)

# 2. AFFICHAGE DE LA HEATMAP
fig7, ax7 = plt.subplots(figsize=(12, 8))

# On utilise une palette "RdYlGn" (Rouge = Mauvais, Vert = Bon)
sns.heatmap(
    pivot_table, 
    annot=True,     # Affiche les chiffres dans les cases
    fmt=".1f",      # 1 chiffre après la virgule
    cmap="RdYlGn",  # Palette Rouge-Jaune-Vert
    linewidths=.5,  # Petites lignes blanches entre les cases
    cbar_kws={'label': 'Taux de Conversion (%)'}
)

ax7.set_xlabel("Mois de l'année")
ax7.set_ylabel("Catégorie Socio-Pro (Job)")
ax7.set_title("Matrice de Rentabilité : Quel profil appeler à quel moment ?")

st.pyplot(fig7)

# --- CONCLUSION FINALE ---
st.markdown("---")
st.header("🎓 RECOMMANDATIONS STRATÉGIQUES")

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.success("""
    ### ✅ CE QU'IL FAUT FAIRE
    1.  **Cibler les extrêmes :** Les **Étudiants** (Mars/Sept) et les **Retraités** (Oct/Déc) sont les segments les plus rentables.
    2.  **Optimiser le Timing :** Concentrer l'effort commercial en **Mars, Septembre et Octobre**.
    3.  **Arrêter l'acharnement :** Si le client ne signe pas au **3ème appel**, abandonner.
    """)

with col_rec2:
    st.error("""
    ### ⛔ CE QU'IL FAUT ÉVITER
    1.  **Le piège du mois de Mai :** C'est le mois avec le plus d'appels mais le pire taux de réussite. Réduire la voilure.
    2.  **Les profils "Ouvriers" (Blue-collar) :** Gros volume mais très faible conversion. À automatiser ou à déprioriser.
    3.  **Ne pas se fier à l'Accuracy :** Le modèle ML doit être calibré sur le **Recall** pour détecter les rares "Oui".
    """)

# Signature (Optionnel)
st.markdown("---")
st.caption("Dashboard réalisé avec Streamlit & AWS S3 • Données Bank Marketing")