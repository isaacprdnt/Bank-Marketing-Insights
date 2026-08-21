---
title: Bank Marketing Simulator
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<div align="center">

# 🏦 Bank Marketing Insights
### Optimisation de campagne marketing bancaire par modélisation prédictive

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![AWS S3](https://img.shields.io/badge/AWS%20S3-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/s3)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

🌐 [Voir le projet sur mon portfolio](https://isaacprdnt.com) &nbsp;•&nbsp; 🚀 [Dashboard & Simulateur (Hugging Face)](https://huggingface.co/spaces/isaacprdnt/Bank-Marketing-Insights)

</div>

---

## 🎯 Contexte business

Les campagnes de télémarketing bancaire coûtent cher en temps et en ressources. Appeler l'intégralité d'une base de données client pour proposer un dépôt à terme génère de la frustration chez les clients non intéressés et un coût d'acquisition (CAC) très élevé pour la banque.

Un modèle prédictif permet de transformer cette approche "marketing de masse" (taux de succès de 11% à l'aveugle) en un ciblage de précision — où chaque appel est justifié par une probabilité de conversion robuste.

---

## ❓ Question centrale

**Comment identifier en amont les clients ayant la plus forte probabilité de souscrire à un dépôt à terme, afin de cibler les appels et maximiser le ROI de la campagne ?**

---

## 📊 Dataset

| Variable | Description |
|----------|-------------|
| age, job, marital, education | Profil socio-démographique du client |
| balance | Solde bancaire |
| housing, loan, default | Situation financière |
| contact, month, day, duration | Modalités du contact |
| campaign, pdays, previous, poutcome | Historique des campagnes |
| **y** | **Cible : souscription au dépôt (yes/no)** |

- **45 211 entrées** | **17 variables** | Source : UCI Machine Learning Repository

---

## 🔧 Méthodologie

### Étape 1 — Architecture & Data Engineering
- Initialisation du repo GitHub (branches, README)
- Déploiement d'un bucket AWS S3 pour centraliser les données (source unique de vérité)
- Configuration des accès via AWS IAM (.env local)

### Étape 2 — Data Pipeline & Préparation (EDA)
- Traitement approfondi d'un jeu de données déséquilibré (88.3% négatifs / 11.7% positifs)
- Identification des tendances, biais et déséquilibres de classes
- Nettoyage et export automatisé vers AWS S3

### Étape 3 — Business Intelligence (Streamlit)
- Développement d'une application via `app.py` structuré pour le pilotage métier
- Connexion au bucket S3 via boto3 pour charger les données en temps réel
- Visualisations dynamiques (Streamlit/Plotly) avec slicers pour l'analyse des campagnes

### Étape 4 — Machine Learning & Analyse Prédictive
- Suppression de la variable `duration` (data leakage post-contact)
- Entraînement d'un modèle Random Forest (class_weight='balanced')
- Ajustement du seuil de probabilité pour équilibrer Précision/Recall selon les coûts vs gains
- Simulation ROI : Profit = (Nombre de Ventes × Gains) − (Nombre d'Appels × 10€)

### Étape 5 — Simulateur Prédictif
- Architecture dynamique : connexion au modèle via AWS S3 en temps réel
- Scoring nuancé : remplacement de la décision binaire par un système de "feux" (vert/orange/rouge)
- Aide à la décision : recommandations commerciales concrètes à chaque appel

---

## 📈 Résultats

| Métrique | Valeur |
|----------|--------|
| Seuil optimal retenu | 0.45 |
| Précision | 56.7% |
| Recall | 44.7% |
| Clients ciblés | 893 |
| **Profit estimé (max)** | **41 876 €** |

> Marketing de masse → 11% de succès à l'aveugle
> Ciblage prédictif → chaque appel justifié par une probabilité robuste

---

## 💡 Insights clés

- **La durée du contact** (`duration`) est fortement corrélée à la conversion — mais c'est une variable post-contact : l'inclure crée un data leakage. Elle a été exclue du modèle final.
- **Le résultat de la campagne précédente** (`poutcome`) est le meilleur prédicteur disponible avant contact
- **Le solde bancaire** et **l'âge** jouent un rôle significatif dans la probabilité de souscription
- **L'ajustement du seuil** est la clé du ROI : un seuil trop haut rate des clients convertibles, trop bas génère des appels inutiles

---

## 🚀 Prochaines étapes

**Optimisation temporelle** — identifier le meilleur moment pour lancer les campagnes (saisonnalité, heure de la journée) pour maximiser le taux de décroché.

**Continuous Learning** — passer d'un modèle statique à une architecture dynamique capable d'apprendre en temps réel, en réinjectant chaque jour les résultats des appels effectués (Feedback Loop).

---

## 📁 Structure du repo

```
Bank-Marketing-Insights/
├── data/                          # Données (Silver layer via AWS S3)
├── pages/                         # Pages Streamlit multipage
├── 1_Dashboard.py                 # Dashboard principal
├── 1_processing_eda.ipynb         # Notebook EDA & preprocessing
├── 2_machine_learningV2.ipynb     # Notebook ML & modélisation
├── main.py                        # Point d'entrée Streamlit
├── Dockerfile                     # Déploiement Docker (Hugging Face)
├── requirements.txt
└── README.md
```

---

## 👤 Auteur

**Isaac Prudent** — Business & Data Analyst
[Portfolio](https://isaacprdnt.com) &nbsp;•&nbsp; [LinkedIn](https://linkedin.com/in/isaac-prdnt/) &nbsp;•&nbsp; [GitHub](https://github.com/isaacprdnt)
