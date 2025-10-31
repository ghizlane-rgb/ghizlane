import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib
import unicodedata
from difflib import SequenceMatcher
import re
from urllib.parse import quote
import shap
import re
import numpy as np

import re
import numpy as np
import xgboost as xgb   # <-- assure‑toi d’avoir xgboost importé
# Grille de décote pour CV >= 10
# === PDF GENERATION FUNCTION ===
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from io import BytesIO
import datetime

@st.cache_data(show_spinner=False)
def generate_pdf_report(results, form_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    title_style = ParagraphStyle(
        'TitleCustom', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor("#004D98"), alignment=1, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'HeadingCustom', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor("#A50044"), spaceAfter=10
    )
    normal_style = ParagraphStyle(
        'NormalCustom', parent=styles['Normal'], fontSize=11, leading=14
    )
    gold_style = ParagraphStyle(
        'Gold', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor("#FFD700"), fontWeight='bold'
    )

    story = []

    # Titre
    story.append(Paragraph("Rapport d'Estimation Automobile", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')}", normal_style))
    story.append(Spacer(1, 20))

    # Infos véhicule
    story.append(Paragraph("Informations du Véhicule", heading_style))
    info_data = [
        ["Marque", form_data['Marque']],
        ["Modèle", form_data['Modele']],
        ["Année", str(form_data['Mc'])],
        ["Kilométrage", f"{form_data['Km']:,} km"],
        ["Carburant", form_data['Carburant']],
        ["Transmission", form_data['Transmission']],
        ["Version", form_data.get('Version', 'Non spécifiée')]
    ]
    info_table = Table(info_data, colWidths=[3*cm, 8*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#004D98")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))

    # Prix estimés
    story.append(Paragraph("Prix Estimés", heading_style))
    price_data = [
        ["Prix Neuf", format_price(results['prix_neuf'])],
        ["Prix Expert", format_price(results['prix_expert'])],
        ["Prix Moyen (ML)", format_price(results['average'])],
        ["Fourchette (P20 - P80)", f"{format_price(results['p20'])} → {format_price(results['p80'])}"],
        ["Amplitude", format_price(results['p80'] - results['p20'])]
    ]
    price_table = Table(price_data, colWidths=[5*cm, 6*cm])
    price_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#A50044")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.gold),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#fff8e1")),
        ('TEXTCOLOR', (1,1), (1,1), colors.HexColor("#FFD700")),
        ('FONTNAME', (1,1), (1,1), 'Helvetica-Bold'),
    ]))
    story.append(price_table)
    story.append(Spacer(1, 20))

    # Comparables
    if results['comparables']:
        story.append(Paragraph("Véhicules Comparables", heading_style))
        comp_data = [["Marque", "Modèle", "Km", "Année", "Prix"]]
        for c in results['comparables'][:5]:
            comp_data.append([
                c['Marque'], c['Modele'], f"{c['Km']:,}", str(c['Mc']), format_price(c['Prix'])
            ])
        comp_table = Table(comp_data, colWidths=[2.5*cm, 3*cm, 2*cm, 1.5*cm, 3*cm])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#004D98")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 10),
        ]))
        story.append(comp_table)

    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Rapport généré automatiquement par l'Estimateur Prix Automobile", normal_style))
    story.append(Paragraph("© 2025 - Ghizlane Chichouki", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
GIF_PATH = "photofunky.gif"
MOTS_TECHNIQUES = {
    'bluehdi', 'hdi', 'vvt', 'vti', 'tsi', 'tfsi', 'dci', 'cdti', 'crdi', 'multijet',
    'puretech', 'ecoboost', 'turbo', 'hybride', 'hybrid', 'electric', 'diesel', 'essence',
    '50', '60', '70', '71', '75', '80', '90', '92', '95', '100', '110', '115', '120', '122',
    '130', '140', '150', '160', '170', '180', '190', '200', '220', '240', '250', '300',
    '1.0', '1.2', '1.4', '1.5', '1.6', '1.8', '2.0', '2.2', '2.5', '3.0',
    'promo', 'promotion', 'pack', 'plus', '+', 'edition', 'limited', 'special'
}
keywords = [
  
    # Moteurs
    "BlueHDi", "dCi", "TDI", "CRDi", "HDi", "CDTi", "CDTI", "CTDi", "D_4D", "Multijet",
    "THP", "TFSI", "TSI", "VVT_i", "MPI", "GDI", "EcoBoost",
    "ETech", "Hybride", "HEV", "PHEV", "kWh",

    # Boîtes
    "EAT8", "BVA", "AT", "DCT", "STT", "STronic", "Tiptronic",

    # Traction
    "quattro", "xDrive", "sDrive", "4WD", "AWD",

    # Finitions
    "Active", "Allure", "GT", "GTi", "RS", "RS_Line", "Sline", "M_Sport",
    "Titanium", "ST_Line", "Vignale",
    "Business", "Ambiance", "Confort", "Prestige", "Luxe", "Luxury",
    "Life", "Edition", "Trend", "Elegance", "Premium", "Exclusive",
    "Techno", "Explore", "Executive", "Dynamic", "Intens",
    "Seductive", "Stepway", "Ultimate", "Avantage", "X_Line",
    "Attractive", "Panorama", "Lounge", "Innovation", "Distinctive",
    "Ambiente", "Pop", "Style", "Shine", "Feel", "Live", "Access",
    "Enjoy", "Essentia", "First", "GS", "Gold", "Privilege", "Laureate",
    "City", "Street", "Aera", "Energia", "Inventive", "Enginereed",
    "Edison", "Alpine", "Pop_Star",

    # Types
    "Combi", "van", "express", "Convertible",

    # Options
    "Pack", "promo", "Clim"
]

def parse_version_keywords(version):
    resultats = {}
    meilleure_version = version.strip()
    for kw in keywords:
        pattern = re.compile(rf"\b{re.escape(kw).replace(' ', r'[\s-]?')}\b", flags=re.IGNORECASE)
        col_name = kw.replace(" ", "_").replace("-", "_")
        resultats[col_name] = 1 if pattern.search(meilleure_version) else 0

    resultats["RS_Line"] = 1 if re.search(r"RS[\s-]?Line", meilleure_version, re.IGNORECASE) else 0
    resultats["X_Line"] = 1 if re.search(r"X[\s-]?Line", meilleure_version, re.IGNORECASE) else 0
    resultats["ST_Line"] = 1 if re.search(r"ST[\s-]?Line", meilleure_version, re.IGNORECASE) else 0
    resultats["M_Sport"] = 1 if re.search(r"M[\s-]?Sport", meilleure_version, re.IGNORECASE) else 0
    return resultats
grille_cv_plus_10 = {
    10000: {12: 75.0, 18: 68.0, 24: 65.0, 30: 58.0, 36: 52.0, 42: 48.0, 48: 45.0, 54: 38.0, 60: 33.0, 66: 31.0, 72: 30.0, 78: 27.0, 84: 25.0},
    20000: {12: 72.5, 18: 67.3, 24: 64.0, 30: 57.6, 36: 51.5, 42: 47.5, 48: 44.4, 54: 37.5, 60: 32.4, 66: 30.3, 72: 29.2, 78: 26.4, 84: 24.3},
    30000: {12: 70.0, 18: 66.5, 24: 63.0, 30: 57.1, 36: 51.0, 42: 47.0, 48: 43.7, 54: 36.9, 60: 31.9, 66: 29.6, 72: 28.5, 78: 25.8, 84: 23.7},
    40000: {12: 69.3, 18: 65.8, 24: 62.0, 30: 56.7, 36: 50.5, 42: 46.5, 48: 43.1, 54: 36.4, 60: 31.3, 66: 28.9, 72: 27.7, 78: 25.3, 84: 23.0},
    50000: {12: 68.7, 18: 65.0, 24: 61.0, 30: 56.3, 36: 50.0, 42: 46.0, 48: 42.5, 54: 35.8, 60: 30.7, 66: 28.3, 72: 27.1, 78: 24.7, 84: 22.3},
    60000: {12: 68.0, 18: 64.0, 24: 60.0, 30: 55.9, 36: 49.5, 42: 45.5, 48: 41.8, 54: 35.3, 60: 30.1, 66: 27.7, 72: 26.4, 78: 23.9, 84: 21.7},
    70000: {12: 67.3, 18: 63.0, 24: 58.9, 30: 55.4, 36: 49.0, 42: 45.0, 48: 41.2, 54: 34.8, 60: 29.6, 66: 27.1, 72: 25.8, 78: 23.2, 84: 20.9},
    80000: {12: 66.7, 18: 62.0, 24: 57.8, 30: 55.0, 36: 48.5, 42: 44.5, 48: 40.5, 54: 34.2, 60: 29.0, 66: 26.5, 72: 25.2, 78: 22.4, 84: 20.2},
    90000: {12: 66.0, 18: 61.0, 24: 56.7, 30: 53.6, 36: 48.0, 42: 44.0, 48: 39.9, 54: 33.7, 60: 28.4, 66: 25.9, 72: 24.6, 78: 21.7, 84: 19.4},
    100000: {12: 65.3, 18: 60.0, 24: 55.6, 30: 52.1, 36: 46.3, 42: 43.5, 48: 39.3, 54: 33.2, 60: 27.9, 66: 25.4, 72: 24.0, 78: 20.9, 84: 18.7},
    110000: {12: 64.7, 18: 59.0, 24: 54.4, 30: 50.7, 36: 44.7, 42: 43.0, 48: 38.6, 54: 32.6, 60: 27.3, 66: 24.9, 72: 23.5, 78: 20.2, 84: 17.9},
    120000: {12: 64.0, 18: 58.0, 24: 53.3, 30: 49.3, 36: 43.0, 42: 40.5, 48: 38.0, 54: 32.1, 60: 26.7, 66: 24.4, 72: 23.0, 78: 19.4, 84: 17.1},
    130000: {12: 63.3, 18: 57.0, 24: 52.2, 30: 47.9, 36: 41.3, 42: 38.0, 48: 35.3, 54: 31.5, 60: 26.1, 66: 23.9, 72: 22.5, 78: 18.7, 84: 16.4},
    140000: {12: 62.7, 18: 56.0, 24: 51.1, 30: 46.4, 36: 39.7, 42: 35.5, 48: 32.7, 54: 31.0, 60: 25.6, 66: 23.4, 72: 22.0, 78: 17.9, 84: 15.6},
    150000: {12: 62.0, 18: 55.0, 24: 50.0, 30: 45.0, 36: 38.0, 42: 33.0, 48: 30.0, 54: 28.0, 60: 25.0, 66: 21.0, 72: 19.0, 78: 17.0, 84: 15.0},
    160000: {12: 61.4, 18: 54.6, 24: 49.6, 30: 44.6, 36: 37.6, 42: 32.6, 48: 29.6, 54: 27.6, 60: 24.6, 66: 20.7, 72: 18.7, 78: 16.7, 84: 14.7},
    170000: {12: 60.9, 18: 54.2, 24: 49.2, 30: 44.2, 36: 37.2, 42: 32.2, 48: 29.2, 54: 27.2, 60: 24.2, 66: 20.3, 72: 18.4, 78: 16.4, 84: 14.4},
    180000: {12: 60.3, 18: 53.8, 24: 48.8, 30: 43.8, 36: 36.8, 42: 31.8, 48: 28.8, 54: 26.8, 60: 23.8, 66: 20.0, 72: 18.2, 78: 16.2, 84: 14.2},
    190000: {12: 59.8, 18: 53.4, 24: 48.4, 30: 43.4, 36: 36.4, 42: 31.4, 48: 28.4, 54: 26.4, 60: 23.4, 66: 19.7, 72: 17.9, 78: 15.9, 84: 13.9},
    200000: {12: 59.2, 18: 53.1, 24: 48.1, 30: 43.1, 36: 36.1, 42: 31.1, 48: 28.1, 54: 26.1, 60: 23.1, 66: 19.3, 72: 17.6, 78: 15.6, 84: 13.6},
    210000: {12: 58.7, 18: 52.7, 24: 47.7, 30: 42.7, 36: 35.7, 42: 30.7, 48: 27.7, 54: 25.7, 60: 22.7, 66: 19.0, 72: 17.3, 78: 15.3, 84: 13.3},
    220000: {12: 58.1, 18: 52.3, 24: 47.3, 30: 42.3, 36: 35.3, 42: 30.3, 48: 27.3, 54: 25.3, 60: 22.3, 66: 18.7, 72: 17.1, 78: 15.1, 84: 13.1},
    230000: {12: 57.6, 18: 51.9, 24: 46.9, 30: 41.9, 36: 34.9, 42: 29.9, 48: 26.9, 54: 24.9, 60: 21.9, 66: 18.3, 72: 16.8, 78: 14.8, 84: 12.8},
    240000: {12: 57.0, 18: 51.5, 24: 46.5, 30: 41.5, 36: 34.5, 42: 29.5, 48: 26.5, 54: 24.5, 60: 21.5, 66: 18.0, 72: 16.5, 78: 14.5, 84: 12.5},
    250000: {12: 56.4, 18: 51.1, 24: 46.1, 30: 41.1, 36: 34.1, 42: 29.1, 48: 26.1, 54: 24.1, 60: 21.1, 66: 17.7, 72: 16.2, 78: 14.2, 84: 12.2},
    260000: {12: 55.9, 18: 50.7, 24: 45.7, 30: 40.7, 36: 33.7, 42: 28.7, 48: 25.7, 54: 23.7, 60: 20.7, 66: 17.3, 72: 15.9, 78: 13.9, 84: 11.9},
    270000: {12: 55.3, 18: 50.3, 24: 45.3, 30: 40.3, 36: 33.3, 42: 28.3, 48: 25.3, 54: 23.3, 60: 20.3, 66: 17.0, 72: 15.7, 78: 13.7, 84: 11.7},
    280000: {12: 54.8, 18: 49.9, 24: 44.9, 30: 39.9, 36: 32.9, 42: 27.9, 48: 24.9, 54: 22.9, 60: 19.9, 66: 16.7, 72: 15.4, 78: 13.4, 84: 11.4},
    290000: {12: 54.2, 18: 49.6, 24: 44.6, 30: 39.6, 36: 32.6, 42: 27.6, 48: 24.6, 54: 22.6, 60: 19.6, 66: 16.3, 72: 15.1, 78: 13.1, 84: 11.1},
    300000: {12: 53.7, 18: 49.2, 24: 44.2, 30: 39.2, 36: 32.2, 42: 27.2, 48: 24.2, 54: 22.2, 60: 19.2, 66: 16.0, 72: 14.8, 78: 12.8, 84: 10.8},
    310000: {12: 53.1, 18: 48.8, 24: 43.8, 30: 38.8, 36: 31.8, 42: 26.8, 48: 23.8, 54: 21.8, 60: 18.8, 66: 15.7, 72: 14.6, 78: 12.6, 84: 10.6},
    320000: {12: 52.6, 18: 48.4, 24: 43.4, 30: 38.4, 36: 31.4, 42: 26.4, 48: 23.4, 54: 21.4, 60: 18.4, 66: 15.3, 72: 14.3, 78: 12.3, 84: 10.3},
    330000: {12: 52.0, 18: 48.0, 24: 43.0, 30: 38.0, 36: 31.0, 42: 26.0, 48: 23.0, 54: 21.0, 60: 18.0, 66: 15.0, 72: 14.0, 78: 12.0, 84: 10.0}
}

def trouver_valeur_proche(valeur, liste_valeurs):
    liste_triee = sorted(liste_valeurs)
    for v in liste_triee:
        if v >= valeur:
            return v
    return liste_triee[-1]
def calculer_prix_expert_ligne(ligne_test):
    try:
        cv = ligne_test.get('Cv')
        km = ligne_test.get('Km')
        prix_neuf = ligne_test.get('prix_neuf')
        mc = ligne_test.get('Mc')
        mois_scraping = ligne_test.get('MoisScraping', 10)

        annee_actuelle = 2025
        if mc and mois_scraping:
            age_mois = (annee_actuelle - mc) * 12 + (mois_scraping - 1)
        else:
            return None

        if cv is None or km is None or prix_neuf is None:
            return None

        grille = grille_cv_moins_10 if cv < 10 else grille_cv_plus_10

        ages_disponibles = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
        age_proche = trouver_valeur_proche(age_mois, ages_disponibles)

        kms_disponibles = list(grille.keys())
        km_proche = trouver_valeur_proche(km, kms_disponibles)

        pourcentage = grille[km_proche][age_proche]
        prix_expert = prix_neuf * (pourcentage / 100)
        return round(prix_expert, 2)

    except Exception as e:
        return None
def fix_xgboost_base_score(model):
    """
    Corrige le base_score corrompu après un joblib.load().
    Gère : str comme '[1.9844711E5]', list, np.array, float.
    """
    if not hasattr(model, "get_booster"):
        return model  # Pas un modèle XGBoost

    base_score = model.get_params().get("base_score")
    if base_score is None:
        return model

    cleaned = None

    # --- Cas 1 : Chaîne comme '[1.9844711E5]' ---
    if isinstance(base_score, str):
        cleaned_str = re.sub(r'[\[\]]', '', base_score).strip()
        m = re.search(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?', cleaned_str)
        if m:
            cleaned = float(m.group(0))
        else:
            raise ValueError(f"Impossible de parser base_score : {base_score}")

    # --- Cas 2 : list ou np.ndarray ---
    elif isinstance(base_score, (list, np.ndarray)):
        arr = np.array(base_score).flatten()
        if len(arr) > 0:
            cleaned = float(arr[0])
        else:
            raise ValueError("Tableau base_score vide")

    # --- Cas 3 : déjà un float ---
    elif isinstance(base_score, (int, float, np.floating)):
        return model

    else:
        raise TypeError(f"Type de base_score non géré : {type(base_score)}")

    # --- Appliquer la correction ---
    booster = model.get_booster()
    booster.set_param("base_score", cleaned)
    model.set_params(base_score=cleaned)

    print(f"[FIX] base_score corrigé : {base_score} → {cleaned}")
    return model

# AJOUT: Fonction de chargement des modèles avec cache
@st.cache_resource
def load_models():
    try:
        # Chargement des fichiers
        models_m9 = joblib.load('models_m9.joblib')
        X_train_encoded = joblib.load('X_train_encoded_m9.joblib')
        scaler = joblib.load('scaler_m9.joblib')
        df_combined = pd.read_csv('df_combined.csv')
        print("Colonnes attendues par le modèle :", X_train_encoded.columns.tolist()) 
        # CORRECTION XGBoost AVANT RETOUR (CRUCIAL)
        if 'XGB_M9' in models_m9:
            print("[INFO] Correction du base_score XGBoost dans cache_resource...")
            models_m9['XGB_M9'] = fix_xgboost_base_score(models_m9['XGB_M9'])
        
        # Forcer X_train_encoded en float pour SHAP
        X_train_encoded = X_train_encoded.astype(float)
        
        # Retourner les modèles corrigés
        return {
            'models': {
                'RandomForest': models_m9['RF_M9'],
                'GradientBoosting': models_m9['GB_M9'],
                'XGBoost': models_m9['XGB_M9'],  # ← maintenant corrigé
                'LightGBM': models_m9['LGBM_M9']
            },
            'X_train': X_train_encoded,
            'scaler': scaler,
            'df_combined': df_combined
        }
    except Exception as e:
        st.error(f"Modèles non trouvés : {e}")
        return None

# Configuration de la page avec les couleurs du FC Barcelona
st.set_page_config(
    page_title="Estimateur Prix Automobile",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec les couleurs du Barça (Bleu #004D98, Rouge #A50044, Or #EDBB00)
st.markdown("""
<style>
    /* Couleurs principales FCB */
    :root {
        --fcb-blue: #004D98;
        --fcb-red: #A50044;
        --fcb-gold: #EDBB00;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #004D98 0%, #A50044 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 77, 152, 0.3);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #EDBB00;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .stCard {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 77, 152, 0.15);
        border-left: 5px solid #004D98;
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #004D98 0%, #A50044 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 77, 152, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(165, 0, 68, 0.4);
    }
    
    /* Selectbox et Input */
    .stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
        border: 2px solid #004D98;
        border-radius: 8px;
    }
    
    /* Prix principal */
    .price-card {
        background: linear-gradient(135deg, #004D98 0%, #A50044 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0, 77, 152, 0.4);
    }
    
    .price-card h2 {
        color: #EDBB00;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .price-card .price {
        color: white;
        font-size: 4rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .price-card .range {
        color: white;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    
    /* Modèles ML */
    .model-prediction {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #004D98;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 77, 152, 0.1);
    }
    
    .model-prediction:hover {
        border-left-color: #A50044;
        transform: translateX(5px);
        transition: all 0.3s ease;
    }
    
    /* Comparables */
    .comparable-card {
        background: linear-gradient(to right, #f8f9fa, white);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #EDBB00;
        margin: 0.5rem 0;
    }
    
    /* Success messages */
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #EDBB00 0%, #FFC107 100%);
        color: #004D98;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #004D98 0%, #A50044 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        color: #004D98;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #004D98 0%, #A50044 100%);
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #004D98 0%, #A50044 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #004D98;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def normalize_text(s):
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('-', ' ')
    return s

def clean_price(price_str):
    clean_str = str(price_str).lower().replace("dh", "").replace(".", "").replace(",", "").strip()
    try:
        return int(clean_str)
    except ValueError:
        return None

def format_price(price):
    return f"{price:,.0f} MAD".replace(",", " ")

# URL de l'API Lambda
LAMBDA_URL = "https://w7e62hoex6.execute-api.us-east-1.amazonaws.com/prod/getScrapingData"

def load_data():
    try:
        response = requests.get(LAMBDA_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Convertir en DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            st.error("Format de données non supporté")
            return pd.DataFrame()

        # Nettoyage des colonnes numériques si elles existent
        for col in ['Km', 'Mc']:
            if col in df.columns:
                cleaned = df[col].astype(str).str.replace(r'[^0-9]', '', regex=True)
                cleaned = cleaned.replace('', pd.NA)
                df[col] = pd.to_numeric(cleaned, errors='coerce').astype('Int64')

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du traitement des données: {e}")
        return pd.DataFrame()

# CONFIGURATION GOOGLE SHEET
sheet_id = "1lgi96Qu-eQH1D2rNNGwUsbW66uChboEzHZS0cLZKO0k"
sheet_name = "Références"
sheet_name_enc = quote(sheet_name)
google_sheet_csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={sheet_name_enc}"

def load_google_sheet():
    try:
        df_sheet = pd.read_csv(google_sheet_csv_url)
        for col in ['Marque', 'Modele', 'Fuel', 'Version', 'Annee', 'Price']:
            if col in df_sheet.columns:
                df_sheet[col] = df_sheet[col].astype(str).str.strip().str.lower()
        return df_sheet
    except Exception as e:
        st.error(f"ERREUR lors du chargement du Google Sheet: {e}")
        return pd.DataFrame()

def normaliser_marque(marque):
    return 'mercedes' if str(marque).strip().lower() == 'mercedes-benz' else str(marque).strip().lower()

def similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()



def extraire_mots_cles(version):
    if not version or pd.isna(version):
        return []
    version_clean = re.sub(r'[+\-_/]', ' ', str(version).lower())
    mots = re.findall(r'\b[a-zA-Z]+\b', version_clean)
    return [m for m in mots if len(m) >= 3 and m not in MOTS_TECHNIQUES]

def correspondance_par_mots_communs(version_df, version_sheet):
    mots_df = extraire_mots_cles(version_df)
    mots_sheet = extraire_mots_cles(version_sheet)
    if not mots_df or not mots_sheet:
        return False
    for m1 in mots_df:
        for m2 in mots_sheet:
            if similarity(m1, m2) >= 0.9:
                return True
    return False

VERSIONS_VALIDES_PAR_DEFAUT = {'van', 'express'}

def chercher_version_unique(row, df_sheet):
    marque = normaliser_marque(row.get('Marque', ''))
    modele = str(row.get('Modele', '')).strip().lower()
    carburant = str(row.get('Carburant', '')).strip().lower()
    mc = str(row.get('Mc', '')).strip()
    version_originale = str(row.get('Version', '')).strip()

    if version_originale.lower() in VERSIONS_VALIDES_PAR_DEFAUT:
        return {
            'status': 'VERSION_VALIDE',
            'version_corrigee': version_originale,
            'prix_neuf': None,
            'message': 'Version valide par défaut'
        }

    if not marque or not modele:
        return {'status': 'ERREUR', 'message': 'Marque ou Modèle manquant'}

    df_candidats = df_sheet[
        (df_sheet['Marque'] == marque) &
        (df_sheet['Modele'] == modele)
    ].copy()

    if df_candidats.empty:
        return {'status': 'NON_TROUVE', 'message': f'Aucun {marque}/{modele} dans Google Sheet'}

    if 'Fuel' in df_sheet.columns and carburant and carburant != 'nan':
        df_candidats = df_candidats[df_candidats['Fuel'] == carburant]

    if 'Annee' in df_sheet.columns and mc and mc != 'nan':
        df_candidats = df_candidats[df_candidats['Annee'].astype(str).str.strip() == mc]

    versions_disponibles = df_candidats['Version'].dropna().unique().tolist()
    if not versions_disponibles:
        return {'status': 'AUCUNE_VERSION', 'message': 'Aucune version disponible'}

    # 1. Similarité globale > 90%
    meilleure_sim = 0
    meilleure_version = None
    prix_neuf = None
    for v in versions_disponibles:
        sim = similarity(version_originale, v)
        if sim >= 0.9 and sim > meilleure_sim:
            meilleure_sim = sim
            meilleure_version = v
            prix_row = df_candidats[df_candidats['Version'] == v]
            prix_neuf = prix_row['Price'].iloc[0] if 'Price' in prix_row.columns and not prix_row['Price'].empty else None

    if meilleure_version:
        return {
            'status': 'TOLERANCE_10%',
            'version_corrigee': meilleure_version,
            'prix_neuf': prix_neuf,
            'message': f'Trouvé avec {meilleure_sim:.1%} de similarité'
        }

    # 2. Mots communs
    for v in versions_disponibles:
        if correspondance_par_mots_communs(version_originale, v):
            prix_row = df_candidats[df_candidats['Version'] == v]
            prix_neuf = prix_row['Price'].iloc[0] if 'Price' in prix_row.columns and not prix_row['Price'].empty else None
            return {
                'status': 'MOTS_COMMUNS',
                'version_corrigee': v,
                'prix_neuf': prix_neuf,
                'message': 'Trouvé par mots communs'
            }

    return {
        'status': 'AUCUNE_CORRESPONDANCE',
        'version_corrigee': version_originale,
        'prix_neuf': None,
        'message': f'Aucune version similaire parmi {len(versions_disponibles)} versions'
    }

# Grille de décote pour CV < 10
grille_cv_moins_10 = {
    10000: {12: 80.0, 18: 72.0, 24: 71.0, 30: 66.0, 36: 63.0, 42: 58.0, 48: 52.0, 54: 48.0, 60: 43.0, 66: 41.0, 72: 40.0, 78: 37.0, 84: 35.0},
    20000: {12: 78.5, 18: 71.5, 24: 70.4, 30: 65.7, 36: 62.6, 42: 57.7, 48: 51.8, 54: 47.8, 60: 42.8, 66: 40.6, 72: 39.6, 78: 36.6, 84: 34.6},
    30000: {12: 77.0, 18: 71.0, 24: 69.8, 30: 65.4, 36: 62.3, 42: 57.4, 48: 51.6, 54: 47.5, 60: 42.6, 66: 40.3, 72: 39.3, 78: 36.2, 84: 34.2},
    40000: {12: 76.4, 18: 70.5, 24: 69.2, 30: 65.1, 36: 61.9, 42: 57.1, 48: 51.5, 54: 47.3, 60: 42.4, 66: 39.9, 72: 38.9, 78: 35.8, 84: 33.9},
    50000: {12: 75.8, 18: 70.0, 24: 68.6, 30: 64.9, 36: 61.5, 42: 56.8, 48: 51.3, 54: 47.1, 60: 42.1, 66: 39.5, 72: 38.5, 78: 35.3, 84: 33.5},
    60000: {12: 75.3, 18: 69.6, 24: 68.0, 30: 64.6, 36: 61.1, 42: 56.5, 48: 51.1, 54: 46.8, 60: 41.9, 66: 39.2, 72: 38.2, 78: 35.0, 84: 33.1},
    70000: {12: 74.7, 18: 69.2, 24: 67.4, 30: 64.3, 36: 60.8, 42: 56.2, 48: 50.9, 54: 46.6, 60: 41.7, 66: 38.8, 72: 37.8, 78: 34.7, 84: 32.7},
    80000: {12: 74.1, 18: 68.8, 24: 66.9, 30: 64.0, 36: 60.4, 42: 55.9, 48: 50.7, 54: 46.4, 60: 41.5, 66: 38.4, 72: 37.4, 78: 34.4, 84: 32.3},
    90000: {12: 73.5, 18: 68.4, 24: 66.3, 30: 63.4, 36: 60.0, 42: 55.6, 48: 50.5, 54: 46.2, 60: 41.3, 66: 38.0, 72: 37.0, 78: 34.0, 84: 31.9},
    100000: {12: 72.9, 18: 68.0, 24: 65.8, 30: 62.9, 36: 59.2, 42: 55.3, 48: 50.4, 54: 45.9, 60: 41.1, 66: 37.6, 72: 36.6, 78: 33.7, 84: 31.5},
    110000: {12: 72.3, 18: 67.6, 24: 65.2, 30: 62.3, 36: 58.3, 42: 55.0, 48: 50.2, 54: 45.7, 60: 40.9, 66: 37.2, 72: 36.2, 78: 33.4, 84: 31.1},
    120000: {12: 71.8, 18: 67.2, 24: 64.7, 30: 61.7, 36: 57.5, 42: 54.3, 48: 50.0, 54: 45.5, 60: 40.6, 66: 36.8, 72: 35.8, 78: 33.1, 84: 30.7},
    130000: {12: 71.2, 18: 66.8, 24: 64.1, 30: 61.1, 36: 56.7, 42: 53.5, 48: 49.3, 54: 45.2, 60: 40.4, 66: 36.4, 72: 35.4, 78: 32.7, 84: 30.3},
    140000: {12: 70.6, 18: 66.4, 24: 63.6, 30: 60.6, 36: 55.8, 42: 52.8, 48: 48.7, 54: 45.0, 60: 40.2, 66: 36.0, 72: 35.0, 78: 32.4, 84: 29.9},
    150000: {12: 70.0, 18: 66.0, 24: 63.0, 30: 60.0, 36: 55.0, 42: 52.0, 48: 48.0, 54: 43.0, 60: 40.0, 66: 36.0, 72: 35.0, 78: 32.0, 84: 30.0},
    160000: {12: 69.4, 18: 65.4, 24: 62.4, 30: 59.4, 36: 54.4, 42: 51.4, 48: 47.4, 54: 42.4, 60: 39.4, 66: 35.6, 72: 34.6, 78: 31.6, 84: 29.6},
    170000: {12: 68.9, 18: 64.9, 24: 61.9, 30: 58.9, 36: 53.9, 42: 50.9, 48: 46.9, 54: 41.8, 60: 38.9, 66: 35.1, 72: 34.1, 78: 31.1, 84: 28.9},
    180000: {12: 68.3, 18: 64.3, 24: 61.3, 30: 58.3, 36: 53.3, 42: 50.3, 48: 46.3, 54: 41.3, 60: 38.3, 66: 34.7, 72: 33.7, 78: 30.7, 84: 28.5},
    190000: {12: 67.7, 18: 63.7, 24: 60.7, 30: 57.7, 36: 52.7, 42: 49.7, 48: 45.7, 54: 40.7, 60: 37.7, 66: 34.3, 72: 33.3, 78: 30.3, 84: 28.1},
    200000: {12: 67.1, 18: 63.1, 24: 60.1, 30: 57.1, 36: 52.1, 42: 49.1, 48: 45.1, 54: 40.1, 60: 37.1, 66: 33.9, 72: 32.9, 78: 29.9, 84: 27.7},
    210000: {12: 66.5, 18: 62.5, 24: 59.5, 30: 56.5, 36: 51.5, 42: 48.5, 48: 44.5, 54: 39.5, 60: 36.5, 66: 33.5, 72: 32.5, 78: 29.5, 84: 27.3},
    220000: {12: 65.9, 18: 61.9, 24: 58.9, 30: 55.9, 36: 50.9, 42: 47.9, 48: 43.9, 54: 38.9, 60: 35.9, 66: 33.1, 72: 32.1, 78: 29.1, 84: 26.9},
    230000: {12: 65.3, 18: 61.3, 24: 58.3, 30: 55.3, 36: 50.3, 42: 47.3, 48: 43.3, 54: 38.3, 60: 35.3, 66: 32.7, 72: 31.7, 78: 28.7, 84: 26.5},
    240000: {12: 64.7, 18: 60.7, 24: 57.7, 30: 54.7, 36: 49.7, 42: 46.7, 48: 42.7, 54: 37.7, 60: 34.7, 66: 32.3, 72: 31.3, 78: 28.3, 84: 26.1},
    250000: {12: 64.1, 18: 60.1, 24: 57.1, 30: 54.1, 36: 49.1, 42: 46.1, 48: 42.1, 54: 37.1, 60: 34.1, 66: 31.9, 72: 30.9, 78: 27.9, 84: 25.7}
}

# Note: Les parties tronquées comme grille_cv_plus_10, calculer_prix_expert_ligne, parse_version_keywords doivent être complétées si disponibles, mais comme tronquées, assumez qu'elles sont définies ailleurs ou ajoutez-les si nécessaire.


def predict_vehicle_price_with_shap(models, X_encoded_columns, scaler, vehicle_data, X_encoded=None, max_features_shap=5):
    print("\n🔍 Predicting price for new vehicle...")

    test_df = pd.DataFrame([vehicle_data])
    categorical_cols = ['Carburant', 'Couleur', 'Etat', 'Marque', 'Modele',
                       'Origine', 'Transmission', 'PremiereMain', 'Source', 'location']
    categorical_cols = [col for col in categorical_cols if col in test_df.columns]

    def normalize_text(s):
        s = str(s).lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = s.replace('-', ' ')
        return s

    for col in categorical_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].apply(normalize_text)

    required_cols = ['Km', 'Mc', 'Marque', 'Modele']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in vehicle_data: {missing_cols}")

    try:
        test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
    except Exception as e:
        print(f"❌ Error during one-hot encoding: {str(e)}")
        raise

    missing_cols = [col for col in X_encoded_columns if col not in test_encoded.columns]
    if missing_cols:
        test_encoded = pd.concat([test_encoded, pd.DataFrame(0, index=test_encoded.index, columns=missing_cols)], axis=1)
    test_encoded = test_encoded[X_encoded_columns]

    try:
        test_scaled = scaler.transform(test_encoded)
        test_scaled = pd.DataFrame(test_scaled, columns=X_encoded_columns, index=test_encoded.index)
    except Exception as e:
        print(f"❌ Error during scaling: {str(e)}")
        raise

    predictions = {}
    explanations = {}

    for model_name, model in models.items():
        try:
            pred = model.predict(test_scaled)[0]
            predictions[model_name] = float(pred)  # Convert to float for JSON serialization

            if X_encoded is not None:
                try:
                    explainer = shap.TreeExplainer(model, X_encoded.astype(float))
                    shap_values = explainer(test_scaled.astype(float))
                    shap_vals = shap_values.values[0] if shap_values.values.ndim > 1 else shap_values.values

                    local_df = pd.DataFrame({
                        'feature': X_encoded_columns,
                        'importance': np.abs(shap_vals),
                        'contribution': shap_vals,
                        'value': test_encoded.iloc[0].values
                    }).sort_values('importance', ascending=False).head(max_features_shap)

                    explanations[model_name] = {
                        'local_top_features': local_df.to_dict(orient='records'),  # Convert DataFrame to list of dicts
                        'base_value': float(explainer.expected_value)  # Convert to float for JSON
                    }
                except Exception as e:
                    #print(f"⚠️ SHAP explanation failed for {model_name}: {str(e)}")
                    explanations[model_name] = None
            else:
                explanations[model_name] = None
        except Exception as e:
            print(f"❌ Error predicting with {model_name}: {str(e)}")
            predictions[model_name] = None  # Use None instead of np.nan for JSON compatibility
            explanations[model_name] = None

    average_prediction = float(np.nanmean([p for p in predictions.values() if p is not None]))
    return {
        'predictions': predictions,
        'average_prediction': average_prediction,
        'explanations': explanations
    }
def find_similar_cars(df, car_test, limit=5):
    car_test = {k: normalize_text(v) if k in ['Marque', 'Modele'] else v for k, v in car_test.items()}

    subset = df[
        (df['Marque'].str.lower() == car_test['Marque'].lower()) &
        (df['Modele'].str.lower() == car_test['Modele'].lower())
    ]

    if subset.empty:
        return pd.DataFrame()

    km_min, km_max = car_test['Km'] * 0.9, car_test['Km'] * 1.1
    mc_min, mc_max = car_test['Mc'] - 1, car_test['Mc'] + 1

    filtered = subset[
        (subset['Km'].between(km_min, km_max)) &
        (subset['Mc'].between(mc_min, mc_max))
    ]

    if filtered.empty:
        filtered = subset

    return filtered.head(limit)

def price_fair_range(preds):
    all_preds = np.array([p for p in preds if p is not None])
    if len(all_preds) == 0:
        return None, None, None, None
    p20 = float(np.percentile(all_preds, 20))
    p50 = float(np.percentile(all_preds, 50))
    p80 = float(np.percentile(all_preds, 80))
    width = p80 - p20
    return p20, p50, p80, width

# Initialisation de la session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 Estimateur de Prix Automobile</h1>
    <p>Analyse intelligente basée sur le Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Progression
with st.sidebar:
    st.markdown("### 📊 Progression")
    progress = (st.session_state.step - 1) / 2
    st.progress(progress)
    st.markdown(f"**Étape {st.session_state.step} sur 3**")
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info("À l’aide de l’Intelligence Artificielle et des données en temps réel, nous avons réalisé cette application d’estimation automobile.Elle fournit des prix fiables, des analyses comparatives et des recommandations pour optimiser la vente de votre véhicule, en combinant prédictions multi-modèles et visualisations interactives.Réalisé par Ghizlane chouki.")
    
   

# ÉTAPE 1 : Informations principales
if st.session_state.step == 1:
    st.markdown("## 📝 Étape 1 : Informations du Véhicule")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transmission = st.selectbox("🔧 Transmission", ["Automatique", "Manuelle"])
        carburant = st.selectbox("⛽ Carburant", ["Diesel", "Essence", "Hybride", "Électrique"])
        origine = st.selectbox("🌍 Origine", ["maroc", "dedouanee", "importee","ww"])
    
    with col2:
        source = st.selectbox("📱 Source", ['kifal','autocaz','ayvens','arval','wandaloo','hasta','vivalis','eqdom','wafa','avito','globaloccaz','auto24','salafin','autocash','sofac'])
        marque = st.selectbox("🏭 Marque", ["AUDI", "BMW", "Dacia", "Peugeot", "Volkswagen", "Mercedes-Benz", "Renault", "Citroën"])
        premiere_main = st.selectbox("👤 Première Main", ["Oui", "Non"])
    
    with col3:
        couleur = st.selectbox("🎨 Couleur", ["noir", "blanc", "gris", "rouge", "bleu", "argent"])
        etat = st.selectbox("⭐ État", [ "bon", "très bon", "neuf", "excellent"])
        location = st.selectbox("📍 Ville", ['Casablanca', 'Fès','Rabat', 'Agadir', 'Safi', 'Marrakech','Tanger', 'Settat', 'Souk El Arbaa','Meknès', 'tétouan', 'Témara','Salé', 'Oujda', 'Kénitra', 'Laâyoune', 'Errachidia', 'Mohammedia','nouaceur', 'Béni Mellal', 'Bouznika', 'Nador', 'Sraghna', 'Taza','Khouribga', 'Sidi Kacem','ouazzane', 'midelt', 'El Jadida', 'Ouarzazate', 'missour','Boujdour', 'Berrechid', 'Bouskoura', 'Sefrou'])
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("➡️ Continuer vers l'étape 2", use_container_width=True):
            if all([transmission, source, carburant, marque, origine, premiere_main, couleur, etat, location]):
                st.session_state.form_data.update({
                    'Transmission': transmission,
                    'Source': source,
                    'Carburant': carburant,
                    'Marque': marque,
                    'Origine': origine,
                    'PremiereMain': premiere_main,
                    'Couleur': couleur,
                    'Etat': etat,
                    'location': location
                })
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("⚠️ Veuillez remplir tous les champs")

# ÉTAPE 2 : Détails techniques
elif st.session_state.step == 2:
    st.markdown("## 🔧 Étape 2 : Détails Techniques")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        km = st.number_input("📏 Kilométrage (km)", min_value=0, value=50000, step=1000)
        modele = st.text_input("🚘 Modèle", value="Q5")
        cv = st.number_input("⚡ Puissance fiscale (CV)", min_value=0.0, value=8.0, step=1.0)
    
    with col2:
        mc = st.number_input("📅 Année de mise en circulation", min_value=2000, max_value=2025, value=2023, step=1)
        portes = st.number_input("🚪 Nombre de portes", min_value=2, max_value=5, value=5, step=1)
        mois_scraping = st.number_input("📆 Mois de référence", min_value=1, max_value=12, value=10, step=1)
    
    version = st.text_input("📋 Version", value="30 TDI 136 S-Tronic Advanced")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    
    with col3:
        if st.button("🚀 Estimer le Prix", use_container_width=True):
            if all([km, modele, cv, mc, portes, mois_scraping, version]):
                st.session_state.form_data.update({
                    'Km': int(km),
                    'Modele': modele,
                    'Cv': float(cv),
                    'Mc': int(mc),
                    'Portes': int(portes),
                    'MoisScraping': float(mois_scraping),
                    'Version': version
                })
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("⚠️ Veuillez remplir tous les champs")

# ÉTAPE 3 : Résultats
elif st.session_state.step == 3:
    with st.spinner("🔄 Analyse en cours... Veuillez patienter"):
        placeholder = st.empty()
        placeholder.image("photofunky.gif", width=1000)
        # Chargement des données réelles
        df = load_data()
        df_sheet = load_google_sheet()
        ligne_test = st.session_state.form_data.copy()

        # Correction de la version et prix neuf
        resultat = chercher_version_unique(ligne_test, df_sheet)
        print('*************',resultat)
        prix_neuf = clean_price(resultat['prix_neuf'])
        print('*************',prix_neuf)
        ligne_test["prix_neuf"] = prix_neuf
        ligne_test["Version"] = resultat['version_corrigee']

        # Calcul du prix expert
        prix_expert = calculer_prix_expert_ligne(ligne_test)  # Assumez défini

        # Parsing des keywords de la version
        resultats_keywords = parse_version_keywords(ligne_test["Version"])  # Assumez défini

        # Création du dict_final
        dict_final = ligne_test.copy()
        dict_final["prix_expert"] = prix_expert
        dict_final.update(resultats_keywords)
        print('*************',dict_final)
        del dict_final["Version"]

        # Chargement des modèles avec cache (CORRECTION INTÉGRÉE)
        loaded_data = load_models()
        if loaded_data is None:
            st.error("Impossible de charger les modèles ML")
            st.stop()

        # Extraction des données
        trained_models = loaded_data['models']
        X_train_encoded = loaded_data['X_train']
        scaler = loaded_data['scaler']
        df_combined = loaded_data['df_combined']

        # Prédictions ML
        result = predict_vehicle_price_with_shap(
            models=trained_models,
            X_encoded_columns=X_train_encoded.columns.tolist(),
            scaler=scaler,
            vehicle_data=dict_final,
            X_encoded=X_train_encoded,
            max_features_shap=5
        )

        # Voitures similaires
        similar_cars = find_similar_cars(df_combined, dict_final, limit=5)

        # Fourchette de prix
        preds = list(result['predictions'].values())
        p20, p50, p80, width = price_fair_range(preds)
        average = result['average_prediction']
        predictions = result['predictions']
        comparables = similar_cars[['Marque', 'Modele', 'Km', 'Mc', 'Prix', 'Carburant', 'Transmission']].to_dict(orient='records') if not similar_cars.empty else []

        st.session_state.results = {
            'predictions': predictions,
            'average': average,
            'p20': p20,
            'p50': p50,
            'p80': p80,
            'comparables': comparables,
            'prix_neuf': prix_neuf,
            'prix_expert': prix_expert
        }

    # Affichage des résultats
    st.markdown("## 🎯 Résultats de l'Estimation")
    st.markdown("---")
    placeholder.empty()
    # Success message
    st.markdown(f"""
    <div class="success-box">
        <h2>✅ Estimation Complétée avec Succès!</h2>
        <p>Analyse pour {st.session_state.form_data['Marque']} {st.session_state.form_data['Modele']} - {st.session_state.form_data['Mc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prix principal
    st.markdown(f"""
    <div class="price-card">
        <h2>💰 Prix Estimé Moyen</h2>
        <div class="price">{format_price(st.session_state.results['average'])}</div>
        <div class="range">
            Fourchette: {format_price(st.session_state.results['p20'])} - {format_price(st.session_state.results['p80'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage prix neuf et prix expert
    
    

    st.markdown("### Prix Neuf et Prix Expert")

    st.markdown("""
<style>
.price-container1 {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin: 20px 0;
    width: 100%;
}
.price-card1 {
    background: linear-gradient(135deg, #004D98 0%, #A50044 100%);
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    flex: 1;                    /* Chaque carte prend 50% */
    min-width: 280px;
    border: 2px solid #FFD700;  /* Bordure dorée */
}
.price-card1 h2 {
    margin: 0 0 12px 0;
    color: #FFD700;             /* Titre en OR */
    font-size: 1.5em;
    font-weight: bold;
}
.price1 {
    font-size: 2.2em;
    font-weight: bold;
    color: #FFD700;             /* Prix en OR */
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Responsive : mobile = 1 colonne */
@media (max-width: 768px) {
    .price-container1 {
        flex-direction: column;
        align-items: center;
    }
    .price-card1 {
        width: 90%;
        max-width: 400px;
    }
}
</style>
""", unsafe_allow_html=True)

# === AFFICHAGE DES 2 CARTES ===
    st.markdown(f"""
<div class="price-container1">
    <div class="price-card1">
        <h2>Prix Neuf</h2>
        <div class="price1">{format_price(st.session_state.results['prix_neuf'])}</div>
    </div>
    <div class="price-card1">
        <h2>Prix Expert</h2>
        <div class="price1">{format_price(st.session_state.results['prix_expert'])}</div>
    </div>
</div>
""", unsafe_allow_html=True)


    # Métriques
   
    
    # -------------------------------------------------
# 1. CSS (à mettre une seule fois dans votre app)
# -------------------------------------------------
    st.markdown("""
<style>
.metric-container {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin: 30px 0;
    width: 100%;
}
.metric-card {
    background: linear-gradient(135deg, #004D98 0%, #A50044 100%);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    flex: 1;                     /* chaque carte = 25% */
    min-width: 180px;
    border: 2px solid #FFD700;   /* bordure dorée */
}
.metric-card .label {
    margin: 0 0 8px 0;
    color: #FFD700;              /* titre en OR */
    font-size: 1.1em;
    font-weight: bold;
}
.metric-card .value {
    font-size: 1.8em;
    font-weight: bold;
    color: #FFD700;              /* valeur en OR */
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Mobile : 1 ou 2 colonnes */
@media (max-width: 900px) {
    .metric-container { flex-wrap: wrap; }
    .metric-card { flex: 1 1 45%; }
}
@media (max-width: 500px) {
    .metric-card { flex: 1 1 100%; }
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# 2. Calcul de l’amplitude (une seule fois)
# -------------------------------------------------
    range_width = st.session_state.results['p80'] - st.session_state.results['p20']


# -------------------------------------------------
# 3. Affichage des 4 cartes
# -------------------------------------------------
    st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="label">Prix Bas (P20)</div>
        <div class="value">{format_price(st.session_state.results['p20'])}</div>
    </div>
    <div class="metric-card">
        <div class="label">Prix Médian (P50)</div>
        <div class="value">{format_price(st.session_state.results['p50'])}</div>
    </div>
    <div class="metric-card">
        <div class="label">Prix Haut (P80)</div>
        <div class="value">{format_price(st.session_state.results['p80'])}</div>
    </div>
    <div class="metric-card">
        <div class="label">Amplitude</div>
        <div class="value">{format_price(range_width)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
















































    # Tabs pour les détails
    tab1, tab2, tab3 = st.tabs(["🤖 Prédictions ML", "🚗 Véhicules Comparables", "📊 Statistiques"])
    
    with tab1:
        st.markdown("### Prédictions par Modèle de Machine Learning")
        for model, price in st.session_state.results['predictions'].items():
            st.markdown(f"""
            <div class="model-prediction">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #004D98; font-size: 1.2rem;">{model}</span>
                    <span style="font-weight: bold; color: #A50044; font-size: 1.3rem;">{format_price(price)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Véhicules Similaires sur le Marché")
        for car in st.session_state.results['comparables']:
            st.markdown(f"""
            <div class="comparable-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #004D98; font-size: 1.1rem;">{car['Marque']} {car['Modele']}</strong><br>
                        <span style="color: #666;">📏 {car['Km']:,} km | 📅 {car['Mc']} | ⛽ {car['Carburant']}</span>
                    </div>
                    <span style="font-weight: bold; color: #A50044; font-size: 1.2rem;">{format_price(car['Prix'])}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Analyse Statistique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Informations du Véhicule")
            info_data = {
                "Kilométrage": f"{st.session_state.form_data['Km']:,} km",
                "Année": st.session_state.form_data['Mc'],
                "Puissance": f"{st.session_state.form_data['Cv']} CV",
                "Transmission": st.session_state.form_data['Transmission'],
                "Carburant": st.session_state.form_data['Carburant']
            }
            for key, value in info_data.items():
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            st.markdown("#### 🎯 Niveau de Confiance")
          
            prix_expert = st.session_state.results['prix_expert']
            prix_moyen_ml = st.session_state.results['average']

            if prix_expert and prix_moyen_ml and prix_expert > 0:
        # Écart relatif entre prix expert et moyenne ML
                 ecart_relatif = abs(prix_expert - prix_moyen_ml) / prix_moyen_ml
                 confiance = max(0, min(100, 100 * (1 - ecart_relatif)))  # 0 à 100%
            else:
                 confiance = 50  # Valeur par défaut si données manquantes
                 ecart_relatif = None

    # Barre de progression stylisée
    st.progress(confiance / 100)

    # Couleur et niveau de confiance
    if confiance >= 90:
        niveau = "Très Élevé"
        couleur = "#1DB954"  # Vert Spotify
        icone = "✅"
    elif confiance >= 75:
        niveau = "Élevé"
        couleur = "#34C759"  # Vert clair
        icone = "✅"
    elif confiance >= 60:
        niveau = "Bon"
        couleur = "#FFD60A"  # Jaune or
        icone = "👍"
    elif confiance >= 40:
        niveau = "Modéré"
        couleur = "#FF9F0A"  # Orange
        icone = "⚠️"
    else:
        niveau = "Faible"
        couleur = "#FF453A"  # Rouge Apple
        icone = "⚠️"

    # Affichage du score
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px;">
            <h3 style="color: {couleur}; margin: 0; font-size: 1.6rem;">
                {icone} {niveau}
            </h3>
            <p style="color: #666; margin: 5px 0; font-size: 1rem;">
                Score de confiance: <strong>{confiance:.0f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📄 Télécharger le Rapport", use_container_width=True):
            if st.session_state.results:
               pdf_data = generate_pdf_report(st.session_state.results, st.session_state.form_data)
            st.download_button(
                label="Télécharger PDF",
                data=pdf_data,
                file_name=f"rapport_{st.session_state.form_data['Marque']}_{st.session_state.form_data['Modele']}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.error("Aucun résultat à exporter.")
    
    with col2:
        if st.button("📧 Envoyer par Email", use_container_width=True):
            st.info("Fonctionnalité à venir...")
    
    with col3:
        if st.button("🔄 Nouvelle Estimation", use_container_width=True):
            st.session_state.step = 1
            st.session_state.form_data = {}
            st.session_state.results = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>🚗 Estimateur Prix Automobile</strong> - Propulsé par l'Intelligence Artificielle</p>
    <p>© 2025 - Tous droits réservés</p>
    <p style="margin-top: 0.5rem; font-style: italic;">Réalisé par Ghizlane Chichouki</p>
</div>
""", unsafe_allow_html=True)