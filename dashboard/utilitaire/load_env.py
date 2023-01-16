# -*- coding: utf-8 -*- 
import re
import json
import os
from configparser import ConfigParser, ExtendedInterpolation
import streamlit as st
import pickle
import boto3
import plotly.graph_objects as go
import numpy as np
import shap

# ====================================================================
# Chargement du fichier configuration
# ====================================================================
DIRPRJ = st.secrets["DIRPRJ"]
in_file = f'{DIRPRJ}/ressources/conf/dashboard.ini'
AWS_ACCESS_ID = st.secrets["AWS_ACCESS_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_id = st.secrets["aws_id"]
conf = ConfigParser(interpolation=ExtendedInterpolation())
conf.optionxform=str

conf.read(in_file, encoding='utf-8')
# charger d'abord default
exceptions_list = [v.split(',') for k, v in conf.items('DEFAULT') if k == 'exceptions_list'][0]
exceptions_dic = [v.split(',') for k, v in conf.items(
    'DEFAULT') if k == 'exceptions_dic'][0]

for sec in conf.sections():
    for k, v in conf.items(sec):
        if k not in ['nslash', 'racine', 'prefix']:
            if k in exceptions_list:
                globals()[k] = re.sub("\n|\s+", "", v).split(",")
            elif k in exceptions_dic:
                globals()[k] = json.loads(v)
            else:
               # print(k)
                globals()[k] = v.format(DIRPRJ=DIRPRJ,aws_id=aws_id)
 

# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================


# Chargement du modèle et des différents dataframes
# Optimisation en conservant les données non modifiées en cache mémoire
@st.cache(persist = True,allow_output_mutation=True)
def load_data():
    with st.spinner('Import des données'):
        
        # Import du dataframe des informations des traits stricts du client
        with open(FILE_CLIENT_INFO, 'rb') as df_info_client:
            df_info_client = pickle.load(df_info_client)
            
        # Import du dataframe des informations sur le prêt du client
        with open(FILE_CLIENT_PRET, 'rb') as df_pret_client:
            df_pret_client = pickle.load(df_pret_client)
            
        # Import du dataframe des informations des traits stricts des voisins
        with open(FILE_VOISINS_INFO, 'rb') as df_info_voisins:
            df_info_voisins = pickle.load(df_info_voisins)
            
        # Import du dataframe des informations sur le prêt des voisins
        with open(FILE_VOISIN_PRET, 'rb') as df_pret_voisins:
            df_pret_voisins = pickle.load(df_pret_voisins)

        # Import du dataframe des informations sur le dashboard
        with open(FILE_DASHBOARD, 'rb') as df_dashboard:
            df_dashboard = pickle.load(df_dashboard)

        # Import du dataframe des informations sur les voisins aggrégés
        with open(FILE_VOISIN_AGG, 'rb') as df_voisin_train_agg:
            df_voisin_train_agg = pickle.load(df_voisin_train_agg)

        # Import du dataframe des informations sur les voisins aggrégés
        with open(FILE_ALL_TRAIN_AGG, 'rb') as df_all_train_agg:
            df_all_train_agg = pickle.load(df_all_train_agg)

        # Import du dataframe du test set nettoyé et pré-procédé
        with open(FILE_TEST_SET, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)

        # Import du dataframe du test set brut original
        with open(FILE_APPLICATION_TEST, 'rb') as df_application_test:
            application_test = pickle.load(df_application_test)

        # Import du dataframe du test set brut original
        with open(FILE_SHAP_VALUES, 'rb') as shap_values:
            shap_values = pickle.load(shap_values)
            
    # Import du meilleur modèle lgbm entrainé
    with st.spinner('Import du modèle'):
        
        # Import du meilleur modèle lgbm entrainé
        with open(FILE_BEST_MODELE, 'rb') as model_lgbm:
            best_model = pickle.load(model_lgbm)
    # SHAP values
    """
    with st.spinner('Lancement SHAP values'):
   
        # Test set sans l'identifiant
        X_test = test_set.set_index('SK_ID_CURR')
        # Entraînement de shap sur le test set
        test_explainer = shap.Explainer(best_model, X_test)
        test_shap_values = test_explainer(X_test, check_additivity=False)
    """     
    return df_info_client, df_pret_client, df_info_voisins, df_pret_voisins, \
        df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
            application_test, shap_values, best_model

# ====================================================================
# Chargement du fichier css
# ====================================================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# ====================================================================
# fonctions pour Sagemaker 
# ====================================================================

def check_status(app_name,region):
    sage_client = boto3.client('sagemaker', region_name=region)
    try :
       endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
       endpoint_status = endpoint_description["EndpointStatus"]
    except :
       endpoint_status = {'EndpointStatus':'point de terminaison injoignable'}
    return endpoint_status


def query_endpoint(app_name,region, encoded_tabular_data,target_version=target_version):
    client = boto3.session.Session().client("sagemaker-runtime", region)
    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=encoded_tabular_data,
        ContentType='text/csv', # important!
        TargetVariant=target_version
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    print("response: {}".format(preds))
    return preds


def check_password():
    """Returns `True` if the user had the correct password."""
    invite = "Saisissez le mot de passe et appuyer sur entrer :leftwards_arrow_with_hook:"
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
    col, _, _ = st.columns([1,1,1])
    if "password_correct" not in st.session_state:
        # First run, show input for password.
        col.text_input(
            invite, type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        col.text_input(
            invite, type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Oubairouk Mbarek">
        <meta name="viewport" content="width=device-width, initial-scale=1">
               <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
   
    </head>             
    <h1 class="h1head"> Prêt à dépenser <br>
        <h2 class="h2head"> Dashboard Scoring Credit</h2>
        <hr class ="hrhead" />
     </h1>
"""

cadre_menu_agauche = """
    <style>
    div[data-testid="stSidebarNav"] {
        border-radius: 0.5rem;
        border: 1px solid lightgrey;
        background-color: rgba(151, 250, 195, 0.15);
    }
    div[data-testid="stSidebarNav"] >{
            margin-right: 10px;
            background-color:green;
            margin: 10px;
        }

    </style>
    """

def header_style():

    st.markdown(html_header, unsafe_allow_html=True)

def menu_style():
    
    st.markdown(cadre_menu_agauche, unsafe_allow_html=True)

def update_menu():
    pages = st.source_util.get_pages("homepage.py")
    new_page_names = {
      'homepage': "🏦 Page d'acceuil",
      'score': '🥇 Score du client',
      'profil': '👨‍👩‍👧‍👦 Profil du client',
      'interpretation': '🤔 Interprétation'
      }

    #st.markdown(pages.values())
    for key, page in pages.items():
      if page['page_name'] in new_page_names:
         page['page_name'] = new_page_names[page['page_name']]
  



def getgauge(value=0, refval=0,id_client=0):
    plot_bgcolor = "white"
    d = {1:"#2bad4e",2:"#85e043",3:"#orange",4:"#f25829"}
    quadrant_colors = [plot_bgcolor, "#f25829", "orange", "#85e043", "#2bad4e"]
    quadrant_text = ["", "<b>Très mauvais</b>", "<b>Mauvais</b>","<b>Bon</b>", "<b>Excellent</b>"]
    n_quadrants = len(quadrant_colors) - 1
    current_value = value
    min_value = 0
    max_value = 100
    hand_length = np.sqrt(2) / 5
    hand_angle = np.pi * (1 - (max(min_value, min(max_value,
                                                  current_value)) - min_value) / (max_value - min_value))

    clr_score = d.get(int(np.ceil(current_value/25)), '#333')
    delta = current_value - refval

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) /
                                2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",

            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, t=30, l=10, r=10),
            width=300,
            height=300,

            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    #text=f'<b style="color:{clr_score};font-size:15px">Score:<i style="font-size:25px">{current_value} </i></b>',
                    text=f'<b style="color:#4F9CF5;font-size:25px">{current_value}</b>',
                    x=0.52, xanchor="center", xref="paper",
                    y=0.36, yanchor="bottom", yref="paper",
                    showarrow=False,
                ),
                go.layout.Annotation(
                    text= f'<span style="color:red;font-size:15px">▲{delta}</span>' if delta>0 else f'<span style="color:green;font-size:15px">▼{delta}</span>',
                    x=0.50, xanchor="center", xref="paper",
                    y=0.32, yanchor="bottom", yref="paper",
                    showarrow=False,
                ),

            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor='#333',
                    line_color='#333',
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color='#333', width=4)
                )
            ]
        )
    )

    fig.update_layout(
        title={
            'text': f'Probabilité de défaut client n°:<span style="color:green">{id_client}</span>',
            'x': 0.5,
            'xanchor': 'center'
        })
    return fig,clr_score
