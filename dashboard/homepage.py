# -*- coding: utf-8 -*- 
"""Application : Dashboard de Crédit Score

Auteur: Oubairouk Mbarek 
Source: https://github.com/loedata/P7-DASHBOARD
Local URL: http://localhost:8501
Lancement en local depuis une console anaconda prompt : 
    cd vers_repertoire du fichier dashboard_streamlit.py
    streamlit run dashboard_streamlit.py
Arrêt dans la console anaconda-prompt
"""

# ====================================================================
# Version : 0.0.1 -  Oubairouk Mbarek 03/01/2023
# ====================================================================

__version__ = '0.0.0'

# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st

#import utilitaire.load_env as env
from utilitaire.load_env import * 

st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")

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
    </head>             
    <h1 class="h1head"> Prêt à dépenser <br>
        <h2 class="h2head"> DASHBOARD</h2>
        <hr class ="hrhead" />
     </h1>
"""

st.markdown(html_header, unsafe_allow_html=True)

#st.markdown('<style>div[class="css-6qob1r e1fqkh3o3"] {color:black; font-weight: 900; ;background-repeat: no-repeat;background-size:350%;} </style>', unsafe_allow_html=True)

st.markdown("""
    <style>
    div[data-testid="stSidebarNav"] {
        border-radius: 0.5rem;
        border: 1px solid lightgrey;
        background-color: rgba(151, 250, 195, 0.15);
    }
    div[data-testid="stSidebarNav"] >{
            margin-right: 10px;
            background-color:green;
            margin: 3px;
        }

    </style>
    """, unsafe_allow_html=True)

	
pages = st.source_util.get_pages("homepage.py")
new_page_names = {
  'homepage': "🏡 Page d'acceuil",
  'score': '🥇 Score du client',
  'profil': '📊 profil du client',
  'plot': '📊 plot'
}

#st.markdown(pages.values())
for key, page in pages.items():
  if page['page_name'] in new_page_names:
    page['page_name'] = new_page_names[page['page_name']]
# Chargement du fichier css
# ====================================================================


local_css(FILE_CSS)
           
# ====================================================================
# HTML MARKDOWN
# ====================================================================


# Chargement des dataframes et du modèle

with st.spinner('Changement de données'):
    df_info_client, df_pret_client, df_info_voisins, df_pret_voisins, \
        df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
        application_test, shap_values, best_model = load_data()

# diffusion des données

for df in LIST_DF:
    st.session_state[df]=eval(df)
st.session_state['home'] = True