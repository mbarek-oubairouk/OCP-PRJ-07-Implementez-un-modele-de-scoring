# -*- coding: utf-8 -*- 
"""Application : Dashboard de Crédit Score

Auteur: Oubairouk Mbarek 
Local URL: http://localhost:8501
Lancement en local depuis une console anaconda prompt : 
    cd vers_repertoire du fichier dashboard_streamlit.py
    streamlit run homepage.py
Arrêt dans la console anaconda-prompt
"""

# ====================================================================
# Version : 0.0.1 -  Oubairouk Mbarek 03/01/2024
# ====================================================================

__version__ = '0.0.0'

# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st

#import utilitaire.load_env as env
from utilitaire.load_env import * 

st.set_page_config(page_title="Prêt à dépenser - Dashboard: conçu pour les chargés de relations clients.", 
page_icon="💰", layout="wide")

update_menu()
menu_style()
header_style()
#st.markdown('<style>div[class="css-6qob1r e1fqkh3o3"] {color:black; font-weight: 900; ;background-repeat: no-repeat;background-size:350%;} </style>', unsafe_allow_html=True)


# Description du projet

# espace entre les tabs
st.markdown("""
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] >p{
  font-size: 18px;
  background-color:  rgba(245, 174, 79, 0.15); /*Green #4CAF50; */
  border: none;
  color: #F56C4F;
  padding: 20px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  margin: 4px 2px;
  cursor: pointer;
  border-radius: 20px;
}
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] {
        /*border: 1px solid lightgrey;
        background-color: #F9DC44;*/
        border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['🥇 Score du client','👨‍👩‍👧‍👦 Profil du client','🤔 Interprétation du modèle'])
with tab1:
   st.markdown("""
            <span style="font-size: 25px;">👈</span> En cliquant sur le menu à droite **Score du client**, vous pourrez découvrir le score estimé que le client a obtenu pour sa demande de prêt.<br>

Vous y trouverez aussi une explication intelligible de la manière dont il a été calculé.<br>
Ce score représente la probabité de défaut du client pour rembourser son crédit.<br>

Ce score est calculé à l'aide d'un modèle de prédiction appliqué à un ensemble de **307 511 clients** dont on connaît déjà la probablité de défaut.<br>
Cette procédure permet de confronter les résultats du modèle de prédiction à la réalité et donc de **valider l'efficacité du modèle de prédiction**.
            """, unsafe_allow_html=True)

with tab2:
   st.markdown("""
<span style="font-size: 25px;">👈</span> Sur la page **Profil du client**, vous trouverez une comparaison des informations descriptives de votre client à un groupe de clients similaires.
            """, unsafe_allow_html=True)

with tab3:
   st.markdown("""
           <span style="font-size: 25px;">👈</span> Enfin, sur la page **Interprétation du modèle**, vous trouverez une interprétation globale du modèle de prédiction.
            """, unsafe_allow_html=True)



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
st.session_state['page-home'] = True

