# -*- coding: utf-8 -*- 

# ====================================================================
# Chargement des librairies
# ====================================================================

from re import I
import streamlit as st
import numpy as np
# import pandas as pd
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap

from utilitaire.load_env import * 

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
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)


# ====================================================================
# Chargement du fichier css
# ====================================================================
local_css(FILE_CSS)

 

for df in LIST_DF:
    if df in st.session_state:
        globals()[df] = st.session_state[df]
# ====================================================================
# CHOIX DU CLIENT
# ====================================================================

html_select_client="""
         <h3 class="titrecli" > Informations sur le client / demande de prêt</h3>
    """

st.markdown(html_select_client, unsafe_allow_html=True)

def formcli():
    #st.header("**ID Client**")
    #form = st.form(key='my-form')
    with st.container():
        col1, col2 = st.columns([1, 3])
        client_id = col1.selectbox('Sélectionnez un client : :point_down:',
                                   df_info_voisins['ID_CLIENT'].unique())
        # Infos principales client
        client_info = df_info_client.query('SK_ID_CURR == @client_id')
        client_info.set_index('SK_ID_CURR', inplace=True)
        st.session_state['client_info'] = client_info
        #st.table(client_info)
            # Infos principales sur la demande de prêt
            # st.write("*Demande de prêt*")
        client_pret = df_pret_client.query('SK_ID_CURR == @client_id')
        client_pret.set_index('SK_ID_CURR', inplace=True)
        st.session_state['client_pret'] = client_pret
            #st.table(client_pret)
        # Every form must have a submit button.
        #submitted = st.form_submit_button("submit",type='primary')
        
        with col1:
                 st.write("")
                 col1.header("**ID Client**")
        with col2:
                 st.table(client_info)
            #with col2:
                 st.table(client_pret)
    return client_id



#if not check_cli_info:
client_id = formcli()

st.session_state['client_id'] = client_id
# mettre à jour la liste des fichiers diffusées

st.session_state['list_df'] = client_id
st.session_state['list_df'] = LIST_DF+['client_info','client_pret' ]
# ====================================================================
# SCORE - PREDICTIONS
# ====================================================================

html_score ="""
         <h3 class="titrescore" > Crédit Score</h3>
    """

# Préparation des données à afficher dans la jauge ==============================================

# ============== Score du client en pourcentage ==> en utilisant le modèle ======================
# Sélection des variables du clients
X_test = test_set.query('SK_ID_CURR == @client_id')
# Score des prédictions de probabiltés
data_predict = X_test.drop('SK_ID_CURR', axis=1)

############################Via amazone sagemaker################################################
if model_used == "sagemaker" :
    encoded_tabular_data = data_predict.to_csv(index=False).encode("utf-8")
    """
    pred_sagemaker=query_endpoint(app_name,region=region_name,
                                   encoded_tabular_data=encoded_tabular_data,
                                   target_version=target_version)
    """
    # Score du client en pourcentage arrondi et nombre entier
    [[sain,defaut]] = pred_sagemaker['predictions']
    y_proba_sain_clienn,y_proba_defaut_client = (np.rint(sain * 100),np.rint(defaut * 100)) 
    st.markdown(f'via sagemaker={y_proba_defaut_client}')
else:
    # predication en local
    y_proba = best_model.predict_proba(data_predict)[:, 1]
    # Score du client en pourcentage arrondi et nombre entier
    y_proba_defaut_client = int(np.rint(y_proba * 100))
st.markdown(f'model_used={model_used}')
# ============== Score moyen des 10 plus proches voisins du test set en pourcentage =============

# Score moyen des 10 plus proches voisins du test set en pourcentage
score_moy_voisins_test = int(np.rint(df_dashboard.query('SK_ID_CURR == @client_id')['SCORE_10_VOISINS_MEAN_TEST'] * 100))

# ============== Pourcentage de clients voisins défaillants dans l'historique des clients =======
pourc_def_voisins_train = int(np.rint(df_dashboard.query('SK_ID_CURR == @client_id')['%_NB_10_VOISINS_DEFAILLANT_TRAIN']))

# ============== Pourcentage de clients voisins défaillants prédits parmi les nouveaux clients ==
pourc_def_voisins_test = int(np.rint(df_dashboard.query('SK_ID_CURR == @client_id')['%_NB_10_VOISINS_DEFAILLANT_TEST']))


ma_gauge = go.Figure(go.Indicator(
    mode = 'gauge+number+delta',
    # Score du client en % df_dashboard['SCORE_CLIENT_%']
    value = y_proba_defaut_client,  
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': 'Probabilité de défaut client', 'font': {'size': 24}},
    # Score des 10 voisins test set
    # df_dashboard['SCORE_10_VOISINS_MEAN_TEST']
    delta = {'reference':score_moy_voisins_test ,
             'increasing': {'color': 'Crimson'},
             'decreasing': {'color': 'Green'}},
    gauge = {'axis': {'range': [None, 100],
                      'tickwidth': 3,
                      'tickcolor': 'darkblue'},
             'bar': {'color': 'white', 'thickness' : 0.20,},
             'bgcolor': 'white',
             'borderwidth': 2,
             'bordercolor': 'gray',
             'steps': [{'range': [0, 25], 'color': 'Green'},
                       {'range': [25, 49.49], 'color': 'LimeGreen'},
                       {'range': [49.5, 50.5], 'color': 'red'},
                       {'range': [50.51, 75], 'color': 'Orange'},
                       {'range': [75, 100], 'color': 'Crimson'}],
             'threshold': {'line': {'color': 'white', 'width': 10},
                           'thickness': 0.8,
                           # Score du client en %
                           # df_dashboard['SCORE_CLIENT_%']
                           'value': y_proba_defaut_client}}))

ma_gauge.update_layout(paper_bgcolor='white',
                        height=250, width=300,
                        font={'color': 'darkblue', 'family': 'Arial'},
                        margin=dict(l=0, r=0, b=0, t=0, pad=0))

#if check_cli_info:
st.markdown(html_score, unsafe_allow_html=True)
with st.container():
    # JAUGE + récapitulatif du score moyen des voisins
    col1, col2,_ = st.columns([1.1, 2,0.5])
    with col1:
        st.plotly_chart(ma_gauge)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        # Texte d'accompagnement de la jauge
        if 0 <= y_proba_defaut_client < 25:
            score_text = 'Crédit score : EXCELLENT'
            st.success(score_text)
        elif 25 <= y_proba_defaut_client < 50:
            score_text = 'Crédit score : BON'
            st.success(score_text)
        elif 50 <= y_proba_defaut_client < 75:
            score_text = 'Crédit score : MOYEN'
            st.warning(score_text)
        else:
            score_text = 'Crédit score : BAS'
            st.error(score_text)
        st.write("")

        st.markdown(
            f'Crédit score moyen des 10 clients similaires : **<span class="badge badge-info">{score_moy_voisins_test}</span>**'
            , unsafe_allow_html=True)
        st.markdown(
            f'**<span class="badge badge-info">{pourc_def_voisins_train}%</span>** de clients voisins réellement défaillants dans l\'historique'
            , unsafe_allow_html=True)
        st.markdown(
            f'**<span class="badge badge-info">{pourc_def_voisins_test}%</span>** de clients voisins défaillants prédits pour les nouveaux clients'
            , unsafe_allow_html=True)


# ====================================================================
# SIDEBAR
# ====================================================================

# Toutes Les informations non modifiées du client courant
df_client_origin = application_test.query('SK_ID_CURR == @client_id')

# Toutes Les informations modifiées du client courant
df_client_test = test_set.query('SK_ID_CURR == @client_id')

# Toutes les informations du client courant
df_client_courant = df_dashboard.query('SK_ID_CURR == @client_id')

st.session_state['df_client_courant'] = df_client_courant
st.session_state['score'] = y_proba_defaut_client
st.session_state['score_10test'] = score_moy_voisins_test

# --------------------------------------------------------------------
# PLUS INFORMATIONS
# --------------------------------------------------------------------
def all_infos_clients():
    ''' Affiche toutes les informations sur le client courant
    '''
    html_all_infos_clients="""  
                         <h3 class="titrecli" > Plus infos client</h3>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    check_cli_info = st.sidebar.checkbox("Voir toutes infos clients ?")
    if  check_cli_info :     
        st.markdown(html_all_infos_clients, unsafe_allow_html=True)
        with st.spinner('**Affiche toutes les informations sur le client courant...**'):
            with st.expander(f'Toutes les informations du client courant',
                             expanded=True):
                st.dataframe(df_client_origin)
                st.dataframe(df_client_test)
    return check_cli_info

st.sidebar.subheader('Plus infos')
check_cli_info = all_infos_clients()

