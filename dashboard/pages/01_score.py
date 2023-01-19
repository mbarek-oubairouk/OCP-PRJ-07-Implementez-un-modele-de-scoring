# -*- coding: utf-8 -*- 

# ====================================================================
# Chargement des librairies
# ====================================================================

from re import I
import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
import boto3

from utilitaire.load_env import * 

# ====================================================================
# HEADER - TITRE
# ====================================================================

#st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)

AWS_S3_CREDS = {
		"aws_access_key_id":st.secrets["AWS_ACCESS_ID"], 
		"aws_secret_access_key":st.secrets["AWS_SECRET_ACCESS_KEY "]
	}
# ====================================================================
# Chargement du fichier css
# ====================================================================
local_css(FILE_CSS)
#list_history=[]

def update_pred(list_histo):
    ex=f"""
    <button type="button" class="btn btn-primary">
     <span class="badge badge-light">{len(list_histo)}</span> prédictions 
     </button> """
    return ex



menu_style()
ga, gd = st.columns([1,1])
#global list_history
with ga:
    header_style()

def update_resum():
    if len(list_history) > 1  :
       global gd
       #gd=gd.empty()
    with gd.container():
        ex=update_pred(list_history)
        st.markdown(ex, unsafe_allow_html=True)
        with st.expander("👉",expanded=False):
            st.dataframe(pd.DataFrame(list_history,columns=['id','status','score','date']))

if 'histo' in st.session_state:
    list_history = st.session_state['histo']
    gd=gd.empty()
    update_resum()
    
else:
    list_history = []
    
    





for df in LIST_DF:
    if df in st.session_state:
        globals()[df] = st.session_state[df]
# ====================================================================
# CHOIX DU CLIENT
# ====================================================================

html_select_client="""
         <h3 class="titrecli" > Informations sur le client / demande de prêt</h3>
    """

	
if 'page-home'  not in st.session_state:
    st.error("Vous devez d'abord cliquer sur le menu Page d'acceuil")
    st.stop()


global modele_used 
modele_used = ""
#with st.container():
auth=False

def check_auth():
    if check_password():
           auth = True
    else :
           auth = False

    return auth


login, aide = st.columns([3,2])
with aide:
    with st.spinner('**Selection du modéle...**'):                 
            with st.expander('Aide',expanded=False):
                      st.markdown("""
                                   <div class="card border-info  mb-3" >
                                       <div class="card-header">Explications</div>
                                       <div class="card-body text-info">
                                              <h5 class="card-title">Le choix du modéles</h5>
                                              <p class="card-text"><p>
                                              <ul><li><b>premise</b> : pour utiliser le modéle déployé localement dans Streamlit, pas besoin du mot de passe</li>
                                                  <li><b>cloud aws</b> : pour utiliser le modéle déployé dans aws sagemaker</li></ul>
                                   </div>"""
                        ,unsafe_allow_html=True)
with login:
    modele_used = st.radio("Inférence en : 👇",("premise","cloud aws 🔐"),
                         horizontal=True,help="mot de passe nécessaire pour cloud aws",key="auth",index=0)
if "aws" in modele_used:
    auth = check_password() 
else:
    auth = True


def formcli():
    st.markdown(html_select_client, unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([1, 3])
        client_id = col1.selectbox('Sélectionnez un **ID Client**: :point_down:',
                                   [" "]+df_info_voisins['ID_CLIENT'].unique().tolist())
        # Infos principales client
        if re.search('[0-9]+', str(client_id)):
            client_info = df_info_client.query('SK_ID_CURR == @client_id')
            client_info.set_index('SK_ID_CURR', inplace=True)
            st.session_state['client_info'] = client_info
        # Infos principales sur la demande de prêt
            client_pret = df_pret_client.query('SK_ID_CURR == @client_id')
            client_pret.set_index('SK_ID_CURR', inplace=True)
            st.session_state['client_pret'] = client_pret
            with col1:
                st.markdown('<i class="fa fa-triangle"></i>', unsafe_allow_html=True)

            with col2:
                 st.table(client_info)
                 st.table(client_pret)
        else :
            client_id=-1
    return client_id



if auth:
    client_id = formcli()
    if client_id == -1 :
        st.stop()
        
    st.session_state['client_id'] = client_id
    # mettre à jour la liste des fichiers diffusées
    st.session_state['list_df'] = client_id
    st.session_state['list_df'] = LIST_DF+['client_info', 'client_pret']
    # ====================================================================
    # SCORE - PREDICTIONS
    # ====================================================================

    html_score = """<h3 class="titrescore" > Crédit Score</h3>"""
    # Préparation des données à afficher dans la jauge ==============================================
    # ============== Score du client en pourcentage ==> en utilisant le modèle ======================
    # Sélection des variables du clients
    X_test = test_set.query('SK_ID_CURR == @client_id')
    # Score des prédictions de probabiltés
    data_predict = X_test.drop('SK_ID_CURR', axis=1)
    ############################ Via amazone sagemaker################################################
    if 'aws' in modele_used:
        # status du service
        sage_client = boto3.client('sagemaker', region_name=region_name,**AWS_S3_CREDS)
        endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
        endpoint_status = endpoint_description["EndpointStatus"]
        st.markdown(endpoint_status)
        status_srv = check_status(app_name,region_name)

        if 'InService' in status_srv:
            encoded_tabular_data = data_predict.to_csv(index=False).encode("utf-8")
            pred_sagemaker = query_endpoint(app_name, region=region_name,
                                        encoded_tabular_data=encoded_tabular_data,
                                        target_version=target_version)

            # Score du client en pourcentage arrondi et nombre entier
            [[sain, defaut]] = pred_sagemaker['predictions']
            y_proba_sain_clienn, y_proba_defaut_client = (
                np.rint(sain * 100), np.rint(defaut * 100))
        else:
            st.error(f"⚠️ aws : **{status_srv.get('EndpointStatus')}**, utiliser l'option **premise**")
            #y_proba = best_model.predict_proba(data_predict)[:, 1]
             # Score du client en pourcentage arrondi et nombre entier
            #y_proba_defaut_client = int(np.rint(y_proba * 100))
            st.stop()
    else :
        # predication en local
        y_proba = best_model.predict_proba(data_predict)[:, 1]
        # Score du client en pourcentage arrondi et nombre entier
        y_proba_defaut_client = int(np.rint(y_proba * 100))

    # ============== Score moyen des 10 plus proches voisins du test set en pourcentage =============

    # Score moyen des 10 plus proches voisins du test set en pourcentage
    score_moy_voisins_test = int(np.rint(df_dashboard.query(
        'SK_ID_CURR == @client_id')['SCORE_10_VOISINS_MEAN_TEST'] * 100))

    # ============== Pourcentage de clients voisins défaillants dans l'historique des clients =======
    pourc_def_voisins_train = int(np.rint(df_dashboard.query(
        'SK_ID_CURR == @client_id')['%_NB_10_VOISINS_DEFAILLANT_TRAIN']))

    # ============== Pourcentage de clients voisins défaillants prédits parmi les nouveaux clients ==
    pourc_def_voisins_test = int(np.rint(df_dashboard.query(
        'SK_ID_CURR == @client_id')['%_NB_10_VOISINS_DEFAILLANT_TEST']))

    
  
    #st.markdown(html_score, unsafe_allow_html=True)
    with st.container():
        # JAUGE + récapitulatif du score moyen des voisins
        col1, _, col2 = st.columns([1,0.5,2])
        with col1:
            #st.plotly_chart(ma_gauge)
            fig_g, clr_score = getgauge(y_proba_defaut_client,score_moy_voisins_test,client_id)
            st.plotly_chart(fig_g,theme=None,use_container_width=True)
        with col2:
            st.write("")
            st.write("")
            st.write("")
            # Texte d'accompagnement de la jauge
            if 0 <= y_proba_defaut_client < 25:
                score_text = 'EXCELLENT'
                #st.success(score_text)
            elif 25 <= y_proba_defaut_client < 50:
                score_text = 'BON'
                #st.success(score_text)
            elif 50 <= y_proba_defaut_client < 75:
                score_text = 'MAUVAIS'
                #st.warning(score_text)
            else:
                score_text = 'DANGER'
                #st.error(score_text)
            #st.write("")
            st.markdown(f"""
            <div class="card border-info  mb-3" >
                  <div class="card-header" style="background-color:{clr_score}">Crédit score : {score_text}</div>
                  <div class="card-body text-info">
                      <h5 class="card-title">N° client :{client_id}</h5>
                      <p>Crédit score moyen des 10 clients similaires : <span class="badge badge-info">{score_moy_voisins_test}</span></p>
                      <p><span class="badge badge-info">{pourc_def_voisins_train}%</span> de clients voisins réellement défaillants dans l'historique</p>
                      <p><span class="badge badge-info">{pourc_def_voisins_test}%</span> de clients voisins défaillants prédits pour les nouveaux clients</p>
            </div>""", unsafe_allow_html=True)
    list_history.append([client_id,score_text,y_proba_defaut_client,dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    st.session_state['histo'] = list_history

    gd=gd.empty()
    update_resum()
    


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
    # utile pour la page profil
    st.session_state['page-score'] = True
    # --------------------------------------------------------------------
    # PLUS INFORMATIONS
    # --------------------------------------------------------------------

    def all_infos_clients():
        ''' Affiche toutes les informations sur le client courant
        '''
        html_all_infos_clients = """ <h3 class="titrecli" > Plus infos client</h3>"""
        # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES ===========================
        check_cli_info = st.sidebar.checkbox("Voir toutes infos clients ?")
        if check_cli_info:
            st.markdown(html_all_infos_clients, unsafe_allow_html=True)
            with st.spinner('**Affiche toutes les informations sur le client courant...**'):
                with st.expander(f'Toutes les informations du client courant',
                                 expanded=True):
                    st.dataframe(df_client_origin)
                    st.dataframe(df_client_test)
                    
        return check_cli_info

    st.sidebar.subheader('Plus infos')
    check_cli_info = all_infos_clients()

