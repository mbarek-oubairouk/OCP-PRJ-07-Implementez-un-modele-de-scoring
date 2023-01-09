# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
from PIL import Image
import pickle
import re
import json
from configparser import ConfigParser, ExtendedInterpolation



#st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
#st.title('Simuleur de prêt')

# ====================================================================
# Chargement du fichier configuration
# ====================================================================

in_file = '../ressources/conf/dashboard.ini'
conf = ConfigParser(interpolation=ExtendedInterpolation())

conf.read(in_file,encoding='utf-8')

file_css = conf['css']['FILE_CSS']


#descrpition = json.loads(conf['features']['descrpition'])
# ====================================================================
# Chargement du fichier css
# ====================================================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



# ====================================================================
# VARIABLES STATIQUES
# ====================================================================
# Répertoire de sauvegarde du meilleur modèle
FILE_BEST_MODELE = conf['model']['FILE_BEST_MODELE']
# Répertoire de sauvegarde des dataframes nécessaires au dashboard
# Test set brut original
FILE_APPLICATION_TEST = conf['data']['FILE_APPLICATION_TEST']
# Test set pré-procédé
FILE_TEST_SET = conf['data']['FILE_TEST_SET']
# Dashboard
FILE_DASHBOARD = conf['data']['FILE_DASHBOARD']
# Client
FILE_CLIENT_INFO = conf['data']['FILE_CLIENT_INFO']
FILE_CLIENT_PRET = conf['data']['FILE_CLIENT_PRET']
# 10 plus proches voisins du train set
FILE_VOISINS_INFO = conf['data']['FILE_VOISINS_INFO']
FILE_VOISIN_PRET = conf['data']['FILE_VOISIN_PRET']
FILE_VOISIN_AGG = conf['data']['FILE_VOISIN_AGG']
FILE_ALL_TRAIN_AGG = conf['data']['FILE_ALL_TRAIN_AGG']
# Shap values
FILE_SHAP_VALUES = conf['data']['FILE_SHAP_VALUES']


# ====================================================================
# VARIABLES GLOBALES
# ====================================================================
LIST_DF = re.sub("\n|\s+","",conf['dataframe']['LIST_DF']).split(",")
grp_cols_1 =  re.sub("\n|\s+","",conf['features']['grp_cols_1']).split(",")



grp_cols_2 = re.sub("\n|\s+","",conf['features']['grp_cols_2']).split(",")

grp_cols_3 = re.sub("\n|\s+","",conf['features']['grp_cols_3']).split(",")

grp_cols_4 = re.sub("\n|\s+","",conf['features']['grp_cols_4']).split(",")

descrpition = json.loads(conf['features']['champs'])
features_vars = list(descrpition.keys())

# Legende
#<div style='border:1px dashed;width:auto'>
html_legende = """

 <div class="cadrelegende">
<table id="cap">
  <caption >Légende:</caption>
  <tr>
    <td>
      <p class='legende' style="background: Orange;"></p>
    </td>
    <td >Client courant</td>
  </tr>
  <tr>
    <td>
      <p class='legende' style="background: SteelBlue;"></p>
    </td>
    <td >Moyenne des Clients voisins</td>
  </tr>
  <tr>
    <td>
      <p class='legende' style="background: Green;"></p>
    </td>
    <td>Moyenne des Clients non-défaillants</td>
  </tr>
  <tr>
    <td>
      <p class='legende' style="background: Crimson;"></p>
    </td>
    <td>Moyenne des Clients défaillants</td>
  </tr>
</table>
</div>
"""


# ====================================================================
# IMAGES
# ====================================================================
FILE_IMAGES = conf['images']['FILE_IMAGES']
# Logo de l'entreprise
logo =  Image.open(f"{FILE_IMAGES}/logo.png") 
# Légende des courbes
lineplot_legende =  Image.open(f"{FILE_IMAGES}/lineplot_legende.png") 

# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================


# Chargement du modèle et des différents dataframes
# Optimisation en conservant les données non modifiées en cache mémoire
#@st.cache(persist = True)
def load():
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
         
    return df_info_client, df_pret_client, df_info_voisins, df_pret_voisins, \
        df_dashboard, df_voisin_train_agg, df_all_train_agg, test_set, \
            application_test, shap_values, best_model
