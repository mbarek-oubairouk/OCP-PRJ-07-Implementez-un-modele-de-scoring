# -*- coding: utf-8 -*- 
import re
import json
from configparser import ConfigParser, ExtendedInterpolation
import streamlit as st
import pickle
import boto3


# ====================================================================
# Chargement du fichier configuration
# ====================================================================

in_file = '../ressources/conf/dashboard.ini'
conf = ConfigParser(interpolation=ExtendedInterpolation())
conf.optionxform=str

conf.read(in_file, encoding='utf-8')
# charger d'abord default
exceptions_list = [v.split(',') for k, v in conf.items(
    'DEFAULT') if k == 'exceptions_list'][0]
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
                globals()[k] = v
 

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
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
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