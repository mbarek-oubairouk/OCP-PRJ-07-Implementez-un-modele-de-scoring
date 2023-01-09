# -*- coding: utf-8 -*- 

# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
import lightgbm as lgb
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
# Warnings
import warnings
warnings.filterwarnings('ignore')


#st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
#st.markdown(html_legende)
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)

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

# ====================================================================
# Chargement du fichier css
# ====================================================================
local_css(FILE_CSS)



needed_vars = ['client_id','score','best_model','test_set','shap_values']


for df in needed_vars:
    if df in st.session_state:
        #st.markdown(f"{df}:reloand ok", unsafe_allow_html=True)
        globals()[df] = st.session_state[df]
		
client_index = test_set[test_set['SK_ID_CURR'] == client_id].index.item()
df_client_couant = test_set[test_set['SK_ID_CURR'] == client_id]
# --------------------------------------------------------------------
# FACTEURS D'INFLUENCE : SHAP VALUE
# --------------------------------------------------------------------
    
def affiche_facteurs_influence():
    ''' Affiche les facteurs d'influence du client courant
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Variables importantes
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Voir facteurs d\'influence"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Affiche les facteurs d\'influence du client courant...**'):                 
            with st.expander('Explication',expanded=False):
                      st.markdown("""
<div class="card border-info  mb-3" >
  <div class="card-header">Définition</div>
  <div class="card-body text-info">
    <h5 class="card-title">L’importance des variables</h5>
    <p class="card-text"><p>L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap. Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction. Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte la prédiction de manière positive ou négative.</p>
                      <p>Pour résumer, les valeurs de Shapley calculent l’importance d’une variable en comparant ce qu’un modèle prédit avec et sans cette variable. Cependant, étant donné que l’ordre dans lequel un modèle voit les variables peut affecter ses prédictions, cela se fait dans tous les ordres possibles, afin que les fonctionnalités soient comparées équitablement. Cette approche est inspirée de la théorie des jeux.<p>
  </div>
   <div class="card border-info  mb-3" >
  <div class="card-header">Le diagramme d'importance des variables</div>
  <div class="card-body text-info">
    <h5 class="card-title">shap.plots.bar</h5>
                          <p> <b>Le diagramme à barres</b> répertorie les variables les plus significatives par ordre décroissant. Les variables en haut contribuent davantage au modèle que celles en bas et ont donc un pouvoir prédictif élevé.<p></p>
  </div>
</div>
<div class="card border-info  mb-3" >
  <div class="card-header">Le diagramme de décision</div>
  <div class="card-body text-info">
    <h5 class="card-title">shap.decision_plot</h5>
    <p><b>Le diagramme de décision</b> montre le chemin de décision suivi en appliquant les valeurs de shap des entités individuelles une par une à la valeur attendue afin de générer la valeur prédite sous forme de graphique linéaire.<p>
  </div>
</div>
"""
                        ,unsafe_allow_html=True)           
            with st.expander(f"Facteurs d'influence du client courant:{client_id}",
                              expanded=True):

                
                explainer = shap.TreeExplainer(best_model)
                
                X_shap = test_set.set_index('SK_ID_CURR')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2= st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    
                    plt.title("Interprétation Gobale:\nDiagramme d'importance des variables",size=20,backgroundcolor='green',color='white')
                    # BarPlot du client courant
                    shap.plots.bar( shap_values[client_index], max_display=40)
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()
                    plt.title(f"Interprétation Locale:\nDiagramme de décision classe positive({explainer.expected_value[1]:.2f})",size=20,backgroundcolor='green',color='white')
                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1],
                                    X_test_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)

					
					
 
       
def affiche_arbre():
    ''' Affiche l'arbre du décison du molèle 
    '''
    if st.sidebar.checkbox("Arbre de décision"):
        html_arbre ="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Transparence
                  </h3>
            </div>
        </div>
        """
        
        st.markdown(html_arbre, unsafe_allow_html=True)
        X_grpah = test_set.set_index('SK_ID_CURR')
        X_test_courant = X_grpah.iloc[client_index]
        with st.spinner('**Transparence du modèle:**'):                    
            with st.expander(f"**Arbre de décision**:",
                              expanded=True):
                # BarPlot du client courant
                    st.markdown(
                           f'Client courant :**<span class="badge badge-success">{client_id}</span>**      score:**<span class="badge badge-info">{score}</span>**',unsafe_allow_html=True)
                    st.dataframe(df_client_couant) 
                    plt.clf()
                    plt.title(f"Arbre de décision de LightGBM)",size=20,backgroundcolor='green',color='white')
                    # Décision Plot
                    graph = lgb.create_tree_digraph(best_model, tree_index=0,
                        show_info=['split_gain', 'internal_value',
                                   'internal_count', 'internal_weight',
                                   'leaf_count', 'leaf_weight', 'data_percentage'])
                
                    st.graphviz_chart(graph,use_container_width=True)
	   

with st.container():
    st.sidebar.subheader("Facteurs d'influence")
    affiche_facteurs_influence()
    st.sidebar.subheader('Transparence du modèle')
    affiche_arbre()
# --------------------------------------------------------------------
# STATISTIQUES GENERALES
# --------------------------------------------------------------------

dico_stats = {'Variable cible': 'TARGET',
              'Type de prêt': 'NAME_CONTRACT_TYPE',
              'Sexe': 'CODE_GENDER',
              'Tél. professionnel': 'FLAG_EMP_PHONE',
              'Note région où vit client': 'REGION_RATING_CLIENT_W_CITY',
              'Niveau éducation du client': 'NAME_EDUCATION_TYPE',
              'Profession du client': 'OCCUPATION_TYPE',
              'Type d\'organisation de travail du client': 'ORGANIZATION_TYPE',
              'Adresse du client = adresse de contact': 'REG_CITY_NOT_LIVE_CITY',
              'Région du client = adresse professionnelle': 'REG_CITY_NOT_WORK_CITY',
              'Adresse du client = adresse professionnelle': 'LIVE_CITY_NOT_WORK_CITY',
              'Logement du client': 'NAME_HOUSING_TYPE',
              'Statut familial': 'NAME_FAMILY_STATUS',
              'Type de revenu du client': 'NAME_INCOME_TYPE',
              'Client possède une maison ou appartement?': 'FLAG_OWN_REALTY',
              'Accompagnateur lors de la demande de prêt?': 'NAME_TYPE_SUITE',
              'Quel jour de la semaine le client a-t-il demandé le prêt ?': 'WEEKDAY_APPR_PROCESS_START',
              'Le client a-t-il fourni un numéro de téléphone portable ?': 'FLAG_MOBIL',
              'Le client a-t-il fourni un numéro de téléphone professionnel fixe ?': 'FLAG_WORK_PHONE',
              'Le téléphone portable était-il joignable?': 'FLAG_CONT_MOBILE',             
              'Le client a-t-il fourni un numéro de téléphone domicile fixe ?': 'FLAG_PHONE',
              'Le client a-t-il fourni une adresse électronique': 'FLAG_EMAIL',
              'Âge (ans)': 'AGE_YEARS',
              'Combien d\'années avant la demande la personne a commencé son emploi actuel ?': 'YEARS_EMPLOYED',
              'Combien de jours avant la demande le client a-t-il changé son enregistrement ?': 'DAYS_REGISTRATION',
              'Combien de jours avant la demande le client a-t-il changé la pièce d\'identité avec laquelle il a demandé le prêt ?': 'DAYS_ID_PUBLISH',
              'Prix du bien que le client a demandé': 'AMT_GOODS_PRICE',
              'Nombre d\enfants?': 'CNT_CHILDREN',
              'Revenu du client': 'AMT_INCOME_TOTAL',
              'Montant du crédit du prêt': 'AMT_CREDIT',
              'Annuité de prêt': 'AMT_ANNUITY',
              'Âge de la voiture du client': 'OWN_CAR_AGE',
              'Combien de membres de la famille a le client': 'CNT_FAM_MEMBERS',
              'Population normalisée de la région où vit le client': 'REGION_POPULATION_RELATIVE',
              'Notre évaluation de la région où vit le client (1 ou 2 ou 3)': 'REGION_RATING_CLIENT',
              'Indicateur si l\'adresse permanente du client ne correspond pas à l\'adresse de contact': 'REG_REGION_NOT_LIVE_REGION',
              'Indicateur si l\'adresse permanente du client ne correspond pas à l\'adresse professionnelle': 'REG_REGION_NOT_WORK_REGION',
              'Indicateur si l\'adresse de contact du client ne correspond pas à l\'adresse de travail': 'LIVE_REGION_NOT_WORK_REGION',
              'Combien de jours avant la demande le client a-t-il changé de téléphone ?': 'DAYS_LAST_PHONE_CHANGE',
              'Statut des crédits déclarés par le Credit Bureau': 'CREDIT_ACTIVE',
              'Devise recodée du crédit du Credit Bureau': 'CREDIT_CURRENCY',
              'Type de crédit du Bureau de crédit (voiture ou argent liquide...)': 'CREDIT_TYPE',
              'Combien d\années avant la demande actuelle le client a-t-il demandé un crédit au Credit Bureau ?': 'YEARS_CREDIT',
              'Durée restante du crédit CB (en jours) au moment de la demande dans Crédit immobilier': 'DAYS_CREDIT_ENDDATE',
              'Combien de jours avant la demande de prêt la dernière information sur la solvabilité du Credit Bureau a-t-elle été fournie ?': 'DAYS_CREDIT_UPDATE',
              'Nombre de jours de retard sur le crédit CB au moment de la demande de prêt': 'CREDIT_DAY_OVERDUE',
              'Montant maximal des impayés sur le crédit du Credit Bureau jusqu\'à présent': 'AMT_CREDIT_MAX_OVERDUE',
              'Combien de fois le crédit du Bureau de crédit a-t-il été prolongé ?': 'CNT_CREDIT_PROLONG',
              'Montant actuel du crédit du Credit Bureau': 'AMT_CREDIT_SUM',
              'Dette actuelle sur le crédit du Credit Bureau': 'AMT_CREDIT_SUM_DEBT',
              'Limite de crédit actuelle de la carte de crédit déclarée dans le Bureau de crédit': 'AMT_CREDIT_SUM_LIMIT',
              'Montant actuel en retard sur le crédit du Bureau de crédit': 'AMT_CREDIT_SUM_OVERDUE',
              'Annuité du crédit du Credit Bureau': 'AMT_ANNUITY',
              'Statut du prêt du Credit Bureau durant le mois': 'STATUS',
              'Mois du solde par rapport à la date de la demande': 'MONTHS_BALANCE',
              'Statut du contrat au cours du mois': 'NAME_CONTRACT_STATUS',
              'Solde au cours du mois du crédit précédent': 'AMT_BALANCE',
              'Montant total à recevoir sur le crédit précédent': 'AMT_TOTAL_RECEIVABLE',
              'Nombre d\'échéances payées sur le crédit précédent': 'CNT_INSTALMENT_MATURE_CUM',
              'Mois du solde par rapport à la date d\'application': 'MONTHS_BALANCE',
              'Limite de la carte de crédit au cours du mois du crédit précédent': 'AMT_CREDIT_LIMIT_ACTUAL',
              'Montant retiré au guichet automatique pendant le mois du crédit précédent': 'AMT_DRAWINGS_ATM_CURRENT',
              'Montant prélevé au cours du mois du crédit précédent': 'AMT_DRAWINGS_CURRENT',
              'Montant des autres prélèvements au cours du mois du crédit précédent': 'AMT_DRAWINGS_OTHER_CURRENT',
              'Montant des prélèvements ou des achats de marchandises au cours du mois de la crédibilité précédente': 'AMT_DRAWINGS_POS_CURRENT',
              'Versement minimal pour ce mois du crédit précédent': 'AMT_INST_MIN_REGULARITY',
              'Combien le client a-t-il payé pendant le mois sur le crédit précédent ?': 'AMT_PAYMENT_CURRENT',
              'Combien le client a-t-il payé au total pendant le mois sur le crédit précédent ?': 'AMT_PAYMENT_TOTAL_CURRENT',
              'Montant à recevoir pour le principal du crédit précédent': 'AMT_RECEIVABLE_PRINCIPAL',
              'Montant à recevoir sur le crédit précédent': 'AMT_RECIVABLE', 
              'Nombre de retraits au guichet automatique durant ce mois sur le crédit précédent': 'CNT_DRAWINGS_ATM_CURRENT',
              'Nombre de retraits pendant ce mois sur le crédit précédent': 'CNT_DRAWINGS_CURRENT',
              'Nombre d\'autres retraits au cours de ce mois sur le crédit précédent': 'CNT_DRAWINGS_OTHER_CURRENT',
              'Nombre de retraits de marchandises durant ce mois sur le crédit précédent': 'CNT_DRAWINGS_POS_CURRENT',
              'DPD (jours de retard) au cours du mois sur le crédit précédent': 'SK_DPD',
              'DPD (Days past due) au cours du mois avec tolérance (les dettes avec de faibles montants de prêt sont ignorées) du crédit précédent': 'SK_DPD_DEF',
              'La date à laquelle le versement du crédit précédent était censé être payé (par rapport à la date de demande du prêt actuel)': 'DAYS_INSTALMENT',
              'Quand les échéances du crédit précédent ont-elles été effectivement payées (par rapport à la date de demande du prêt actuel) ?': 'DAYS_ENTRY_PAYMENT',
              'Version du calendrier des versements (0 pour la carte de crédit) du crédit précédent': 'NUM_INSTALMENT_VERSION',
              'Sur quel versement nous observons le paiement': 'NUM_INSTALMENT_NUMBER',
              'Quel était le montant de l\'acompte prescrit du crédit précédent sur cet acompte ?': 'AMT_INSTALMENT',
              'Ce que le client a effectivement payé sur le crédit précédent pour ce versement': 'AMT_PAYMENT',
              'Statut du contrat au cours du mois': 'NAME_CONTRACT_STATUS',
              'Durée du crédit précédent (peut changer avec le temps)': 'CNT_INSTALMENT',
              'Versements restant à payer sur le crédit précédent': 'CNT_INSTALMENT_FUTURE',
              'EXT_SOURCE_1': 'EXT_SOURCE_1',
              'EXT_SOURCE_2': 'EXT_SOURCE_2',
              'EXT_SOURCE_3': 'EXT_SOURCE_3',
              'FLAG_DOCUMENT_2': 'FLAG_DOCUMENT_2',
              'FLAG_DOCUMENT_3': 'FLAG_DOCUMENT_3',
              'FLAG_DOCUMENT_4': 'FLAG_DOCUMENT_4',
              'FLAG_DOCUMENT_5': 'FLAG_DOCUMENT_5',
              'FLAG_DOCUMENT_6': 'FLAG_DOCUMENT_6',
              'FLAG_DOCUMENT_7': 'FLAG_DOCUMENT_7',
              'FLAG_DOCUMENT_8': 'FLAG_DOCUMENT_8',
              'FLAG_DOCUMENT_9': 'FLAG_DOCUMENT_9',
              'FLAG_DOCUMENT_10': 'FLAG_DOCUMENT_10',
              'FLAG_DOCUMENT_11': 'FLAG_DOCUMENT_11',
              'FLAG_DOCUMENT_12': 'FLAG_DOCUMENT_12',
              'FLAG_DOCUMENT_13': 'FLAG_DOCUMENT_13',
              'FLAG_DOCUMENT_14': 'FLAG_DOCUMENT_14',
              'FLAG_DOCUMENT_15': 'FLAG_DOCUMENT_15',
              'FLAG_DOCUMENT_16': 'FLAG_DOCUMENT_16',
              'FLAG_DOCUMENT_17': 'FLAG_DOCUMENT_17',
              'FLAG_DOCUMENT_18': 'FLAG_DOCUMENT_18',
              'FLAG_DOCUMENT_19': 'FLAG_DOCUMENT_19',
              'FLAG_DOCUMENT_20': 'FLAG_DOCUMENT_20',
              'FLAG_DOCUMENT_21': 'FLAG_DOCUMENT_21',
              'FONDKAPREMONT_MODE': 'FONDKAPREMONT_MODE',
              'HOUSETYPE_MODE': 'HOUSETYPE_MODE',
              'WALLSMATERIAL_MODE': 'WALLSMATERIAL_MODE',
              'EMERGENCYSTATE_MODE': 'EMERGENCYSTATE_MODE',
              'FLOORSMAX_AVG': 'FLOORSMAX_AVG',
              'FLOORSMAX_MEDI': 'FLOORSMAX_MEDI',
              'FLOORSMAX_MODE': 'FLOORSMAX_MODE',
              'FLOORSMIN_MODE': 'FLOORSMIN_MODE',
              'FLOORSMIN_AVG': 'FLOORSMIN_AVG',
              'FLOORSMIN_MEDI': 'FLOORSMIN_MEDI',
              'APARTMENTS_AVG': 'APARTMENTS_AVG',
              'APARTMENTS_MEDI': 'APARTMENTS_MEDI',
              'APARTMENTS_MODE': 'APARTMENTS_MODE',
              'BASEMENTAREA_AVG': 'BASEMENTAREA_AVG',
              'BASEMENTAREA_MEDI': 'BASEMENTAREA_MEDI',
              'BASEMENTAREA_MODE': 'BASEMENTAREA_MODE',
              'YEARS_BEGINEXPLUATATION_AVG': 'YEARS_BEGINEXPLUATATION_AVG',
              'YEARS_BEGINEXPLUATATION_MODE': 'YEARS_BEGINEXPLUATATION_MODE',
              'YEARS_BEGINEXPLUATATION_MEDI': 'YEARS_BEGINEXPLUATATION_MEDI',
              'YEARS_BUILD_AVG': 'YEARS_BUILD_AVG',
              'YEARS_BUILD_MODE': 'YEARS_BUILD_MODE',
              'YEARS_BUILD_MEDI': 'YEARS_BUILD_MEDI',
              'COMMONAREA_AVG': 'COMMONAREA_AVG',
              'COMMONAREA_MEDI': 'COMMONAREA_MEDI',
              'COMMONAREA_MODE': 'COMMONAREA_MODE',
              'ELEVATORS_AVG': 'ELEVATORS_AVG',
              'ELEVATORS_MODE': 'ELEVATORS_MODE',
              'ELEVATORS_MEDI': 'ELEVATORS_MEDI',
              'ENTRANCES_AVG': 'ENTRANCES_AVG',
              'ENTRANCES_MODE': 'ENTRANCES_MODE',
              'ENTRANCES_MEDI': 'ENTRANCES_MEDI',
              'LANDAREA_AVG': 'LANDAREA_AVG',
              'LANDAREA_MEDI': 'LANDAREA_MEDI',
              'LANDAREA_MODE': 'LANDAREA_MODE',
              'LIVINGAPARTMENTS_AVG': 'LIVINGAPARTMENTS_AVG',
              'LIVINGAPARTMENTS_MODE': 'LIVINGAPARTMENTS_MODE',
              'LIVINGAPARTMENTS_MEDI': 'LIVINGAPARTMENTS_MEDI',
              'LIVINGAREA_AVG': 'LIVINGAREA_AVG',
              'LIVINGAREA_MODE': 'LIVINGAREA_MODE',
              'NONLIVINGAPARTMENTS_AVG': 'NONLIVINGAPARTMENTS_AVG',
              'NONLIVINGAPARTMENTS_MODE': 'NONLIVINGAPARTMENTS_MODE',
              'NONLIVINGAREA_AVG': 'NONLIVINGAREA_AVG',
              'NONLIVINGAREA_MEDI': 'NONLIVINGAREA_MEDI',
              'TOTALAREA_MODE': 'TOTALAREA_MODE',
              'OBS_30_CNT_SOCIAL_CIRCLE': 'OBS_30_CNT_SOCIAL_CIRCLE',
              'DEF_30_CNT_SOCIAL_CIRCLE': 'DEF_30_CNT_SOCIAL_CIRCLE',
              'DEF_60_CNT_SOCIAL_CIRCLE': 'DEF_60_CNT_SOCIAL_CIRCLE',
              'AMT_REQ_CREDIT_BUREAU_HOUR': 'AMT_REQ_CREDIT_BUREAU_HOUR',
              'AMT_REQ_CREDIT_BUREAU_DAY': 'AMT_REQ_CREDIT_BUREAU_DAY',
              'AMT_REQ_CREDIT_BUREAU_WEEK': 'AMT_REQ_CREDIT_BUREAU_WEEK',
              'AMT_REQ_CREDIT_BUREAU_MON': 'AMT_REQ_CREDIT_BUREAU_MON',
              'AMT_REQ_CREDIT_BUREAU_QRT': 'AMT_REQ_CREDIT_BUREAU_QRT',
              'AMT_REQ_CREDIT_BUREAU_YEAR': 'AMT_REQ_CREDIT_BUREAU_YEAR',
              'DAYS_ENDDATE_FACT': 'DAYS_ENDDATE_FACT'}

path_img = "resources/images/stats/"
   
def affiche_stats():
    ''' Affiche les statistiques générales provenant de l'EDA
    '''
    html_facteurs_influence="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: #DEC7CB; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      Distribution des variables générale/pour les défaillants
                  </h3>
            </div>
        </div>
        """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Voir les distributions"):     
        
        st.markdown(html_facteurs_influence, unsafe_allow_html=True)

        with st.spinner('**Affiche les statistiques générales/pour les défaillants...**'):                 
                       
            with st.expander('Distribution des variables',
                              expanded=True):
                choix = st.selectbox("Choisir une variable : ", dico_stats.keys()) 
                nom_img = dico_stats[choix]
                img =  Image.open(path_img + nom_img + ".png")                     
                st.image(img)       
                
st.sidebar.subheader('Stats générales')
affiche_stats()


# ====================================================================
# FOOTER
# ====================================================================
html_line="""
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;">
<p style="color:Gray; text-align: right; font-size:12px;">Auteur : oubairouk@gmail.com - 10/01/2023</p>
"""
st.markdown(html_line, unsafe_allow_html=True)