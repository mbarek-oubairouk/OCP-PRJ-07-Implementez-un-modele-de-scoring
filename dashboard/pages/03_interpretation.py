# -*- coding: utf-8 -*- 

# ====================================================================
# Chargement des librairies
# ====================================================================
import streamlit as st
import streamlit.components.v1 as components
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
import pandas as pd
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
menu_style()
header_style()
# ====================================================================
# Chargement du fichier css
# ====================================================================
local_css(FILE_CSS)



needed_vars = ['client_id','score','best_model','test_set','shap_values']

if 'page-profil'  not in st.session_state:
    st.error("Vous devez cliquer d'abord sur le menu '👨‍👩‍👧‍👦 profil du client'")
    st.stop()


for df in needed_vars:
    if df in st.session_state:
        #st.markdown(f"{df}:reloand ok", unsafe_allow_html=True)
        globals()[df] = st.session_state[df]
		
client_index = test_set[test_set['SK_ID_CURR'] == client_id].index.item()
df_client_couant = test_set[test_set['SK_ID_CURR'] == client_id]


def st_shap(plot, height=None,titre=""):
  shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
  components.html(shap_html, height=height)

# --------------------------------------------------------------------
# FACTEURS D'INFLUENCE : SHAP VALUE
# --------------------------------------------------------------------
    
def affiche_facteurs_influence():
    ''' Affiche les facteurs d'influence du client courant
    '''
    html_facteurs_influence ="""
         <h3 class="titrecli" > Variables importantes</h3>
    """
    
    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.sidebar.checkbox("Voir facteurs d'influence"):     
        
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
    <h5 class="card-title">shap.force_plot</h5>
    <p>Visualisez les valeurs SHAP données avec une disposition de force additive.<p>
    La valeur de base (<b>base value</b>) est la valeur moyenne obtenue comme sortie pour cette classe, alors que la valeur de sortie (<b>f(x)</b>) est la valeur prédite par le modèle. Les valeurs SHAP de chaque variable, proportionnelles aux tailles des flèches, « poussent » la prédiction depuis la valeur de base jusqu’à la valeur prédite.
  </div>
</div>
"""
                        ,unsafe_allow_html=True)           
            with st.expander(f"Facteurs d'influence du client courant:{client_id}",
                              expanded=True):

                

                client_index = test_set[test_set['SK_ID_CURR'] == client_id].index.item()
                X_shap = test_set.set_index('SK_ID_CURR')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)

            
                # Ici, nous utilisons l'implémentation Tree SHAP intégrée à Light GBM pour expliquer l'ensemble de données de test
                explainer = shap.TreeExplainer(best_model)
                shap_values_test = explainer.shap_values(X_shap)
                titre = "Contribution de chaque variable à la prédiction"
                st.markdown(f"""<h5 style="width:400px;padding:2px;background-color:green;color:white;">{titre}</h5>""", unsafe_allow_html=True)
                
  
                st_shap(shap.force_plot(explainer.expected_value[1], shap_values_test[1][client_index,:], X_test_courant,link="logit"))
                
                # BarPlot du client courant
                plt.clf()
                #plt.title("Diagramme d'importance des variables",size=20,backgroundcolor='green',color='white')
                titre = "Diagramme d'importance des variables"
                st.markdown(f"""<h5 style="width:400px;padding:2px;background-color:green;color:white;">{titre}</h5>""", unsafe_allow_html=True)
                fig = plt.gcf()
                shap.plots.bar( shap_values[client_index], max_display=20)
                fig.set_size_inches((10, 10))
                # Plot the graph on the dashboard
                st.pyplot(fig)

					
					
 
       
def affiche_arbre():
    ''' Affiche l'arbre du décison du molèle 
    '''
    if st.sidebar.checkbox("Arbre de décision"):
        html_arbre ="""
         <h3 class="titrecli" > Arbre de décision</h3>
         """
        st.markdown(html_arbre, unsafe_allow_html=True)
        X_grpah = test_set.set_index('SK_ID_CURR')
        X_test_courant = X_grpah.iloc[client_index]
        with st.spinner('**Transparence du modèle:**'):                    
            with st.expander(f"**Arbre de décision**:",
                              expanded=True):
                with st.container():
                    # BarPlot du client courant
                    st.markdown(
                           f'Client courant :**<span class="badge badge-success">{client_id}</span>**      score:**<span class="badge badge-info">{score}</span>**',unsafe_allow_html=True)
                    st.dataframe(df_client_couant) 
                    plt.clf()
                    plt.title(f"Arbre de décision de LightGBM)",size=20,backgroundcolor='green',color='white')
                    # Décision Plot
                    feat_imp = best_model.feature_importances_
                    df_fe = pd.DataFrame({'feature': best_model.feature_name_, 'importance': feat_imp}).sort_values('importance', ascending = False)
                    col_dec = ['EXT_SOURCE_MAX','EXT_SOURCE_VAR','EXT_SOURCE_3','PREV_APP_NAME_CONTRACT_STATUS_MEAN','INST_PAY_AMT_PAYMENT_DIFF_MEAN','BUREAU_CURRENT_DEBT_TO_CREDIT_RATIO_MEAN','EXT_SOURCE_2']
                    #df_client_couant[['SK_ID_CURR']+df_fe['feature'].to_list()]
                    st.dataframe(df_client_couant[col_dec])
                    #st.markdown(df_fe['feature'].to_list(), unsafe_allow_html=True)
                    graph = lgb.create_tree_digraph(best_model, tree_index=0,
                        show_info=['split_gain', 'internal_value',
                                   'internal_count', 'internal_weight',
                                   'leaf_count', 'leaf_weight', 'data_percentage'])
                    #plt.style.use('fivethirtyeight')
                    st.graphviz_chart(graph)
        
        

with st.container():
    st.sidebar.subheader("Facteurs d'influence")
    affiche_facteurs_influence()
    st.sidebar.subheader('Transparence du modèle')
    affiche_arbre()

    


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
<p style="color:Gray; text-align: right; font-size:12px;">Auteur : Mbarek oubairouk</p>
"""
st.markdown(html_line, unsafe_allow_html=True)
st.session_state['page-explication'] = True
