# -*- coding: utf-8 -*- 

# ====================================================================
# Chargement des librairies
# ====================================================================

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
# Warnings
import warnings
warnings.filterwarnings('ignore')


#st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
#st.markdown(html_legende)
# ====================================================================
# HEADER - TITRE
# ====================================================================
menu_style()
header_style()

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

st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)

# ====================================================================
# Chargement du fichier css
# ====================================================================
local_css(FILE_CSS)


if 'client_id' in st.session_state:
     client_id = st.session_state['client_id']
if 'score' in st.session_state:
     score = st.session_state['score']
if 'score_10test' in st.session_state:
     score_10test = st.session_state['score_10test']

needed_df = ['df_info_voisins','df_pret_voisins','df_dashboard',
             'df_voisin_train_agg','df_all_train_agg','df_client_courant',
             'client_info','client_pret']

if 'page-score'  not in st.session_state:
    st.error("Vous devez cliquer d'abord sur le menu '🥇 Score du client'")
    st.stop()

for df in needed_df:
    if df in st.session_state:
        #st.markdown(f"{df}:reloand ok", unsafe_allow_html=True)
        globals()[df] = st.session_state[df]


#st.markdown(df_client_courant.columns, unsafe_allow_html=True)
# --------------------------------------------------------------------
# CLIENTS SIMILAIRES 
# --------------------------------------------------------------------
def infos_clients_similaires():
    ''' Affiche les informations sur les clients similaires :
            - traits stricts.
            - demande de prêt
    '''
    html_clients_similaires = """ <h3 class="titrecli" > {tag}</h3>"""

    # titre = True

    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES ===========================
    chkComparGraphe = st.sidebar.checkbox("Similarité graphique?")
    if chkComparGraphe:

        # if titre:
        st.markdown(html_clients_similaires.format(tag="Similarité graphique"), unsafe_allow_html=True)
            # titre = False

        with st.spinner('**Affiche les graphiques comparant le client courant et les clients similaires...**'):

            with st.expander('Comparaison variables impactantes client courant/moyennes des clients similaires',
                             expanded=True):
                with st.container():
                    # Préparatifs dataframe
                    df_client = df_voisin_train_agg.query(
                        'ID_CLIENT == @client_id').astype(int)
                    # ====================================================================
                    # Lineplot comparatif features importances client courant/voisins
                    # ====================================================================

                    # ===================== Valeurs moyennes des features importances pour le client courant =====================
                    feat_cols = ['SK_ID_CURR', 'DAYS_BIRTH',
                        'DAYS_ID_PUBLISH']+grp_cols_1+grp_cols_2
                    feat_cols = [c for c in feat_cols if c not in [
                        'YEAR_BIRTH', 'YEAR_ID_PUBLISH']]
                    df_feat_client = df_client_courant[feat_cols]
                    df_feat_client['YEAR_BIRTH'] = \
                        np.trunc(
                            np.abs(df_feat_client['DAYS_BIRTH'] / 365)).astype('int8')
                    df_feat_client['YEAR_ID_PUBLISH'] = \
                        np.trunc(
                            np.abs(df_feat_client['DAYS_ID_PUBLISH'] / 365)).astype('int8')
                    df_feat_client.drop(columns=['DAYS_BIRTH', 'DAYS_ID_PUBLISH'],
                                        inplace=True)
                    df_feat_client_gp1 = df_feat_client[grp_cols_1]
                    df_feat_client_gp2 = df_feat_client[grp_cols_2]
                    # X
                    x_gp1 = df_feat_client_gp1.columns.to_list()
                    x_gp2 = df_feat_client_gp2.columns.to_list()
                    # y
                    y_feat_client_gp1 = df_feat_client_gp1.values[0].tolist()
                    y_feat_client_gp2 = df_feat_client_gp2.values[0].tolist()

                    # ===================== Valeurs moyennes des features importances pour les 10 voisins =======================
                    feat_cols_voisins = [
                        'ID_CLIENT', 'DAYS_BIRTH_MEAN', 'DAYS_ID_PUBLISH_MEAN']+grp_cols_3+grp_cols_4
                    feat_cols_voisins = [c for c in feat_cols_voisins if c not in [
                        'YEAR_BIRTH_MEAN', 'YEAR_ID_PUBLISH_MEAN']]
                    df_moy_feat_voisins = df_client[feat_cols_voisins]
                    df_moy_feat_voisins['YEAR_BIRTH_MEAN'] = \
                        np.trunc(
                            np.abs(df_moy_feat_voisins['DAYS_BIRTH_MEAN'] / 365)).astype('int8')
                    df_moy_feat_voisins['YEAR_ID_PUBLISH_MEAN'] = \
                        np.trunc(
                            np.abs(df_moy_feat_voisins['DAYS_ID_PUBLISH_MEAN'] / 365)).astype('int8')
                    df_moy_feat_voisins.drop(columns=['DAYS_BIRTH_MEAN', 'DAYS_ID_PUBLISH_MEAN'],
                                        inplace=True)
                    df_moy_feat_voisins_gp3 = df_moy_feat_voisins[grp_cols_3]
                    df_moy_feat_voisins_gp4 = df_moy_feat_voisins[grp_cols_4]
                    # y
                    y_moy_feat_voisins_gp3 = df_moy_feat_voisins_gp3.values[0].tolist(
                    )
                    y_moy_feat_voisins_gp4 = df_moy_feat_voisins_gp4.values[0].tolist(
                    )

                    # ===================== Valeurs moyennes de tous les clients non-défaillants/défaillants du train sets =======================
                    feat_cols_voisins_mean = [
                        'TARGET', 'DAYS_ID_PUBLISH_MEAN']+grp_cols_3+grp_cols_4
                    feat_cols_voisins_mean = [
                        c for c in feat_cols_voisins_mean if c not in ['YEAR_ID_PUBLISH_MEAN']]
                    df_all_train = df_all_train_agg[feat_cols_voisins_mean]
                    df_all_train['YEAR_ID_PUBLISH_MEAN'] = \
                        np.trunc(
                            np.abs(df_all_train['DAYS_ID_PUBLISH_MEAN'] / 365)).astype('int8')
                    df_all_train.drop(columns=['DAYS_ID_PUBLISH_MEAN'],
                                        inplace=True)
                    # Non-défaillants
                    df_all_train_nondef_gp3 = df_all_train[df_all_train['TARGET']
                        == 0][grp_cols_3]
                    df_all_train_nondef_gp4 = df_all_train[df_all_train['TARGET']
                        == 0][grp_cols_4]
                    # Défaillants
                    df_all_train_def_gp3 = df_all_train[df_all_train['TARGET']
                        == 1][grp_cols_3]
                    df_all_train_def_gp4 = df_all_train[df_all_train['TARGET']
                        == 1][grp_cols_4]
                    # y
                    # Non-défaillants
                    y_all_train_nondef_gp3 = df_all_train_nondef_gp3.values[0].tolist(
                    )
                    y_all_train_nondef_gp4 = df_all_train_nondef_gp4.values[0].tolist(
                    )
                    # Défaillants
                    y_all_train_def_gp3 = df_all_train_def_gp3.values[0].tolist(
                    )
                    y_all_train_def_gp4 = df_all_train_def_gp4.values[0].tolist(
                    )

                    # Légende des courbes
                    # st.image(lineplot_legende)

                    st.caption(html_legende, unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1.5])
                    with col1:
                        # Lineplot de comparaison des features importances client courant/voisins/all ================
                        plt.figure(figsize=(6, 6))
                        plt.plot(x_gp1, y_feat_client_gp1, color='Orange')
                        plt.plot(x_gp1, y_moy_feat_voisins_gp3,
                                 color='SteelBlue')
                        plt.plot(x_gp1, y_all_train_nondef_gp3, color='Green')
                        plt.plot(x_gp1, y_all_train_def_gp3, color='Crimson')
                        plt.xticks(rotation=90)
                        # plt.legend(title='Légende :', title_fontsize=16, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0., fontsize=16)
                        st.pyplot()
                        # plt.xticks(rotation=90)
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                    with col2:
                        # Lineplot de comparaison des features importances client courant/voisins/all ================
                        plt.figure(figsize=(8, 5))
                        plt.plot(x_gp2, y_feat_client_gp2, color='Orange')
                        plt.plot(x_gp2, y_moy_feat_voisins_gp4,
                                 color='SteelBlue')
                        plt.plot(x_gp2, y_all_train_nondef_gp4, color='Green')
                        plt.plot(x_gp2, y_all_train_def_gp4, color='Crimson')
                        plt.xticks(rotation=90)
                        # plt.legend(title='Légende :', title_fontsize=16, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0., fontsize=16)
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                        st.pyplot()
            with st.expander('Comparaison variables impactantes client courant/moyennes des clients similaires',
                             expanded=True):
                    with st.container():

                        vars_select = list(champs.keys())

                        feat_imp_to_show = st.multiselect(label=f"Feature(s) importance(s) à visualiser:",
                                                          options=vars_select)

                        # ==============================================================
                        # Variables seléctionnes
                        # ==============================================================
                        for col in feat_imp_to_show:
                            graphinfo = champs[col]
                            desc = graphinfo.get('desc')
                            charts = graphinfo.get('chart', "")
                            bullets = 'bullets' in charts
                            violin = 'ViolinPlot' in charts
                            distri = 'DistPlot' in charts
                            f = None
                            match_num = re.search(r'fun=([0-9]+)', charts)
                            refence_val = int(df_dashboard.query('SK_ID_CURR == @client_id')[col].values)
                            if match_num:
                                 f = int(match_num.group(1))
                            with st.spinner(f'**Chargement du graphique comparatif {col}...**'):
                                if bullets:
                                    cols = [f'{col}_'+c for c in ['MIN', 'Q25', 'MEAN', 'Q75', 'MAX']]
                                    for c in cols:
                                        if c not in df_client.columns:
                                            print(
                                              f'il manque {c} dans le df_client')
                                            return 0
                                    c_min = int(df_client[f'{col}_MIN'].values*(f))
                                    c_q25 = int(df_client[f'{col}_Q25'].values*(f))
                                    c_mean = int(df_client[f'{col}_MEAN'].values*(f))
                                    c_q75 = int(df_client[f'{col}_Q75'].values*f)
                                    c_max = int(df_client[f'{col}_MAX'].values*f)   
                                    c_axis_min = min(c_min, refence_val)
                                    c_axis_max = max(c_max, refence_val)
                                    fig_c = go.Figure()
                                
                                    fig_c.add_trace(go.Indicator(
                                       mode = "number+gauge+delta",
                                       value = refence_val,
                                       delta = {'reference': c_mean,
                                                'increasing': {'color': 'Crimson'},
                                                'decreasing': {'color': 'Green'}},
                                       domain = {'x': [0.5, 1], 'y': [0.8, 1]},
                                       #title = {'text': desc, 'font': {'size': 12},
                                       #         'align' : 'left'},
                                       gauge = {
                                           'shape': 'bullet',
                                           'axis': {'range': [c_axis_min, c_axis_max]},
                                           'threshold': {
                                               'line': {'color': 'black', 'width': 3},
                                               'thickness': 0.75,
                                               'value': refence_val},
                                           'steps': [
                                                {'range': [0, c_min], 'color': 'white'},
                                                {'range': [c_min, c_q25], 'color': '#de3a5b'},
                                                {'range': [c_q25, c_mean], 'color': '#dec7cb',
                                                           'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [c_mean, c_q75],'color': '#dec7cb',
                                                           'line': {'color': 'DarkSlateGray', 'width': 2}},
                                                {'range': [c_q75, c_max], 'color': '#de3a5b'}],
                                           'bar': {'color': 'black'}}))
                                
                                    fig_c.update_layout(height=200,
                                                      margin={'t':0, 'b':0, 'l':0})
                                
                                    html_col = f"<h4 class='h4col'>{col}</h4> <br/> <h5 class='h5coldes'>{desc}</h5> <hr/>"
                                    st.markdown(html_col, unsafe_allow_html=True)

                                    # ==================== Go Indicator bullets ==============================================
                                    st.plotly_chart(fig_c)
                                if violin:
                                   # ==================== ViolinPlot ========================================================
                                   
                                   sns.violinplot(x='PRED_CLASSE_CLIENT', y=col,
                                               data=df_dashboard,
                                               palette=['Green', 'Crimson'])

                                   plt.plot(df_client_courant['PRED_CLASSE_CLIENT'],
                                         refence_val,
                                         color="orange",
                                         marker="$\\bigodot$", markersize=12)
                                   plt.xlabel('TARGET', fontsize=12)
                                   client = mlines.Line2D([], [], color='orange', marker='$\\bigodot$',
                                                       linestyle='None',
                                                       markersize=12, label='Position du client')
                                   plt.legend(handles=[client], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                   #st.set_option('deprecation.showPyplotGlobalUse', False)
                                   st.pyplot()
                                if distri:                                    
                                  # ==================== DistPlot ==========================================================
                                  # Non-défaillants
                                   #plt.figure(figsize=(1, 1))
                                   sns.kdeplot(df_dashboard.query('PRED_CLASSE_CLIENT == 0')[col],
                                             label='Non-Défaillants', color='Green')
                                  # Défaillants
                                   sns.kdeplot(df_dashboard.query('PRED_CLASSE_CLIENT == 1')[col],
                                             label='Défaillants', color='Crimson')
                                   plt.xlabel(col, fontsize=12)
                                   plt.ylabel('Probability Density', fontsize=12)
                                   plt.xticks(fontsize=12, rotation=90)
                                   plt.yticks(fontsize=12)
                                   # Position du client
                                   plt.axvline(x=refence_val, color='orange', label='Position du client')
                                   plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
                                   #st.set_option('deprecation.showPyplotGlobalUse', False)   
                                   st.pyplot()         



                                    
    # ====================== COMPARAISON TRAITS STRICTS CLIENT COURANT / CLIENTS SIMILAIRES ============================
    if st.sidebar.checkbox("Similarité socio-démographiques ?"):     
        st.markdown(html_clients_similaires.format(tag="Comparaison socio-démographiques"), unsafe_allow_html=True) 
            
        with st.spinner('**Affiche les traits stricts comparant le client courant et les clients similaires...**'):                 
                                          
            with st.expander('Similarité socio-démographiques',
                             expanded=True):
                    # Infos principales clients similaires
                    voisins_info = df_info_voisins[df_info_voisins['ID_CLIENT'] == client_id].iloc[:, 1:]
                    voisins_info.set_index('INDEX_VOISIN', inplace=True)
                    st.markdown(
                           f'Client courant :**<span class="badge badge-success">{client_id}</span>**      score:**<span class="badge badge-info">{score}</span>**',unsafe_allow_html=True)
                    st.dataframe(client_info) 
                    st.markdown(
                           f'10 clients similaires (score moyen :arrow_forward: **<span class="badge badge-info">{score_10test}</span>**):',unsafe_allow_html=True)
                    st.dataframe(voisins_info.style.highlight_max(axis=0))
            
    # ====================== COMPARAISON DEMANDE DE PRÊT CLIENT COURANT / CLIENTS SIMILAIRES ============================
    if st.sidebar.checkbox("Similarité demande prêt ?"):     

        st.markdown(html_clients_similaires.format(tag="Similarité demande prêt"), unsafe_allow_html=True)

        with st.spinner('**Affiche les informations de la demande de prêt comparant le client courant et les clients similaires...**'):                 

            with st.expander('Comparaison demande de prêt',
                             expanded=True):
                    # Infos principales sur la demande de prêt
                    voisins_pret = df_pret_voisins[df_pret_voisins['ID_CLIENT'] == client_id].iloc[:, 1:]
                    voisins_pret.set_index('INDEX_VOISIN', inplace=True)
                    st.write('Client courant')
                    st.dataframe(client_pret)
                    st.write('10 clients similaires')
                    st.dataframe(voisins_pret.style.highlight_max(axis=0))
					
st.sidebar.subheader('Clients similaires')
infos_clients_similaires()
st.session_state['page-profil'] = True