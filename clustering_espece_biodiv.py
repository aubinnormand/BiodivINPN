#clustering_espece_biodiv.py
# Cluster par espèces

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from fonctions_annexes_biodiv import dictionnaire_especes
import matplotlib.pyplot as plt
import pandas as pd
from normalisation_biodiv import (
    normalize_mrl,
    normalize_mcl,
    normalize_log
)

liste_normalisations=['nombreObs','ObsUnique','nombreObs_normalisé_log',
                      'nombreObs_normalisé_par_espece','nombreObs_normalisé_par_maille',
                      'nombreObs_normalisé_par_maille_classe','nombreObs_normalisé_par_maille_regne',
                      'nombreObs_normalisé_mrl','nombreObs_normalisé_mcl',
                     ]

def correlation_matrix_dendogram(df_input,var='nombreObs'):
    pivot_df = df_input.pivot_table(index='codeMaille10Km', columns='nomScientifique', values=var, fill_value=0)
    correlation_matrix = pivot_df.corr()
  
    return correlation_matrix

def dendogram(correlation_matrix,methode='ward',display=0):
    # Appliquer le clustering hiérarchique sur la matrice de corrélation
    Z = linkage(correlation_matrix, method=methode)
    
    # Visualiser le dendrogramme
    if display==1:
        plt.figure(figsize=(10, 7))
        dendrogram(Z, labels=correlation_matrix.columns)
        plt.show()
    return Z

def cluster_from_correlation(df_input, Z, var='nombreObs', level=4, crit='distance'):
    # Créer un tableau croisé à partir du DataFrame d'entrée
    pivot_df = df_input.pivot_table(index='codeMaille10Km', columns='nomScientifique', values=var, fill_value=0)

    # Choisir un seuil pour définir les clusters
    clusters = fcluster(Z, t=level, criterion=crit)

    # Transposer le tableau croisé pour ajouter les clusters
    pivot_df_trans = pivot_df.transpose()
    pivot_df_trans['Cluster_corr'] = clusters
    pivot_df_trans = pivot_df_trans.reset_index()

    # Créer le dictionnaire d'espèces
    df_dico = dictionnaire_especes(df_input)  # Vérifiez que df_input contient toutes les colonnes nécessaires

    # Fusionner les données
    df_cluster = pd.merge(pivot_df_trans, df_dico, on='nomScientifique', how='left')  # Utilisez 'how=left' pour éviter les problèmes d'index

    # Sélectionner les colonnes pertinentes pour le résultat final
    df_corr_cluster = df_cluster[['nomVernaculaire', 'nomScientifique', 'genre', 'famille', 'ordre', 'classe', 'regne', 'all', 'Cluster_corr']]
    return df_corr_cluster

    
def correlation_cluster(df_input,var='ObsUnique',methode='ward',level=4,display=0):
    corr=correlation_matrix_dendogram(df_input,var)
    Z=dendogram(corr,methode,display)
    df_corr_cluster=cluster_from_correlation(df_input,Z,var,level)
    return df_corr_cluster

def chercher_cluster_espece(df_corr_cluster,df,col_values,nom):
    num_cluster=df_corr_cluster[(df_corr_cluster['nomScientifique']==nom)][['Cluster_corr']]
    num_cluster=int(num_cluster.iat[0, 0])
    print(f'Le cluster contenant {nom} est le cluster n° {num_cluster}')
    return num_cluster

def afficher_liste_cluster(df_cluster_espece,df,col_values='nombreObs',liste_norm=liste_normalisations):
    liste_especes=df_cluster_espece[['nomScientifique']]
    df_filt = df[df['nomScientifique'].isin(liste_especes['nomScientifique'])]
    dict_mcl=dictionnaire_especes(df_filt)
    dict_mcl = dict_mcl.reset_index(drop=True)
    grouped = df_filt.groupby(['nomScientifique'])[liste_norm].sum()
    grouped=pd.merge(grouped,dict_mcl,on='nomScientifique')
    # Sort the individual taxon sums in descending orde
    grouped = grouped.sort_values(by=col_values,ascending=False)
    liste_cluster = grouped.reset_index(drop=True)
    #colonnes_ordre = ['nomScientifique','nomVernaculaire',liste_normalisations,'genre','famille','ordre','classe','regne','all']  # Remplace par l'ordre souhaité de tes colonnes
    # Réassignation du DataFrame avec les colonnes dans le nouvel ordre
    #liste_cluster = liste_cluster[colonnes_ordre]
    return liste_cluster

def liste_espece_cluster(df_input,df_corr_cluster,num_cluster,col_values='nombreObs'):
    df_cluster_espece=df_corr_cluster[(df_corr_cluster['Cluster_corr']==num_cluster)]
    liste_cluster=afficher_liste_cluster(df_cluster_espece,df_input,col_values)
    return liste_cluster

def groupe_dans_maille(df_input,df_corr_cluster,col_values='nombreObs',liste_norm=liste_normalisations):

    df_input_cluster_espece=pd.merge(df_input,df_corr_cluster[['nomScientifique','Cluster_corr']],on='nomScientifique')

    df_nombre_espece = df_corr_cluster.groupby('Cluster_corr')['nomScientifique'].nunique().reset_index(name='nombre_espece')
    df_nombre_espece['Cluster_corr'] = df_nombre_espece['Cluster_corr'].astype('int64')
    
    grouped=df_input_cluster_espece.groupby(['codeMaille10Km','Cluster_corr'])[liste_norm].sum()
    
    grouped=grouped.reset_index()
    grouped['Cluster_corr'] = grouped['Cluster_corr'].astype('int64')
    grouped=pd.merge(grouped,df_nombre_espece,on='Cluster_corr')
    grouped['nombreObs_normalisé_par_espece']= grouped['nombreObs_normalisé_par_espece']/grouped['nombre_espece']
    grouped['nombreObs_normalisé_par_maille_regne']= grouped['nombreObs_normalisé_par_maille_regne']/grouped['nombre_espece']
    grouped['nombreObs']= grouped['nombreObs']/grouped['nombre_espece']
    grouped=normalize_mrl(grouped, observation_col='nombreObs_normalisé_par_maille_regne')
    

    # Ajouter le nom des espèces dans les groupes
    df_filt = df_input_cluster_espece.groupby(['nomScientifique','Cluster_corr'])[col_values].sum()
    df_filt=df_filt.reset_index()
    # Créer le dictionnaire d'espèces
    df_dico = dictionnaire_especes(df_input)  # Vérifiez que df_input contient toutes les colonnes nécessaires
    # Fusionner les données
    df_filt = pd.merge(df_filt, df_dico, on='nomScientifique', how='left')  # Utilisez 'how=left' pour éviter les problèmes d'index
    df_filt = df_filt.sort_values(by=[col_values], ascending=[False])
    grouped_nomScientifique = df_filt.groupby(['Cluster_corr'])['nomScientifique'].agg(lambda x: ', '.join(x.unique()))
    grouped_nomVernaculaire = df_filt.groupby(['Cluster_corr'])['nomVernaculaire'].agg(lambda x: ', '.join(x.fillna('').unique()))
    grouped=pd.merge(grouped,grouped_nomScientifique,on='Cluster_corr')
    grouped=pd.merge(grouped,grouped_nomVernaculaire,on='Cluster_corr')
    grouped=grouped.reset_index(drop=True)
    grouped = grouped.sort_values(by=col_values, ascending=[False])
    
    # Réassignation du DataFrame avec les colonnes dans le nouvel ordre
    colonnes_ordre = ['codeMaille10Km','Cluster_corr','nomScientifique','nomVernaculaire']+liste_norm  # Remplace par l'ordre souhaité de tes colonnes
    grouped = grouped[colonnes_ordre]
    
    return grouped

def etude_cluster_local(df_input,liste_mailles,df_corr_cluster,num_cluster,col_values,liste_norm=liste_normalisations):
    df_filt=df_input[df_input['codeMaille10Km'].isin(liste_mailles)]
    
    liste_espece_cluster_local=liste_espece_cluster(df_filt,df_corr_cluster,num_cluster,col_values)
    
    liste_espece_cluster_entier=liste_espece_cluster(df_input,df_corr_cluster,num_cluster,col_values)
    liste_espece_cluster_entier=liste_espece_cluster_entier.drop(columns=liste_norm)
    
    
    liste_espece_merged= pd.merge(liste_espece_cluster_local, liste_espece_cluster_entier,how='outer')
    # Remplacer les valeurs manquantes par 0
    liste_espece_merged.fillna(value={col: 0 for col in liste_norm}, inplace=True)

    liste_espece_merged = liste_espece_merged.sort_values(by=col_values, ascending=False)
    liste_espece_merged=liste_espece_merged.reset_index(drop=True)
    return liste_espece_merged

def chercher_especes_pas_presentes(df_input,df_cluster_maille,df_corr_cluster,liste_codes,col_choice='nombreObs_normalisé_par_espece',liste_norm=liste_normalisations):

    # chercher les espèces qui pourraient être potentiellement présentes
    liste_cluster=df_cluster_maille['Cluster_corr'].unique()[:20]
    df_espece_pas_presentes=pd.DataFrame()
    for num_cluster in liste_cluster:
        df_cluster_local=etude_cluster_local(df_input,liste_codes,df_corr_cluster,num_cluster,col_choice)
        df_espece_pas_presentes_temp=df_cluster_local[df_cluster_local['nombreObs']==0]
        df_espece_pas_presentes=pd.concat([df_espece_pas_presentes, df_espece_pas_presentes_temp], ignore_index=True)
    df_espece_pas_presentes=pd.merge(df_espece_pas_presentes,df_corr_cluster[['nomScientifique','Cluster_corr']],on='nomScientifique')
    df_espece_pas_presentes=df_espece_pas_presentes.drop(columns=liste_norm)

    df_input_grouped=df_input.groupby(['nomScientifique'])[liste_norm].sum()
    df_input_grouped=df_input_grouped.reset_index(drop=False)
    df_espece_pas_presentes=pd.merge(df_espece_pas_presentes,df_input_grouped[['nomScientifique','nombreObs','nombreObs_normalisé_par_maille_regne']],on='nomScientifique')
    df_espece_pas_presentes = df_espece_pas_presentes.groupby('Cluster_corr', sort=False).apply(lambda x: x.sort_values('nombreObs', ascending=False)).reset_index(drop=True)
    return df_espece_pas_presentes

def chercher_zone_espece_pas_presente(df_input,df_corr_cluster,df_inpn_cluster_espece,espece,
                                      col_choice='nombreObs_normalisé_par_maille_regne'):

    num_cluster=chercher_cluster_espece(df_corr_cluster,df_input,col_choice,espece)
    
    df_cluster_espece_manquante=df_inpn_cluster_espece[df_inpn_cluster_espece['Cluster_corr']==num_cluster]
    df_cluster_espece_manquante = df_cluster_espece_manquante.sort_values(by=col_choice,ascending=False)
    

    
    return df_cluster_espece_manquante
