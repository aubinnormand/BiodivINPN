# correlation_biodiv.py

# Fonctions de corrélation
import pandas as pd
import numpy as np

def correlation_sujet_recherche(df_inpn,var,clade_sujet,taxon_sujet,df_global,clade_recherche,methode='ward'):
    df_sujet=df_inpn[(df_inpn[clade_sujet]==taxon_sujet)]
    grouped_df_sujet=df_sujet.groupby(['codeMaille10Km',clade_sujet])[var].sum()
    df_sujet_grouped = grouped_df_sujet.reset_index()
    
    #définition du dataframe où chercher la corrélation
    grouped_df_recherche=df_global.groupby(['codeMaille10Km',clade_recherche])[var].sum()
    df_recherche_grouped = grouped_df_recherche.reset_index()
    
    # Fusion des deux DataFrames en un seul
    df_sujet_grouped.rename(columns={clade_sujet: clade_recherche}, inplace=True)
    df_concat = pd.concat([df_recherche_grouped, df_sujet_grouped])
    
    # Création du tableau croisé dynamique
    pivot_df = df_concat.pivot_table(index='codeMaille10Km', columns=clade_recherche, values=var, fill_value=0)
    
    # Calcul de la matrice de corrélation entre les classes et nomScientifique
    corr_matrix = pivot_df.corr(method=methode)
    
    # Sélection de la colonne de corrélation avec 'Libelloides coccajus'
    corr_finale = corr_matrix[taxon_sujet]
    
    # Trier les corrélations par ordre décroissant de la corrélation absolue
    sorted_corr_series = corr_finale.sort_values(ascending=False)
    sorted_corr = sorted_corr_series.reset_index()
    sorted_corr = sorted_corr[sorted_corr[clade_recherche] != taxon_sujet]

    resultat=pd.merge(sorted_corr,df_inpn[[clade_recherche, 'nomVernaculaire']],on=clade_recherche)
    resultat=resultat.drop_duplicates(subset=clade_recherche)
    resultat['nomVernaculaire'] = resultat['nomVernaculaire'].str.split(',').str[0]
    colonnes_ordre = [clade_recherche, 'nomVernaculaire', taxon_sujet]
    resultat = resultat[colonnes_ordre]
    """
    # Top 10 des mieux corrélées avec Libelloides coccajus (en valeur absolue)
    top_10_corr = sorted_corr.head(10)
    top_10_corr = grouped_df.reset_index()
    
    # Trier les corrélations par ordre croissant pour obtenir les moins corrélées
    # Ici on trie selon les valeurs réelles et non absolues pour avoir les plus petites corrélations (positives ou négatives)
    bottom_5_corr = corr_finale.sort_values().head(5)
    
    # Afficher les résultats
    
    print(f"Top 10 des {clade_recherche} les mieux corrélé(e)s avec {taxon_sujet}:")
    print(top_10_corr)
    
    print(f"Top 5 des {clade_recherche} les moins corrélé(e)s avec {taxon_sujet}:")
    print(bottom_5_corr)
    """
    return resultat.reset_index(drop=True)
