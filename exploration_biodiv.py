# Fonctions d'exploration des données
from fonctions_annexes_biodiv import dictionnaire_especes
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def creation_filtre_top(df,var='nombreObs',nmin=100):
    grouped_df=df.groupby('nomScientifique')[var].sum()
    grouped_df=grouped_df.reset_index()
    top_grouped_df=grouped_df[grouped_df[var]>nmin-1]
    df_filtre_top = df[df['nomScientifique'].isin(top_grouped_df['nomScientifique'])]
    print("nombre d'espèces retenues dans le df :"+str(len(df_filtre_top['nomScientifique'].unique()))+
          ' ('+str(round(len(df_filtre_top['nomScientifique'].unique())/len(df['nomScientifique'].unique())*100))+'%)')
    return(df_filtre_top)
    
def top_especes(df,col_values='nombreObs'):
    # Sum each taxon column individually
    grouped = df.groupby(['nomScientifique'])[col_values].sum()
    grouped=grouped.reset_index()
    df_dico=dictionnaire_especes(df)
    grouped=pd.merge(grouped,df_dico,on='nomScientifique')
    # Sort the individual taxon sums in descending orde
    grouped = grouped.sort_values(by=col_values,ascending=False)
    grouped = grouped.reset_index(drop=True)
    # Afficher le DataFrame trié
    return grouped
    
def chercher_espece(df,var):
    recherche_df = df[
        df['nomScientifique'].str.contains(var, case=False, na=False) |
        df['nomVernaculaire'].str.contains(var, case=False, na=False) |
        df['genre'].str.contains(var, case=False, na=False) |
        df['famille'].str.contains(var, case=False, na=False) |
        df['ordre'].str.contains(var, case=False, na=False) |
        df['classe'].str.contains(var, case=False, na=False) |
        df['regne'].str.contains(var, case=False, na=False) 
    ]
    grouped = recherche_df.groupby('nomScientifique')['nombreObs'].sum()
    grouped_df = grouped.reset_index()
    df_dico=dictionnaire_especes(df)
    grouped_df=pd.merge(grouped_df,df_dico,on='nomScientifique')
    grouped_df = grouped_df.sort_values(by='nombreObs',ascending=False)
    grouped_df = grouped_df.reset_index(drop=True)
    return grouped_df

def explorer_clade(df,choix,taxon):  
    #Afficher toutes les occurences du clade choisi et du taxon choisi
    # Liste des clades
    liste_clade = ['all','regne', 'classe', 'ordre', 'famille', 'genre','nomScientifique','nomVernaculaire']
    
    # Trouver l'index de l'élément choisi
    index_choix = liste_clade.index(choix)
    
    # Variable pour contenir l'élément suivant (par exemple 'ordre')
    # On vérifie que l'élément suivant existe dans la liste
    if index_choix < len(liste_clade) - 1:
        clade_inf = liste_clade[index_choix + 1]
    else:
        clade_inf = None  # Pas d'élément suivant si 'choix' est le dernier élément

    df_filt=df[df[choix]==taxon]
    taxons_inf=df_filt[clade_inf].unique()
    n_obs=int(df_filt['nombreObs'].sum())
    n_especes=len(df_filt['nomScientifique'].unique())
    
    # Afficher les résultats
    #print("Clade choisi:", choix)
    #print("Taxon choisi:", taxon) 
    #print("Clade inférieur:", clade_inf)
    #print("Nombres de taxons inférieurs:", len(taxons_inf))
    #print("Taxons inférieur:", taxons_inf)
    
    # Afficher avec des séparateurs de milliers
    n_especes_formatte = "{:,}".format(n_especes)
    #print("Nombre d'espèces:", n_especes_formatte)
    
    # Afficher avec des séparateurs de milliers
    n_Obs_formatte = "{:,}".format(n_obs)
    #print("Nombre d'observations:", n_Obs_formatte)
    
    #Creer un dataframe avec comme colonnes : le nom des taxons inférieurs, le nombre d'espèces dans ce taxon et le nombre d'observation
    df_explo = pd.DataFrame()
    df_explo['taxons']=taxons_inf
    
    nobs = df_filt.groupby([clade_inf])['nombreObs'].sum()
    
    # Grouper par famille et compter les noms scientifiques uniques
    nesp = df_filt.groupby(clade_inf)['nomScientifique'].nunique().reset_index()
    nesp.columns = [clade_inf, 'nombreEspèces']
    
    # Renommer la colonne pour plus de clarté
    
    resultat=pd.merge(nesp,nobs,on=clade_inf)
    resultat['Ratio Obs/Esp']=round(resultat['nombreObs']/resultat['nombreEspèces'],1)
    # Trier le DataFrame par la colonne nombreObs
    resultat = resultat.sort_values(by='nombreObs', ascending=False).reset_index(drop=True)
    
    # Taille de la figure
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Positions des barres sur l'axe x
    indices = np.arange(len(resultat[clade_inf]))
    largeur_barres = 0.4  # Largeur des barres
    
    # Création du premier axe pour les noms scientifiques uniques
    ax1.set_yscale('log')
    bar1 = ax1.bar(indices - largeur_barres/2, resultat['nombreEspèces'], largeur_barres, label='nombreEspèces', color='b')
    ax1.set_xlabel(clade_inf)
    ax1.set_ylabel('nombreEspèces', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Créer un deuxième axe pour la somme des observations, partageant le même axe x
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    bar2 = ax2.bar(indices + largeur_barres/2, resultat['nombreObs'], largeur_barres, label='nombreObs', color='r')
    ax2.set_ylabel('nombreObs', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Ajuster les ticks de l'axe x pour qu'ils correspondent aux familles
    ax1.set_xticks(indices)
    ax1.set_xticklabels(resultat[clade_inf], rotation=45, ha='right')
    
    # Ajouter une légende combinée
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Ajuster la disposition pour éviter le chevauchement des labels
    #plt.title(f"Nombre de noms scientifiques uniques et somme des observations par famille pour l'ordre {taxon}")
    plt.tight_layout()
    plt.savefig('figure.png')
    # Afficher le graphique
    plt.show()

    # Afficher le DataFrame trié
    return resultat


def etude_mailles(df,liste_codes,col_values='nombreObs',col_groupe='nomScientifique'):
    
    df_filt=df[df['codeMaille10Km'].isin(liste_codes)]
    grouped=df_filt.groupby(col_groupe)[col_values].sum()

    return grouped