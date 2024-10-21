# Fonctions annexes
import pandas as pd
import math
import numpy as np
liste_normalisations=['nombreObs','ObsUnique','nombreObs_normalisé_log',
                      'nombreObs_normalisé_par_espece','nombreObs_normalisé_par_maille',
                      'nombreObs_normalisé_par_maille_classe','nombreObs_normalisé_par_maille_regne',
                      'nombreObs_normalisé_mrl','nombreObs_normalisé_mcl',
                     ]

def round_to_sig(num, sig=3, direction='nearest'):
    if num is None:
        raise ValueError("La valeur à arrondir ne doit pas être None.")
    if num == 0:
        return 0
    # Déterminer l'ordre de grandeur du nombre
    order_of_magnitude = math.floor(math.log10(abs(num)))
    # Calculer le facteur de multiplication
    factor = 10**(sig - order_of_magnitude - 1)
    if direction == 'nearest':
        return round(num * factor) / factor
    elif direction == 'up':
        return math.ceil(num * factor) / factor
    elif direction == 'down':
        return math.floor(num * factor) / factor
    else:
        raise ValueError("Direction must be 'nearest', 'up', or 'down'")

def dictionnaire_especes(df):
    df_dico = df.drop_duplicates(subset=['nomScientifique'])
    df_dico=df_dico[['nomVernaculaire'	,'nomScientifique' 	,'genre' 	,'famille' 	,'ordre' 	,'classe', 	'regne', 'all']]
    #df_dico=df_dico[['nomVernaculaire'	,'nomScientifique' 	,'genre' 	,'famille' 	,'ordre' 	,'classe', 	'regne', 'all','especeProtegee']]
    return df_dico

def affichage_dataframe(df,liste_colonnes,liste_norm=liste_normalisations,col_sort=None):
    # Boucle sur les colonnes de la liste
    for col in liste_norm:
        if col in df.columns:
            if np.issubdtype(df[col].dtype, np.integer):  # Vérifie si la colonne est déjà en entier
                continue  # Si c'est déjà un entier, on passe
            else:
                # Sinon, arrondit à 3 chiffres significatifs
                df[col] = df[col].apply(lambda x: round(x, 1) if pd.notnull(x) else x)

    df=df[liste_colonnes]
    df= df.loc[:, ~df.columns.duplicated()]
    if col_sort is not None:
        df=df.sort_values([col_sort], ascending=[False])
    df =df.reset_index(drop=True)
    return df