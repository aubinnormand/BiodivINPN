#normalisation_biodiv.py 
# Fonctions de normalisation
import numpy as np

# Fonction pour caper les valeurs par la 10ème plus grande
def cap_to_nth_largest(group, n):
    if len(group) > n:
        # Obtenir la n-ième plus grande valeur
        nth_value = group['nombreObs'].nlargest(n).iloc[-1]
        # Créer une nouvelle colonne avec les valeurs capées
        group['nombreObs_capped'] = group['nombreObs'].apply(lambda x: min(x, nth_value))
    else:
        # Si moins de n valeurs, la colonne capée est égale à la colonne originale
        group['nombreObs_capped'] = group['nombreObs']
    return group

def normalize_cap(df,n=10):
# Appliquer la fonction à chaque groupe de nomScientifique
    df_capped = df.groupby('nomScientifique').apply(lambda group: cap_to_nth_largest(group, n)).reset_index(drop=True)
    return df_capped

def normalize_maille(df, code_col='codeMaille10Km', observation_col='nombreObs'):
    """
    Normalise les observations dans un DataFrame pour que la somme des observations
    pour chaque codeMaille10Km soit égale à target_sum.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données à normaliser.
    code_col (str): Le nom de la colonne contenant les codeMaille10Km.
    observation_col (str): Le nom de la colonne contenant les observations.
    target_sum (int): La somme cible pour les observations normalisées par codeMaille10Km.

    Returns:
    pd.DataFrame: Le DataFrame avec une colonne supplémentaire des observations normalisées.
    """
    # Calcul de la somme maximale parmi les groupes
    col_norm=observation_col + '_normalisé_par_maille'
    target_mean = df.groupby([code_col])[observation_col].sum().mean()
    
    somme_obs= df.groupby([code_col])[observation_col].transform('sum')

    # Appliquer le facteur de normalisation à chaque observation
    
    df[col_norm] = (df[observation_col] / somme_obs) * target_mean
    print('Il y a en moyenne '+str(target_mean.round(1))+' observations par maille')
    df[col_norm] = df[col_norm].round(3)
    return df

def normalize_maille_clade(df, code_col='codeMaille10Km', clade_col='classe', observation_col='nombreObs'):
    col_norm=observation_col + '_normalisé_par_maille_'+clade_col
    target_mean = df.groupby([code_col,clade_col])[observation_col].sum().mean()
    
    somme_obs_par_maille_clade= df.groupby([code_col,clade_col])[observation_col].transform('sum')

    # Appliquer le facteur de normalisation à chaque observation
    
    df[col_norm] = (df[observation_col] / somme_obs_par_maille_clade) * target_mean
    print('Il y a en moyenne '+str(target_mean.round(1))+' observations par '+clade_col+' par maille')

    df[col_norm] = df[col_norm].round(3)
    return df

def normalize_espece(df, code_col='nomScientifique', observation_col='nombreObs'):
    # Calcul de la somme maximale parmi les groupes
    col_norm=observation_col + '_normalisé_par_espece'
    # Calcul de la somme des nombreObs par nomScientifique
    somme_obs_par_nom = df.groupby(code_col)[observation_col].transform('sum')
    
    # Normalisation de chaque nombreObs
    df[col_norm] = (df[observation_col] / somme_obs_par_nom) * 10000

    print('Le nombre d observations total par espèce est fixé à 10 000')

    df[col_norm] = df[col_norm].round(3)
    return df

def normalize_clade(df, clade_col='classe', observation_col='nombreObs'):
    col_norm=observation_col + '_normalisé_par_'+clade_col

    target_mean = df.groupby([clade_col])[observation_col].sum().mean()
    
    somme_obs= df.groupby([clade_col])[observation_col].transform('sum')

    # Appliquer le facteur de normalisation à chaque observation
    
    df[col_norm] = (df[observation_col] / somme_obs) * target_mean
    print('Il y a en moyenne '+str(target_mean.round(1))+' observations par '+clade_col)

    df[col_norm] = df[col_norm].round(3)
    return df
    
def normalize_unique(df):
    df['ObsUnique'] = df['nombreObs'].apply(lambda x: 1 if x > 0 else 0)
    return df

def normalize_log(df, observation_col='nombreObs'):
    col_norm=observation_col+ '_normalisé_log'
    min_value = df[observation_col].min()
    df[col_norm] = df[observation_col]/min_value
    # Appliquer la fonction de normalisation sur le DataFrame
    df[col_norm] = df[col_norm].apply(lambda x: np.log10(x) if x > 0 else 0)
    df[col_norm] = df[col_norm].round(2)
    return df

def normalize_mcl(df, observation_col='nombreObs_normalisé_par_maille_classe'):
    col_norm='nombreObs_normalisé_mcl'
    min_value = df[observation_col].min()
    df[col_norm] = df[observation_col]/min_value
    # Appliquer la fonction de normalisation sur le DataFrame
    df[col_norm] = df[col_norm].apply(lambda x: np.log10(x) if x > 0 else 0)
    df[col_norm] = df[col_norm].round(2)
    return df

def normalize_mrl(df, observation_col='nombreObs_normalisé_par_maille_regne'):
    col_norm='nombreObs_normalisé_mrl'
    min_value = df[observation_col].min()
    df[col_norm] = df[observation_col]/min_value
    # Appliquer la fonction de normalisation sur le DataFrame
    df[col_norm] = df[col_norm].apply(lambda x: np.log(x) if x > 0 else 0)
    df[col_norm] = df[col_norm].round(2)
    return df
    
    
    