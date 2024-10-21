# biodiversite_endemisme_biodiv.py

# Fonctions Indices de Biodiversité et Endémisme

import numpy as np
import pandas as pd

def shannon(df_input,col_value,code_col):
    #Mesure de la biodiversité
    grouped = df_input.groupby(code_col).agg(somme_obs=(col_value, 'sum'))
    df_input=pd.merge(df_input,grouped,on=code_col)
    df_input['prop_Obs']=df_input[col_value]/df_input['somme_obs']
    
    #Indice de Shannon
    df_input['shannon_prep']=df_input['prop_Obs']*np.log10(df_input['prop_Obs'])
    grouped_shannon = df_input.groupby(code_col).agg(shannon=('shannon_prep', 'sum'))
    grouped_shannon['shannon']=-grouped_shannon['shannon']

    return grouped_shannon


def simpson(df_input,col_value,code_col):
    #Mesure de la biodiversité
    grouped = df_input.groupby(code_col).agg(somme_obs=(col_value, 'sum'))
    df_input=pd.merge(df_input,grouped,on=code_col)
    df_input['prop_Obs']=df_input[col_value]/df_input['somme_obs']
    
    #Indice de Simpson
    df_input['Simpson_prep']=df_input['prop_Obs']**2
    grouped_simpson = df_input.groupby(code_col).agg(simpson=('Simpson_prep', 'sum'))
    grouped_simpson['simpson']=1-grouped_simpson['simpson']
    
    return grouped_simpson
    
#Mesure de l'endémicité
#Indice de Weighted Endemism (WE)
def WE(df_input,code_col):
    grouped = df_input.groupby('nomScientifique').agg(aire_repartition=('ObsUnique', 'sum'))
    grouped['WE_prep']=1/grouped['aire_repartition']
    df_input=pd.merge(df_input,grouped,on='nomScientifique')
    grouped_WE = df_input.groupby(code_col).agg(WE=('WE_prep', 'sum'))
        
    return grouped_WE

def calcul_indice(df_input,col_value,code_col):
    grouped_shannon=shannon(df_input,col_value,code_col)
    grouped_simpson=simpson(df_input,col_value,code_col)
    grouped_WE=WE(df_input,code_col)
    df_indice=pd.merge(grouped_shannon,grouped_simpson,on=code_col)
    df_indice=pd.merge(df_indice,grouped_WE,on=code_col)
    return df_indice