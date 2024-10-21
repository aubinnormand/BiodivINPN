import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import re
import itertools

def dictionnaire_especes(df):
    df_dico = df.drop_duplicates(subset=['nomScientifique'])
    df_dico=df_dico[['nomVernaculaire'	,'nomScientifique' 	,'genre' 	,'famille' 	,'ordre' 	,'classe', 	'regne', 'all']]
    return df_dico

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    numbers = [num for num in numbers if int(num) <= 900]  # Filtrer les valeurs <= 900
    if '2A' in s or '2B' in s:
        if s=="2A | 2B":
            return '2A','2B'
        if '2A' in s:
            return '2A'
        if '2B' in s:
            return '2B'
    else:
        return ', '.join(numbers)
    
    from datetime import datetime


def extraction_annee_mois_jour_dataframe(df_raw):
    # Convertir la colonne 'dateObservation' en type datetime
    df_raw['dateObservation'] = pd.to_datetime(df_raw['dateObservation'],format='ISO8601', errors='coerce')

    # Créer les colonnes 'year', 'month' et 'day'
    df_raw['year'] = df_raw['dateObservation'].dt.year
    df_raw['month'] = df_raw['dateObservation'].dt.month
    df_raw['day'] = df_raw['dateObservation'].dt.day
    df_raw['day_of_year'] = df_raw['dateObservation'].dt.dayofyear
    return df_raw

def assigner_periode(df,bornes=[1500,1900, 1950, 1970, 1980, 1990,2000, 2010, 2020,2024]):
# Découper en périodes avec pd.cut()
    df['periode'] = pd.cut(df['year'], bins=bornes_periodes, 
                           labels=[f'Période {i+1}: {bornes_periodes[i]} à {bornes_periodes[i+1]}' for i in range(len(bornes_periodes) - 1)],
                           include_lowest=False)  # include_lowest=True inclut la borne inférieureure
    return df
        
def completer_df(df):

    # Extraire les valeurs uniques pour Code, Année et nomScientifique
    codes = df['Code'].unique()
    periodes = df['Année'].unique()
    noms_scientifiques = df['nomScientifique'].unique()

    # Créer toutes les combinaisons possibles
    combinations = list(itertools.product(codes, periodes, noms_scientifiques))

    # Créer un dataframe à partir des combinaisons
    df_combinations = pd.DataFrame(combinations, columns=['Code', 'Période', 'nomScientifique'])

    # Joindre avec les colonnes liées (Libellé et geometry) en utilisant 'Code'
    df_codes = df[['Code', 'Libellé']].drop_duplicates()
    
    # Joindre avec les colonnes liées (Libellé et geometry) en utilisant 'Code'
    df_date = df[['Année', 'Période']].drop_duplicates()

    # Joindre avec les colonnes liées  en utilisant 'nomScientifique'
    df_taxons = df[['nomScientifique', 'nomVernaculaire', 'regne','classe', 'ordre', 'famille', 'genre']].drop_duplicates()
                 
    # Fusionner df_combinations avec df_codes pour obtenir Libellé et geometry associés à chaque Code
    df_combinations = pd.merge(df_combinations, df_codes, on='Code', how='left')
                 
    # Fusionner df_combinations avec df_codes pour obtenir Libellé et geometry associés à chaque Code
    df_combinations = pd.merge(df_combinations, df_date, on='Année', how='left')
                 
    df_combinations = pd.merge(df_combinations, df_taxons, on='nomScientifique', how='left')

    # Fusionner avec le dataframe original pour obtenir les valeurs de nombreObs
    df_merged = pd.merge(df_combinations, df, on=['Code', 'Libellé','Année','Période', 'nomScientifique','nomVernaculaire', 'regne','classe', 'ordre', 'famille', 'genre'], how='left')

    # Remplir les valeurs manquantes pour nombreObs avec 0
    df_merged['nombreObs'] = df_merged['nombreObs'].fillna(0)
    
    return df_merged


def nettoyer_nom_scientifique(nom):
    mots = nom.split()  # Séparer les mots par espace
    if len(mots) == 3 and mots[1].lower() == mots[2].lower():  # Vérifier si le deuxième et troisième mot sont identiques
        return ' '.join(mots[:2])  # Garder seulement les deux premiers mots
    return nom  # Sinon, renvoyer le nom tel quel

