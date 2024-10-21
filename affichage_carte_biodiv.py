#affichage_carte_biodiv.py

# Fonctions d'affichage de carte 

import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import pandas as pd 
import numpy as np
from fonctions_annexes_biodiv import (
    round_to_sig,
    dictionnaire_especes,
    affichage_dataframe
)
import matplotlib.patches as mpatches

from normalisation_biodiv import (
    normalize_mrl,
    normalize_mcl,
    normalize_log
)

def afficher_carte_observations(df,df_geo,col_values='nombreObs',n=1,m=1,cmap_choice='viridis',title=None,save_path=None,basemap_type='OpenStreetMap',zoom_map=6):
    
    # Regrouper les données par 'codeMaille10Km' et 'geometry' et calculer la somme des observations
    if col_values=='nombreObs_normalisé_log':
        grouped = df.groupby(['codeMaille10Km'])['nombreObs'].sum()
        grouped=grouped.reset_index()
        grouped=normalize_log(grouped)
    elif col_values=='nombreObs_normalisé_mcl':
        grouped = df.groupby(['codeMaille10Km'])['nombreObs_normalisé_par_maille_classe'].sum()
        grouped=grouped.reset_index()
        grouped=normalize_mcl(grouped)
    elif col_values=='nombreObs_normalisé_mrl':
        grouped = df.groupby(['codeMaille10Km'])['nombreObs_normalisé_par_maille_regne'].sum()
        grouped=grouped.reset_index()
        grouped=normalize_mrl(grouped)
    else:
        grouped = df.groupby(['codeMaille10Km'])[col_values].sum()
        grouped=grouped.reset_index()
    
    # Convertir l'objet Series résultant en DataFrame
    df=pd.merge(df_geo,grouped,on=["codeMaille10Km"],how='left')

    sum_values_df = df.reset_index()
    gdf = gpd.GeoDataFrame(sum_values_df)


    # Créer une nouvelle figure avec deux axes
    fig, ax1 = plt.subplots( figsize=(15,9))
    ax1.set_xlim(-0.8e6, 1.15e6)
    ax1.set_ylim(5e6, 6.75e6)

    # Choisir un système de coordonnées projetées approprié (par exemple, EPSG:3857 pour les coordonnées Web Mercator)
    gdf = gdf.to_crs(epsg=3857)

    liste_NA=gdf[gdf[col_values].isna()]['codeMaille10Km']
    gdf_NA=gdf[gdf['codeMaille10Km'].isin(liste_NA)]
    gdf=gdf[~gdf['codeMaille10Km'].isin(liste_NA)]
    
    ax1.axis('off')

    # Déterminer les valeurs minimales et maximales de, l
    val_min = round_to_sig((n_minimum(gdf[col_values], n)), direction='down')
    val_max = round_to_sig((n_maximum(gdf[col_values], m)), direction='up')
    quart_val=val_min+(val_max-val_min)/4
    demi_val=val_min+(val_max-val_min)*2/4
    troisquart_val=val_min+(val_max-val_min)*3/4

    # Tracer la carte des départements en utilisant la colonne var pour la coloration
    if not gdf.empty :
        gdf.plot(column=col_values, cmap=cmap_choice, legend=False, alpha=0.75,
             linewidth=0.1, edgecolor='gray',ax=ax1,vmin=val_min,vmax=val_max)
    if not gdf_NA.empty :
        gdf_NA.plot(color='lightgray', alpha=0.2,ax=ax1,vmin=val_min,vmax=val_max)

    # Charger les données géographiques des départements
    bioregion_gfp = r'C:\Users\anormand\Documents\Projet Python\Biodiv\Data\map\region_biogeographique.shp'  # Remplace par le chemin vers ton fichier
    bioregion_gdf = gpd.read_file(bioregion_gfp)
    bioregion_gdf = bioregion_gdf.to_crs(epsg=3857)
    
    # Tracer les départements en premier
    #bioregion_gdf.plot(ax=ax1, edgecolor='white', facecolor='none', linewidth=0.5, linestyle='--')

    basemaps = {
    'OpenStreetMap': ctx.providers.OpenStreetMap.Mapnik,
    'CartoDB': ctx.providers.CartoDB.Positron,
    'Esri': ctx.providers.Esri.WorldImagery,
    'NASAGIBS': ctx.providers.NASAGIBS.BlueMarble,
    'GeoportailSatellite': ctx.providers.GeoportailFrance.orthos,
    'GeoportailDepartements': ctx.providers.GeoportailFrance.parcels,
    'GeoportailRegion': ctx.providers.GeoportailFrance.Cartes_Naturalearth,
    }

    # Ajouter le fond de carte choisi
    if basemap_type in basemaps:
        ctx.add_basemap(ax1, source=basemaps[basemap_type],zoom=zoom_map)
    else:
        print(f"Fond de carte '{basemap_type}' non reconnu. Utilisation du fond par défaut OpenStreetMap.")
        ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik,zoom=6)
    
    # Ajouter une colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_choice,norm=plt.Normalize(vmin=val_min, vmax=val_max))
    extend_param = 'both' if m > 1 and n > 1 else 'max' if m > 1 else 'min' if n > 1 else 'neither'
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.6,ticks=[val_min,round_to_sig(quart_val, direction='nearest'),round_to_sig(demi_val, direction='nearest'),round_to_sig(troisquart_val, direction='nearest'),val_max],extend=extend_param)  # Réduire la taille de la colorbar
    cbar.ax.tick_params(labelsize=10)  # Augmenter la taille de la police de la colorbar
    cbar.set_label(col_values,fontsize=12)
    cbar.ax.yaxis.set_label_position('left') 
    
    if title is not None:
        ax1.set_title(title,fontsize=20,)
    else:
        title=col_values
        ax1.set_title(title,fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path+'/'+title+'.png')
        
    # Afficher la figure
    plt.show()


def afficher_carte_cluster(df_cluster,df_geo,col_values='Cluster',cmap_choice='plasma',title=None,save_path=None,basemap_type='OpenStreetMap',val_alpha=0.75):
    df=pd.merge(df_geo,df_cluster,on=["codeMaille10Km"],how='left')
    
    gdf = gpd.GeoDataFrame(df)

    # Créer une nouvelle figure avec deux axes
    # Créer une figure avec deux sous-graphiques
    fig, ax1 = plt.subplots( figsize=(15,9))
    ax1.set_xlim(-0.8e6, 1.15e6)
    ax1.set_ylim(5e6, 6.75e6)

    # Choisir un système de coordonnées projetées approprié (par exemple, EPSG:3857 pour les coordonnées Web Mercator)
    gdf = gdf.to_crs(epsg=3857)
    
    liste_NA=gdf[gdf[col_values].isna()]['codeMaille10Km']
    gdf_NA=gdf[gdf['codeMaille10Km'].isin(liste_NA)]
    gdf=gdf[~gdf['codeMaille10Km'].isin(liste_NA)]
    
    ax1.axis('off')
    # Déterminer les valeurs minimales et maximales de, l
    val_min = min(gdf[col_values])
    val_max = max(gdf[col_values])
    
    # Tracer la carte des départements en utilisant la colonne var pour la coloration
    if not gdf.empty :
        gdf.plot(column=col_values, cmap=cmap_choice, legend=False, alpha=val_alpha,linewidth=0.1, edgecolor='gray', ax=ax1)
    if not gdf_NA.empty :
        gdf_NA.plot(color='lightgray',  alpha=0.2, ax=ax1)

    # Charger les données géographiques des départements
    bioregion_gfp = r'C:\Users\anormand\Documents\Projet Python\Biodiv\Data\map\region_biogeographique.shp'  # Remplace par le chemin vers ton fichier
    bioregion_gdf = gpd.read_file(bioregion_gfp)
    bioregion_gdf = bioregion_gdf.to_crs(epsg=3857)
    
    # Tracer les départements en premier
     #bioregion_gdf.plot(ax=ax1, edgecolor='white', facecolor='none', linewidth=0.5, linestyle='--')

    basemaps = {
    'OpenStreetMap': ctx.providers.OpenStreetMap.Mapnik,
    'CartoDB': ctx.providers.CartoDB.Positron,
    'Esri': ctx.providers.Esri.WorldImagery,
    'NASAGIBS': ctx.providers.NASAGIBS.BlueMarble,
    'GeoportailSatellite': ctx.providers.GeoportailFrance.orthos,
    'GeoportailDepartements': ctx.providers.GeoportailFrance.parcels,
    'GeoportailRegion': ctx.providers.GeoportailFrance.Cartes_Naturalearth,
    }

    # Ajouter le fond de carte choisi
    if basemap_type in basemaps:
        ctx.add_basemap(ax1, source=basemaps[basemap_type],zoom=6)
    else:
        print(f"Fond de carte '{basemap_type}' non reconnu. Utilisation du fond par défaut OpenStreetMap.")
        ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik,zoom=6)
    
    values = df_cluster[col_values].unique()
    values=np.sort(values)

    values_color = [sorted(values).index(x) + 1 for x in values]
    
    # Obtenir la colormap
    cmap = plt.get_cmap(cmap_choice)
    
    # Calculer les positions des couleurs dans la colormap
    positions = np.linspace(0, 1, len(values))

    # Extraire les couleurs aux positions spécifiées
    colors = [cmap(pos) for pos in positions]
    
    patches = []
    # Creating legend with color box 
    for value in values_color:
        
        patch= mpatches.Patch(color=colors[value-1], label=f'Cluster {values[value-1]}') 
        patches.append(patch)

    # Afficher la légende avec tous les patches
    ax1.legend(handles=patches,loc='upper left', bbox_to_anchor=(0.03, 0.55),fontsize=12)
    
    if title is not None:
        ax1.set_title(title,fontsize=20)
    else:
        title=col_values
        ax1.set_title(title,fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path+'/'+title+'.png')
        
    # Afficher la figure
    plt.show()

def afficher_carte_maille(gdf):

    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Créer une carte centrée sur la zone des géométries
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=5)
    
    # Ajouter les géométries au folium map avec popup (pour afficher le code au clic)
    for _, row in gdf.iterrows():
        # Convertir la géométrie en GeoJSON
        geojson = folium.GeoJson(
            row['geometry'],
            style_function=lambda x: {'fillColor': 'white', 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.2}
        )
        # Ajouter le popup ou tooltip pour le code
        geojson.add_child(folium.Popup(f"Code: {row['codeMaille10Km']}"))  # Affiche lors du clic
        geojson.add_child(folium.Tooltip(f"Code: {row['codeMaille10Km']}"))  # Affiche au survol de la souris
        geojson.add_to(m)
    
    # Afficher la carte
    return m


def afficher_carte_observations_espece_manquante(df,df_avec,df_geo,col_values='nombreObs',n=1,m=1,cmap_choice='viridis',color_present='orange',
                                                 title=None,save_path=None,basemap_type='OpenStreetMap',zoom_map=6):
    
    # Regrouper les données par 'codeMaille10Km' et 'geometry' et calculer la somme des observations
    grouped = df.groupby(['codeMaille10Km'])[col_values].sum()
    grouped=grouped.reset_index()
  
    # Convertir l'objet Series résultant en DataFrame
    df=pd.merge(df_geo,grouped,on=["codeMaille10Km"],how='left')
    #df[col_values] = df[col_values].fillna(0)
    sum_values_df = df.reset_index()
    gdf = gpd.GeoDataFrame(sum_values_df)

    grouped_avec =df_avec.groupby(['codeMaille10Km'])[col_values].sum()
    grouped_avec=grouped_avec.reset_index()
    # Convertir l'objet Series résultant en DataFrame
    df_avec=pd.merge(df_geo,grouped_avec,on=["codeMaille10Km"],how='right')
    #df_avec[col_values] = df_avec[col_values].fillna(0)
    sum_values_df_avec = df_avec.reset_index()
    gdf_avec = gpd.GeoDataFrame(sum_values_df_avec)

    # Créer une nouvelle figure avec deux axes
    fig, ax1 = plt.subplots( figsize=(15,9))
    ax1.set_xlim(-0.8e6, 1.15e6)
    ax1.set_ylim(5e6, 6.75e6)

    # Choisir un système de coordonnées projetées approprié (par exemple, EPSG:3857 pour les coordonnées Web Mercator)
    gdf = gdf.to_crs(epsg=3857)
    gdf_avec = gdf_avec.to_crs(epsg=3857)

    liste_NA=gdf[gdf[col_values].isna()]
    gdf_NA=gdf[gdf['codeMaille10Km'].isin(liste_NA)]
    gdf=gdf[~gdf['codeMaille10Km'].isin(liste_NA)]

    ax1.axis('off')

    # Déterminer les valeurs minimales et maximales de, l
    val_min = round_to_sig((n_minimum(gdf[col_values], n)), direction='down')
    val_max = round_to_sig((n_maximum(gdf[col_values], m)), direction='up')
    quart_val=val_min+(val_max-val_min)/4
    demi_val=val_min+(val_max-val_min)*2/4
    troisquart_val=val_min+(val_max-val_min)*3/4

    # Tracer la carte des départements en utilisant la colonne var pour la coloration
    if not gdf.empty :
        gdf.plot(column=col_values, cmap=cmap_choice, legend=False, alpha=0.8,
             linewidth=0.1, edgecolor='gray',ax=ax1,vmin=val_min,vmax=val_max)
    if not gdf_NA.empty :
        gdf_NA.plot(color='lightgray', alpha=0.2,ax=ax1,vmin=val_min,vmax=val_max)

    gdf_avec.plot(column=col_values, color=color_present, legend=False, alpha=0.6,
             linewidth=0.1, edgecolor='gray',ax=ax1)

    # Charger les données géographiques des départements
    bioregion_gfp = r'C:\Users\anormand\Documents\Projet Python\Biodiv\Data\map\region_biogeographique.shp'  # Remplace par le chemin vers ton fichier
    bioregion_gdf = gpd.read_file(bioregion_gfp)
    bioregion_gdf = bioregion_gdf.to_crs(epsg=3857)
    
    # Tracer les départements en premier
    #bioregion_gdf.plot(ax=ax1, edgecolor='white', facecolor='none', linewidth=0.5, linestyle='--')

    basemaps = {
    'OpenStreetMap': ctx.providers.OpenStreetMap.Mapnik,
    'CartoDB': ctx.providers.CartoDB.Positron,
    'Esri': ctx.providers.Esri.WorldImagery,
    'NASAGIBS': ctx.providers.NASAGIBS.BlueMarble,
    'GeoportailSatellite': ctx.providers.GeoportailFrance.orthos,
    'GeoportailDepartements': ctx.providers.GeoportailFrance.parcels,
    'GeoportailRegion': ctx.providers.GeoportailFrance.Cartes_Naturalearth,
    }

    # Ajouter le fond de carte choisi
    if basemap_type in basemaps:
        ctx.add_basemap(ax1, source=basemaps[basemap_type],zoom=zoom_map)
    else:
        print(f"Fond de carte '{basemap_type}' non reconnu. Utilisation du fond par défaut OpenStreetMap.")
        ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik,zoom=6)
    
    # Ajouter une colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_choice,norm=plt.Normalize(vmin=val_min, vmax=val_max))
    extend_param = 'both' if m > 1 and n > 1 else 'max' if m > 1 else 'min' if n > 1 else 'neither'
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.6,ticks=[val_min,round_to_sig(quart_val, direction='nearest'),round_to_sig(demi_val, direction='nearest'),round_to_sig(troisquart_val, direction='nearest'),val_max],extend=extend_param)  # Réduire la taille de la colorbar
    cbar.ax.tick_params(labelsize=10)  # Augmenter la taille de la police de la colorbar
    cbar.set_label(col_values,fontsize=12)
    cbar.ax.yaxis.set_label_position('left') 
    patches = []
    patch= mpatches.Patch(color=color_present, label='Espèce recensée')
    patches.append(patch)
    ax1.legend(handles=patches,loc='upper right',fontsize=12)
    
    if title is not None:
        ax1.set_title(title,fontsize=20,)
    else:
        title=col_values
        ax1.set_title(title,fontsize=20)
        
    if save_path is not None:
        plt.savefig(save_path+'/'+title+'.png')
        
    # Afficher la figure
    plt.show()
    

def n_minimum(liste, n):
    # Si la liste est vide ou si n est plus grand que la taille de la liste, retourner None
    if len(liste) < n:
        return None
    
    # Initialiser une liste pour stocker les n premiers minimums
    minimums = [float('inf')] * n
    
    # Parcourir la liste pour trouver les n premiers minimums
    for element in liste:
        for i in range(n):
            if element < minimums[i]:
                minimums.insert(i, element)
                minimums.pop()
                break
    
    # Retourner le n-ième minimum trouvé
    return minimums[n-1]
    

def n_maximum(liste, n):
    # Si la liste est vide ou si n est plus grand que la taille de la liste, retourner None
    if len(liste) < n:
        return None
    
    # Initialiser une liste pour stocker les n premiers maximums
    maximums = [-float('inf')] * n
    
    # Parcourir la liste pour trouver les n premiers maximums
    for element in liste:
        for i in range(n):
            if element > maximums[i]:
                maximums.insert(i, element)
                maximums.pop()
                break
    
    # Retourner le n-ième maximum trouvé
    return maximums[n-1]