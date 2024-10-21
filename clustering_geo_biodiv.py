# clustering_geo_biodiv.py 
# Cluster géographiques
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from fonctions_annexes_biodiv import dictionnaire_especes
from sklearn.cluster import KMeans


def PC_analysis(df,col_index='codeMaille10Km',col_values='nombreObs',clade='nomScientifique',n_pca=5):
    pivot_df = df.pivot_table(index=col_index, columns=clade, values=col_values, fill_value=0)
   
    # Supposons que pivot_df soit ton DataFrame à normaliser
    scaler = StandardScaler()
    #pivot_df = scaler.fit_transform(pivot_df)
    
    pca = PCA(n_components=n_pca)
    principal_components = pca.fit_transform(pivot_df)
    
    # Générer dynamiquement les noms des colonnes pour chaque composante
    column_names = [f'PC{i+1}' for i in range(n_pca)]
    
    # Créer un DataFrame avec les résultats de la PCA
    df_pca = pd.DataFrame(data=principal_components, columns=column_names)
    
    explained_variance = pca.explained_variance_ratio_
    print(f'Variance expliquée par chaque composante: {explained_variance}')
    
    # Créer un DataFrame à partir du tableau numpy
    df_with_code_maille = pd.DataFrame(pivot_df, columns=pivot_df.columns, index=pivot_df.index)
    
    # Réinitialiser l'index si nécessaire
    df_with_code_maille = df_with_code_maille.reset_index()

    # Extraire seulement la colonne col_index
    code_maille_column = df_with_code_maille[[col_index]]
    # Concaténer la colonne col_index avec df_pca
    df_pca = pd.concat([code_maille_column, df_pca], axis=1)
    df_pca.set_index(df_pca.columns[0], inplace=True)
    
    return df_pca

def clustering(df_pca,col_index='codeMaille10Km',method='kmeans',k_cluster=5):

    if method=='kmeans':
        # Étape 3 : Appliquer K-means avec le nombre de clusters choisi (par exemple k=3)
        kmeans = KMeans(n_clusters=k_cluster, random_state=42)
        clusters = kmeans.fit_predict(df_pca)
    if method=='ward':
        ward = AgglomerativeClustering(n_clusters=k_cluster, linkage='ward')
        clusters = ward.fit_predict(df_pca)
        """
    if method=='complete_linkage':
        complete_linkage = AgglomerativeClustering(n_clusters=k_cluster, linkage='complete')
        clusters = average_linkage.fit_predict(df_pca)
    if method=='single_linkage':
        single_linkage = AgglomerativeClustering(n_clusters=k_cluster, linkage='single')
        clusters = average_linkage.fit_predict(df_pca)
    if method=='average_linkage':
        average_linkage = AgglomerativeClustering(n_clusters=k_cluster, linkage='average')
        clusters = average_linkage.fit_predict(df_pca)
    if method=='DBSCAN':
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(df_pca)
        """
        
    # Étape 4 : Ajouter les clusters au DataFrame
    df_cluster=df_pca.copy()
    df_cluster['Cluster'] = clusters

    # Étape 1 : Compter le nombre d'occurrences de chaque cluster
    cluster_counts = df_cluster['Cluster'].value_counts()
    
    # Étape 2 : Trier les clusters par nombre d'occurrences de manière décroissante
    sorted_clusters = cluster_counts.index
    
    # Étape 3 : Créer un dictionnaire de mappage pour réorganiser les clusters
    cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(sorted_clusters, start=1)}
    
    # Réorganiser les clusters dans le DataFrame
    df_cluster['Cluster'] = df_cluster['Cluster'].map(cluster_mapping)
    
    # Compter le nombre de points dans chaque cluster
    cluster_counts = df_cluster['Cluster'].value_counts()
    
    # Afficher les résultats
    print(cluster_counts)
    
    # Afficher le DataFrame avec les clusters
    return(df_cluster)


def determine_k(df_pca,col_index='codeMaille10Km',k_cluster=5):
    
    # Étape 2 : Déterminer le nombre optimal de clusters (facultatif)
    # Utiliser l'inertie ou l'Elbow Method (méthode du coude) pour trouver le bon k
    
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_pca)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour déterminer k')
    plt.show()

def composition_cluster(df,df_cluster,col_value='nombreObs',n=5):

    df_cluster=df_cluster.reset_index()
    df_cluster=df_cluster[['codeMaille10Km','Cluster']]
    df_composition=pd.merge(df,df_cluster,on=["codeMaille10Km"],how='right')
    df_grouped=df_composition.groupby(["Cluster","nomScientifique"])[col_value].sum()
    df_grouped=df_grouped.reset_index()
    df_dico=dictionnaire_especes(df)
    
    # Tronquer la colonne nomVernaculaire à la première partie avant la virgule
    df_grouped=pd.merge(df_grouped,df_dico,on='nomScientifique')
    df_grouped['nomVernaculaire'] = df_grouped['nomVernaculaire'].str.split(',').str[0]
    
    # Calculer le total des observations pour chaque cluster
    df_grouped['totalObs'] = df_grouped.groupby('Cluster')[col_value].transform('sum')

    # Calculer le total des observations pour chaque cluster
    df_grouped['ObsUnique'] = 1
    
    # Calculer la colonne % composition
    df_grouped['% composition'] = round((df_grouped[col_value] / df_grouped['totalObs']) * 100,1)
    
    # Trier les espèces par nombre d'observations pour chaque cluster
    sorted_df = df_grouped.sort_values(['Cluster', col_value], ascending=[True, False])
    
    clusters = sorted_df['Cluster'].unique()

    # Afficher les 5 espèces les plus présentes pour chaque cluster
    for cluster in clusters:
        cluster_df = sorted_df[sorted_df['Cluster'] == cluster]
        top_5_df = cluster_df.head(n)
        
        print(f"Cluster {cluster}:")
        for _, row in top_5_df[['nomVernaculaire', 'nomScientifique', '% composition']].iterrows():
            print(f"{row['nomVernaculaire']:<20} {row['nomScientifique']:<20} {row['% composition']:>10}")
        print("\n")
        
    return sorted_df