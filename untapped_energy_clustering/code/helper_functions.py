def plot_top_k(well_data_df, top_k_index = pd.Int64Index([1]),  
               target_index = pd.Int64Index([1])):
    """Highlight the top k wells with matplotlib.

    Keyword arguments:
    well_data_df -- the dataframe with the well data (black)
    top_k_index -- the index for the top k wells (red)
    target_index -- the index for the target well (blue)
    """
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(well_data_df.loc[target_index,:].mean_ss_easting,
               well_data_df.loc[target_index,:].mean_ss_northing, 
               zorder=1, alpha= 1, c='b', s=50)
    ax.scatter(well_data_df.drop(target_index).loc[top_k_index,:].mean_ss_easting,
               well_data_df.drop(target_index).loc[top_k_index,:].mean_ss_northing, 
               zorder=1, alpha= 1, c='r', s=50)
    ax.scatter(well_data_df.drop(top_k_index).mean_ss_easting,
               well_data_df.drop(top_k_index).mean_ss_northing, 
               zorder=1, alpha= 0.5, c='k', s=10)
    ax.set_title('Scatter Plot of Top' + k + 'Wells')
    
# make a kNN predict function
def predict_print_knn(well_data_df, dist_func, k , target_index, target_column):
    """Predict and print a target column using the knn method.

    Keyword arguments:
    well_data_df -- the dataframe with the well data (black)
    dist_func -- a scipy.spatial.distance function
    k -- the number of neighbours
    target_index -- row index of the well we are targetting
    target_column -- column of the target variable
    """
    well_data_df = well_data_df.assign(distance = well_data_df.apply(
        dist_func, axis=1, v=well_data_df.loc[target_index,:]))
    
    top_k_index = well_data_df.nsmallest(k,'distance').index
    
    # note the nested functions
    plot_top_n(well_data_df, top_k_index, target_index)
    
    # let's 'predict' the breakdown_isip_ratio - kinda a fluke
    pred = well_data_df.loc[top_k_index,target_column].mean()
    actual = well_data_df.loc[target_index,target_column]
    
    # print using f-strings
    print(title_str + ' kNN prediction: ' 
          + f'{pred:.3}' 
          + ' vs target: ' 
          + f'{actual:.3}')

# simple 2D colour plot
def plot_single_map(df, xyz_cols = ['x','y','first_order_residual']):
    """Plot a single map.

    Keyword arguments:
    df -- the dataframe with the well data 
    xyz_cols -- a vector of the x,y, and target column
    """
    fig, ax = plt.subplots(figsize = (5.5,5))
    ax.set_title(xyz_cols[2])
    ax.tripcolor(df[xyz_cols[0]], df[xyz_cols[1]], df[xyz_cols[2]])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_four_maps(df):
    """Plot a four maps based on index location, with 0 being an index
    column, 1=x, 2=y and 3=z.

    Keyword arguments:
    df -- the dataframe with the well data
    """
    plt.figure(figsize=(8, 8))
    G = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[1, 0])
    ax4 = plt.subplot(G[1, 1])
    
    ax1.set_title(df.columns[3])
    ax1.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,3])
    ax1.set_aspect('equal')

    ax2.set_title(df.columns[4])
    ax2.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,4])
    ax2.set_aspect('equal')
    
    ax3.set_title(df.columns[5])
    ax3.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,5])
    ax3.set_aspect('equal')
    
    ax4.set_title(df.columns[6])
    ax4.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,6])
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
def plot_pca_variance(pca):
    """Line plot of the pca variance ratio - individual and cummulative 

    Keyword arguments:
    pca -- pca object
    """
    plt.figure(figsize=(8, 5))
    G = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    
    ax1.set_title('Individual Variance')
    ax1.set_xlabel('PCA')
    ax1.plot(pca.explained_variance_ratio_, 'k-')
    ax1.plot(pca.explained_variance_ratio_, 'ro')

    ax2.set_title('Cummulative Variance')
    ax2.plot(np.cumsum(pca.explained_variance_ratio_), 'k-')
    ax2.plot(np.cumsum(pca.explained_variance_ratio_), 'ro')
    
    plt.tight_layout()
    plt.show()
       
def plot_pca_unit_circle(pca, df):
    """Plot pca correlation circle for first two PCAs

    Keyword arguments:
    pca -- pca object
    df -- the dataframe with the well data
    """
    coefficients = np.transpose(pca.components_)
    
    ex_var_ratio = pca.explained_variance_ratio_
    
    pca_cols = ['PC-'+str(x) for x in range(len(ex_var_ratio))]
    
    pca_info = pd.DataFrame(coefficients, 
                            columns=pca_cols, 
                            index=df.columns)
    
    plt.figure(figsize=(5.5, 5))
    plt.Circle((0,0),radius=10, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pca_info["PC-0"])):
        x = pca_info["PC-0"][idx]
        y = pca_info["PC-1"][idx]
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(pca_info.index[idx], xy=(x,y))
    plt.xlabel("PC-0 (%s%%)" % str(ex_var_ratio[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ex_var_ratio[1])[:4].lstrip("0."))
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations")
    plt.tight_layout()
    plt.show()
    
def plot_cluster_results(clust_res, df):
    """Scatter plot of cluster results, with xy of dataframe and
    cluster centres.

    Keyword arguments:
    clust_res -- the cluster result - should have cluster_centers_, labels_
    df -- the dataframe with the well data
    """
    plt.figure(figsize=(8, 5))
    G = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    
    clust_centres = pd.DataFrame(clust_res.cluster_centers_, 
                                 columns = df.columns)
    
    ax1.set_title('XY Scatter Plot')
    ax1.set_xlabel('PCA')
    ax1.scatter(df.loc[:,'mean_ss_easting'], 
                df.loc[:, 'mean_ss_northing'], c=clust_res.labels_)
    ax1.scatter(clust_centres.loc[:,'mean_ss_easting'], 
                clust_centres.loc[:, 'mean_ss_tvd'], c='red', s=300)
    
    ax2.set_title('XZ Scatter Plot')
    ax2.set_xlabel('PCA')
    ax2.scatter(df.loc[:,'mean_ss_easting'], 
                df.loc[:, 'mean_ss_tvd'], c=clust_res.labels_)
    ax2.scatter(clust_centres.loc[:,'mean_ss_easting'], 
                clust_centres.loc[:, 'mean_ss_tvd'], c='red', s=300)
    
    plt.tight_layout()
    plt.show()
    
def kmeans_elbowplot(df, kmax = 12):
    """Iterate the k-means algorith and produce an elbow plot to optimize 
    the number of clusters using the inertia_ attribute and the two sklearn
    unsupervised scoring metrics (silhouette and calinski harabaz)

    Keyword arguments:
    df -- the dataframe with the well data
    kmax -- the maximum number of clusters
    """
    wcss = []
    silhouette = []
    calinski_harabaz = []
    for i in range(2, kmax):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        ch_score = metrics.calinski_harabaz_score(df, kmeans.labels_)
        sil_score = metrics.silhouette_score(df, kmeans.labels_, metric='euclidean')
        silhouette.append(sil_score)
        calinski_harabaz.append(ch_score)
    
    plt.figure(figsize=(8, 5))
    G = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[0, 2])
    
    ax1.set_title('Intertia')
    ax1.plot(range(2, kmax), wcss, c='k')
    ax1.set_xticks(range(2, 12))
    ax1.set_xlabel('Clusters')
    
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Clusters')
    ax2.plot(range(2, kmax), silhouette, c='r')
    ax2.set_xticks(range(2, 12))

    ax3.set_title('Calinski Harabaz Score')
    ax3.set_xlabel('Clusters')
    ax3.plot(range(2, kmax), calinski_harabaz, c='g')
    ax3.set_xticks(range(2, 12))
    
    plt.tight_layout()
    plt.show()
    
def plot_dendrogram(df, method = 'ward'):
    """Plot the dendrogram for heirarchical clustering

    Keyword arguments:
    df -- the dataframe with the well data
    method -- agglomoerative method to calculate the linkage matrix
    """
    linkage_matrix = linkage(df, method)
    figure = plt.figure(figsize=(8, 8))
    dendrogram(
        linkage_matrix,
        color_threshold=0,
    )
    plt.title('Hierarchical Clustering Dendrogram (' + method + ')')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.show()
    
def print_clustering_scores(df, cluster_res):
    """Print clustering scores

    Keyword arguments:
    df -- the dataframe with the well data
    cluster_res -- cluster results from k_means or other
    """
    inertia = 'Inertia: ' + str(cluster_res.inertia_)
    print(inertia)
    ch_score = 'Calinski Harabaz: ' + str(metrics.calinski_harabaz_score(df, cluster_res.labels_))
    print(ch_score)
    sil_score = 'Silhouette: ' + str(metrics.silhouette_score(df, cluster_res.labels_, metric='euclidean'))
    print(sil_score)