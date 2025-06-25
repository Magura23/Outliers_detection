from sklearn.decomposition import PCA

def pca_init(X, q  = 8):
   
	pca = PCA(n_components=q)
	
	scores = pca.fit_transform(X)  
	components = pca.components_  

    # n_genes = p
	W_e_init = components.T.copy()  # shape (p, q)


	
	W_d_init = components.copy()  # shape (q, p)
 
 
	return W_e_init, W_d_init  # numpy arrays
