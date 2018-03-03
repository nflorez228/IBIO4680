def computeTextons(fim,k):
	import numpy as np
	from sklearn.cluster import KMeans
	d = np.product(np.array(fim).shape[:2])
	n = np.product(np.array(fim).shape[2:])
	data = np.zeros((d,n))
	count = 0
	for i in range(np.array(fim).shape[0]):
		for j in range(np.array(fim).shape[1]):
	  		data[count,:] = np.array(fim[i][j]).reshape(-1)
			count += 1

	kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(data.transpose()) #Ensuring KMeans has the same parameters as the Matlab function
	map = kmeans.labels_
	textons = kmeans.cluster_centers_
	w,h = np.array(fim[0][0]).shape
	map = map.reshape(w,h)

	return map, textons


