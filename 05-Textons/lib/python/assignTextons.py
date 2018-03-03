def assignTextons(fim,textons):
	import numpy as np
	from distSqr import distSqr
	d = np.product(np.array(fim).shape[:2])
	n = np.product(np.array(fim).shape[2:])
	data = np.zeros((d,n))

	count = 0
	for i in range(np.array(fim).shape[0]):
		for j in range(np.array(fim).shape[1]):
	  		data[count,:] = np.array(fim[i][j]).reshape(-1)
			count += 1
			
	d2 = distSqr(np.array(data), np.array(textons));
	y = np.min(d2, axis=1)
	map = np.argmin(d2, axis=1)
	w,h = np.array(fim).shape[2:]
	map = map.reshape(w,h)
	return map
