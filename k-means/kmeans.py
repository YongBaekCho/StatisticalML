import scipy.io
import numpy as np
import random
from matplotlib import pyplot as plt
from numpy import linalg

def init_center_randomly(data, k): #initialize centroids randomly
	centroids = list()
	fi_centroids = list()
	for i in range(k):
		a = random.randint(0,299)
		centroids.append(data[a])
		fi_centroids = np.array(centroids)
	return fi_centroids

def clustering(n, centroids):
	euclidean_distance = list()
	ll = []
	for new in centroids:
		norm = ((n[0] - new[0])**2 + (n[1] - new[1])**2)**1/2
		euclidean_distance.append(norm)
	final = np.argwhere(euclidean_distance == min(euclidean_distance))
	return final[0][0]
	
	
def distance(data, k): #calculate the distance for choosing initial centorids for strategy 2
	centroids = list()
	a = random.randint(0,299)
	centroids.append(data[a])
	for i in range(1, k):
		for j in range(300):
			for new in centroids:
				total_distance = ((data[j][0] - new[0])**2 + (data[j][1] - new[1])**2)
	
		np.array(centroids.append(data[j]))
	return centroids
	#print(len(centroids))

def squarederror(centroids, obj): #Find objective function value
	error = 0
	for val in obj:
		newone = centroids[val]
		for val1 in obj[val]:
			error += (val1[0] - newone[0])**2 + (val1[1] - newone[1])**2
	return error

def a_clustering(data, centroids): #finding clusters
	finalize = dict()
	while (True):
		clusters = dict()
		updated = list()
		for n in data:
			if clustering(n, centroids) in clusters:
				clusters[clustering(n, centroids)].append(n)
			elif clustering(n, centroids) not in clusters:
				clusters[clustering(n, centroids)] = [n]
			if i in clusters:
				cluster1 = np.array(clusters[i])
				mu = np.mean(cluster1, axis =0)
				updated.append(mu)
			else:
				updated.append(centroids[i])
		a = np.array(centroids)
		b = np.array(updated)
		if np.all(a==b):
			finalize = clusters
			break
		else:
			centroids = updated
	
	return centroids, finalize


Numpyfile= scipy.io.loadmat('AllSamples.mat')# Dataset(300, 2)
data = Numpyfile['AllSamples'] 


#Plotting the Points Initially
tdata = np.transpose(data)
plt.scatter(tdata[0],tdata[1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#Strategy 1 :iteration 1
cluster = []
error = []
k = 2
k1 = 11
for k in range(k, k1):
	centroids, final = a_clustering(data, init_center_randomly(data, k))
	
	cluster.append(k)
	error.append(squarederror(centroids, final))
print(cluster, error)
plt.plot(cluster, error, color = 'blue')
plt.title("Figure"+ '1' + 'Strategy 1: Iteration: 1')
plt.xlabel("Number of Clusters")
plt.ylabel("Objective Function")

plt.show()
#Strategy 1 :iteration 2
cluster = []
error = []
k = 2
k1 = 11
for k in range(k, k1):
	centroids, final = a_clustering(data, init_center_randomly(data, k))
	
	cluster.append(k)
	error.append(squarederror(centroids, final))
print(cluster, error)


plt.plot(cluster, error, color = 'blue')
plt.title("Figure "+ '2' + 'Strategy 1: Iteration: 2')
plt.xlabel("Number of Clusters")
plt.ylabel("Objective Function")

plt.show()

#strategy 2: iteration 1
cluster = []
error = []
k = 2
k1 = 11
for k in range(k, k1):
	centroids, final = a_clustering(data, distance(data, k))
	
	cluster.append(k)
	error.append(squarederror(centroids, final))
print(cluster, error)

plt.plot(cluster, error, color = 'blue')
plt.title("Figure "+ '3' +" Objective function & numbers of cluster(distance base) - iteration: 1")
plt.xlabel("Number of Clusters")
plt.ylabel("Value of Objective Function")

plt.show()
#strategy 2: iteration 2
cluster = []
error = []
k = 2
k1 = 11
for k in range(k, k1):
	centroids, final = a_clustering(data, distance(data, k))
	cluster.append(k)
	error.append(squarederror(centroids, final))
print(cluster, error)
#plot objective function vs number of cluster
plt.plot(cluster, error,color = 'blue')
plt.title("Figure "+ '4' +" Objective function & numbers of cluster(distance base) - iteration: 2")
plt.xlabel("Number of Clusters")
plt.ylabel("Value of Objective Function")

plt.show()

