import csv 
import numpy as np
from operator import itemgetter


def feature_extract():
	feature_list = [np.zeros(16)] * 194
	with open('country.csv', newline='') as csvfile:
		reader = csv.reader(csvfile,delimiter = ",")
		read_list = list(reader)
		labels = read_list[0]
		read_list = read_list[1:]
		for i in range(len(read_list)):
			feature_list[i] = (np.array([read_list[i][1:]])).astype(dtype=float)
	return feature_list


X = feature_extract() #0..m-1



def squared_Dist(x_vectors,mu_vectors):
	diff = x_vectors - mu_vectors
	return diff.dot(diff.transpose())

def get_closest(x_vectors,mu_vectors):
	comp_list = [squared_Dist(x_vectors,mu) for mu in mu_vectors]
	return min(enumerate(comp_list),key=itemgetter(1))[0]

def update_cluster(vectors , clust_list, K):
	mu = [np.zeros(16)]*K
	for k in range(K):
		indices = [i for i,index in enumerate(clust_list) if index == k]
		mu[k] = (np.add.reduce(list(map(lambda f: vectors[f],indices))))/len(indices)
	return mu
 
def curr_cost(vectors, clust_list, mu):
	m = len(vectors)
	cost = [np.zeros(1)]
	for i in range(m):
		cost += squared_Dist(vectors[i],mu[clust_list[i]])
	return cost/m


def K_means(x,K,max_iter = 500):
	m = len(x)
	mu = x[0:K] #0..k-1
	clust_list = [None] * m
	prev = [None] * m
	print(" ")
	print("k = " + str(K) + " " + "n= " + str(16) + " " + "m= " + str(194))
	for j in range(max_iter):
		prev[:] = clust_list[:]
		for i in range(m):
			clust_list[i] = get_closest(x[i],mu)
		print("Cost after cluster assignment: ",curr_cost(x,clust_list,mu))
		if clust_list == prev:
			break
		mu[:] = update_cluster(x,clust_list,K)
		print("Cost after updating centroids: ",curr_cost(x,clust_list,mu))
	print("converged after: " + str(j) + " iterations,")	




           
def main():
	x = feature_extract()
	K_means(x,3)





if __name__ == "__main__":

    main()