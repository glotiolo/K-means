import csv 
import numpy as np
from operator import itemgetter
import random


def feature_extract():
	feature_list = [np.zeros(16)] * 194
	with open('country.csv', newline='') as csvfile:
		reader = csv.reader(csvfile,delimiter = ",")
		read_list = list(reader)
		#print(read_list)
		labels = read_list[0]
		read_list = read_list[1:]
		country_list = []
		for i in range(len(read_list)):
			country_list.append(read_list[i][0])
			feature_list[i] = (np.array([read_list[i][1:]])).astype(dtype=float)
	return feature_list,country_list

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
	mu = random.sample(x,K)
	clust_list = [None] * m
	prev = [None] * m
	#print(" ")
	#print(" ".join(["k=",str(K),"n=",str(16),"m=",str(194)]))
	for j in range(max_iter):
		prev[:] = clust_list[:]
		for i in range(m):
			clust_list[i] = get_closest(x[i],mu)
		#print("Cost after cluster assignment: ",curr_cost(x,clust_list,mu))
		if clust_list == prev:
			break
		mu[:] = update_cluster(x,clust_list,K)
	#print("Cost after updating centroids: ",curr_cost(x,clust_list,mu))
	# print("converged after: " + str(j) + " iterations,")	
	return mu ,clust_list

X,countries = feature_extract() #0..m-1

def run(reps,K):
	min_cost = 1000000
	num_seen = 0
	for i in range(reps):
		mu,clust_list = K_means(X,K)
		cost = curr_cost(X,clust_list,mu)
		if cost < min_cost:
			num_seen = 0
			min_cost = cost
		if cost == min_cost:
			num_seen += 1
	print("k= " + str(K) + ", " + "cost="+ str(min_cost[0][0]) + " " + "(" + str(num_seen)+"/"+str(reps)+"), " + 
		"Cluster size of USA " + str(find_clust_size("United States of America",countries,clust_list)))

def find_clust_size(country,country_list,clust_list):
	country_index = country_list.index(country)
	print(country_index)
	cluster = clust_list[country_index]
	return clust_list.count(cluster)


           
def main():
	#K_means(x,1)
	run(100,3)
	#print(countries)





if __name__ == "__main__":

    main()