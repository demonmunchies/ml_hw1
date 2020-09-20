# CS 7830 Assignment 1
from random import random
from math import sqrt
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pandas

# K Means Implementation
def my_k_means(num_clusters, data):

	# Initialize cluster starting locations (random)
	cluster_centers = []
	ind_lst = []
	for i in range(num_clusters):
		ind = int(random() * len(data))
		while(ind in index_lst):
			ind = int(random() * len(data))
		ind_lst.append(ind)
		cluster_centers.append(data[ind])


	# Initialize cluster sets
	cluster_labels = [-1 for _ in range(len(data))]

	prev_cost = 0

	# Repeat
	while True:

		# Assignment
		for i in range(len(data)):
			minim = float('inf')
			label = -1
			for j in range(num_clusters):
				dist = euclidean_distance(data[i], cluster_centers[j])
				if  dist < minim:
					label = j
					minim = dist
			cluster_labels[i] = label

		# Update
		upd_cluster_centers = [list_tuple_generator(len(data[0]), 0) for _ in range(num_clusters)]
		# print(upd_cluster_centers)
		for i in range(len(data)):
			for j in range(len(data[i])):
				upd_cluster_centers[cluster_labels[i]][j] = upd_cluster_centers[cluster_labels[i]][j] + data[i][j]
				if upd_cluster_centers[cluster_labels[i]][j] != upd_cluster_centers[cluster_labels[i]][j]:
					print('Panic!')
					break
			 
		for i in range(num_clusters):
			m = sum([1 for x in cluster_labels if x == i])

			try:
				for j in range(len(data[0])):
					upd_cluster_centers[i][j] = upd_cluster_centers[i][j] / m
			except:
				upd_cluster_centers[i][j] = random()
				# print('Empty cluster detected, crashing gracefully!')

		# Check 'stability'

		cost = 0
		for i in range(len(data)):
			eucl_dist = euclidean_distance(data[i], cluster_centers[cluster_labels[i]])
			cost = cost + eucl_dist
		cost = cost / len(data)
	
		if (abs(prev_cost - cost) / cost) < .001:
			break

		# print((abs(prev_cost - cost) / cost))

		prev_cost = cost

	return cluster_labels, cost, cluster_centers

# Fuzzy C Implementation
def my_fuzzy_c(num_clusters, data, m):
	# Initialize cluster starting locations (random)
	cluster_centers = []
	for i in range(num_clusters):
		cluster_centers.append(list_tuple_generator(len(data[0])))

	# Initialize membership matrix
	mem_mat = [[-1 for _ in range(num_clusters)] for _ in range(len(data))]
	# print(mem_mat)

	prev_cost = 0

	# Repeat
	count = 0
	while(True):
		# Update mem mat
		for i in range(len(mem_mat)):
			for j in range(len(mem_mat[i])):
				mem = 0
				for k in range(num_clusters):
					mem = mem + ((euclidean_distance(data[i], cluster_centers[j])/euclidean_distance(data[i], cluster_centers[k]))**(2/(m-1)))
				mem = 1 / mem
				mem_mat[i][j] = mem

		# for i in range(len(mem_mat)):
		# 	print(mem_mat[i])
		
		# Update clusters
		# For each cluster
		for j in range(num_clusters):
			upd_fuzz_cent = [0, 0, 0]
			uij = 0
			# For each data point
			for i in range(len(data)):
				# mul uij by xi
				# For each dim
				uij = uij + (mem_mat[i][j] ** m)
				for k in range(len(data[0])):
					upd_fuzz_cent[k] = upd_fuzz_cent[k] + ((mem_mat[i][j] ** m) * data[i][k])

			upd_fuzz_cent = [x / uij for x in upd_fuzz_cent]
			cluster_centers[j] = upd_fuzz_cent

		# print('==============================')

		# for i in range(len(cluster_centers)):
		# 	print(cluster_centers[i])

		# Find total cost
		cost = 0
		for i in range(len(mem_mat)):
			for j in range(num_clusters):
				cost = cost + ((mem_mat[i][j] ** m) * euclidean_distance(data[i], cluster_centers[j]))

		if (abs(prev_cost - cost) / cost) < .001:
			break

		# print((abs(prev_cost - cost) / cost))

		prev_cost = cost


	# for i in range(len(cluster_centers)):
	# 	print(cluster_centers[i])

	return mem_mat, cost, cluster_centers

def list_tuple_generator(dimensionality, val=None):
	t = []
	for x in range(dimensionality):
		if not val == None:
			t.append(val)
		else:
			t.append(random())
	return t

def euclidean_distance(point_1, point_2):
	dist = 0
	for x in range(len(point_1)):
		dist = dist + ((point_1[x] - point_2[x])**2)
	return sqrt(dist)

def k_means_execution(data, num_clusters, iters):

	cluster_labels_lst = []
	cost_lst = []
	clus_cent_lst = []

	for x in range(iters):
		clust_label, cost, cluster_centers = my_k_means(num_clusters, data)

		cluster_labels_lst.append(clust_label)
		cost_lst.append(cost)
		clus_cent_lst.append(cluster_centers)

	counter = -1
	min_val = float('inf')
	for i in range(len(cost_lst)):
		if cost_lst[i] < min_val:
			counter = i

	clust_label = cluster_labels_lst[counter]

	return clus_cent_lst[counter], clust_label, cost_lst[counter]

	# data_x = [i for i, j, k in data]
	# data_y = [j for i, j, k in data]
	# data_z = [k for i, j, k in data]
	# color = ['red' if x == 0 else 'orange' if x == 1 else 'yellow' if x ==2 else 'green' if x ==3 else 'blue' if x ==4  else 'black' for x in clust_label]

	# ax = plt.axes(projection ="3d")

	# ax.scatter(data_x, data_y, data_z, c=color)
	# plt.show()

def my_dunn_index(cluster_centers, cluster_labels, data):

	diam_lst = []
	dist_lst = []

	# Calculate diameters of cluster
	for k in range(len(cluster_centers)):
		clust_diam_lst = []
		for x in range(len(data)):
			for y in range(len(data)):
				if cluster_labels[x] == cluster_labels[y] and cluster_labels[x] == k:
					clust_diam_lst.append(euclidean_distance(data[x], data[y]))
		diam_lst.append(max(clust_diam_lst))
	max_diam = max(diam_lst)

	# Calculate min distance between points in different clusters
	for i in range(len(cluster_centers)):
		for j in range(i+1, len(cluster_centers)):
			clust_dist_lst = []
			for x in range(len(data)):
				for y in range(len(data)):
					if cluster_labels[x] == i and cluster_labels[y] == j:
						clust_dist_lst.append(euclidean_distance(data[x], data[y]))
			dist_lst.append(min(clust_dist_lst))
	min_dist = min(dist_lst)

	dunn_index = min_dist / max_diam

	return dunn_index

def fuzzy_c_execution(data, num_clusters, m, iters):

	mem_mat_lst = []
	cost_lst = []
	clus_cent_lst = []

	for x in range(iters):
		mem_mat, cost, cluster_centers = my_fuzzy_c(num_clusters, data, m)

		mem_mat_lst.append(mem_mat)
		cost_lst.append(cost)
		clus_cent_lst.append(cluster_centers)

	counter = -1
	min_val = float('inf')
	for i in range(len(cost_lst)):
		if cost_lst[i] < min_val:
			counter = i

	for x in clus_cent_lst[counter]:
		print(x)

def ingest(filename, drop_fields):
	df = pandas.read_csv(filename, usecols=drop_fields)
	df = df.apply(lambda x: x/x.max(), axis=0)
	# print(df)
	# print(df.dtypes)
	df = df.dropna()
	# print(df)
	return df.values.tolist()

# def __main__():

data = ingest('Assignment1_data.csv', ['Risk', 'NoFaceContact', 'Sick'])
# k_means_execution(data, 2, 500)
# fuzzy_c_execution(data, 5, 2, 50)

cost_lst = []
index_lst = []

for i in range(100):
	try:
		# print(i)
		cluster_centers, cluster_labels, cost = k_means_execution(data, 5, 500)
		index = my_dunn_index(cluster_centers, cluster_labels, data)
		cost_lst.append(cost)
		index_lst.append(index)
	except:
		# print('###################################')
		# print('Some error occured.')
		# print(cost)
		# print(cluster_centers)
		# print(cluster_labels)
		# print('###################################')
		error = -1

plt.scatter(cost_lst, index_lst)
plt.show()


