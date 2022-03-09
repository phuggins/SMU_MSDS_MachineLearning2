#%% 
# Load Dataset and Packages
from re import X
from sklearn import datasets
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal 

# %% 
# Inspect the data
data = datasets.load_wine()
X = data.data
y = data.target

# Verify length and shape
print("Length of features: " + str(len(X)))
print("Shape of features: " + str(X.shape))
print("Length of target: " + str(len(y)))
print("Length of full dataset: " + str(y.shape))

# %% 

# Combine the X & y into one array
data = np.c_[X,y]

# Verify length and shape
print("Length of full dataset: " + str(len(data)))
print("Shape of full dataset: " + str(data.shape))

# For use later - count of each class
counts = np.bincount(y)
print('Total occurences of "class_0" in array: ', counts[0])
print('Total occurences of "class_1" in array: ', counts[1])
print('Total occurences of "class_2" in array: ', counts[2])
# %%
# Parameter Defintiion

# Number of Clusters. We know that there are three clusters in the wine dataset
k = 3
# Maximum number of iterations
max_iter = 100
# Number of data points
n = len(data)
# Degree of Fuzzification
m = 1.5

#%%
# Plot the data first (chose random features to look at)
plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,11],marker='o')
plt.axis('equal')
plt.xlabel('Alcohol', fontsize=16)
plt.ylabel('Hue', fontsize=16)
plt.title('Alcohol & Hue', fontsize=22)
plt.grid()
plt.show()


#%%
# Accuracy Function
def accuracy(cluster_labels, y):
    correct_pred = 0
    #print(cluster_labels)
    c0 = max(set(labels[0:59]), key=labels[0:59].count)
    c1 = max(set(labels[59:130]), key=labels[59:130].count)
    c2 = max(set(labels[130:]), key=labels[130:].count)
    
    for i in range(len(data)):
        if cluster_labels[i] == c0 and y[i] == 'class_0':
            correct_pred = correct_pred + 1
        if cluster_labels[i] == c1 and y[i] == 'class_1' and c1!=c0:
            correct_pred = correct_pred + 1
        if cluster_labels[i] == c2 and y[i] == 'class_2' and c2!=c1 and c2!=c0:
            correct_pred = correct_pred + 1
            
    accuracy = (correct_pred/len(data))*100
    return accuracy

# %%
# Create the membership matrix
def initializeMembershipMatrix():
    membership_mat = []
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        flag = temp_list.index(max(temp_list))
        for j in range(0,len(temp_list)):
            if(j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0
        membership_mat.append(temp_list)
    return membership_mat

membership_mat = initializeMembershipMatrix()

# %%
# Calculating the cluster center
def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(data[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers
    
calculateClusterCenter(membership_mat)

# %%
# Update membership values
def updateMembershipValue(membership_mat, cluster_centers): # Updating the membership value
    p = float(2/(m-1))
    for i in range(n):
        x = list(data[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat

# %%
# Get Clusters
def getClusters(membership_mat): # getting the clusters
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

# %%
def fuzzyCMeansClustering():
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc=[]
    while curr < max_iter:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        acc.append(cluster_labels)
        if(curr == 0):
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    print(np.array(membership_mat))
    return cluster_labels, cluster_centers, acc


# %%
labels, centers, acc = fuzzyCMeansClustering()
a = accuracy(labels, y)


# %%
c0_data = data[:,[0,11]]
c0_data = np.array(c0_data)

m1 = random.choice(c0_data)
m2 = random.choice(c0_data)
m3 = random.choice(c0_data)

cov1 = np.cov(np.transpose(c0_data))
cov2 = np.cov(np.transpose(c0_data))
cov3 = np.cov(np.transpose(c0_data))



# %%
x1 = np.linspace(10.5,16,50)  
x2 = np.linspace(1,5,500)
X, Y = np.meshgrid(x1,x2) 

Z1 = multivariate_normal(m1, cov1)  
Z2 = multivariate_normal(m2, cov2)
Z3 = multivariate_normal(m3, cov3)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y  

plt.figure(figsize=(8,6))
plt.scatter(c0_data[:,0], c0_data[:,1], marker='o')     
plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5) 
plt.axis('equal')                   
plt.xlabel('Alcohol', fontsize=16)
plt.ylabel('Hue', fontsize=16)
plt.title('Initial Random Clusters(Sepal)', fontsize=22)
plt.grid()
plt.show()




# %%

#finding mode
c0 = max(set(labels[0:58]), key=labels[0:58].count)
c1 = max(set(labels[58:129]), key=labels[58:129].count)
c2 = max(set(labels[129:]), key=labels[129:].count)

p_mean_clus1 = np.array([centers[c0][0],centers[c0][0]])
p_mean_clus2 = np.array([centers[c1][1],centers[c1][1]])
p_mean_clus3 = np.array([centers[c2][2],centers[c2][3]])

feats = data[:,[0,11]]

c0_data = feats[feats.index.isin(c0)]
c1_data = feats[feats.index.isin(c1)]
c2_data = feats[feats.index.isin(c2)]

values = np.array(y)

#search all 3 species
searchval_c0 = c0
searchval_c1 = c1
searchval_c2 = c2

#index of all 3 species
ii_c0 = np.where(values == searchval_c0)[0]
ii_c1 = np.where(values == searchval_c1)[0]
ii_c2 = np.where(values == searchval_c2)[0]
ind_c0 = list(ii_c0)
ind_c1 = list(ii_c1)
ind_c2 = list(ii_c2)

cov_c0 = np.cov(np.transpose(np.array(c0_data)))
cov_c1 = np.cov(np.transpose(np.array(c1_data)))
cov_c2 = np.cov(np.transpose(np.array(c2_data)))

feats_arr= np.array(feats)

x1 = np.linspace(0.5,7,150)  
x2 = np.linspace(-1,4,150)
X, Y = np.meshgrid(x1,x2) 

Z1 = multivariate_normal(p_mean_clus1, cov_c0)  
Z2 = multivariate_normal(p_mean_clus2, cov_c1)
Z3 = multivariate_normal(p_mean_clus3, cov_c2)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y   

plt.figure(figsize=(10,10))
plt.scatter(feats[:,0], feats[:,11], marker='o')     
plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5) 
plt.axis('equal')
plt.xlabel('Petal Length', fontsize=16)
plt.ylabel('Petal Width', fontsize=16)
plt.title('Final Clusters(Petal)', fontsize=22)
plt.grid()# displaying gridlines
plt.show()
# %%
