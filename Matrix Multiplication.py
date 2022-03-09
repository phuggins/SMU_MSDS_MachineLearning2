
#%% 
# ^Package Installs
import numpy as np
import re as re
from statistics import mean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#%%

#^ You asked your 10 work friends to answer a survey. They gave you back the following dictionary object. 
people = {'Paul': {'travel': 3,
                  'experience': 7,
                  'cost': 6,
                  'indian': 3,
                  'mexican': 6,
                  'hipster': 1,
                  'vegetarian': 1 },
        'Terry': {'travel': 5,
                  'experience': 6,
                  'cost': 3,
                  'indian':2,
                  'mexican':7,
                  'hipster':1,
                  'vegetarian':9 },
        'Chad': {'travel': 3,
                  'experience': 5,
                  'cost':7,
                  'indian':3,
                  'mexican': 2,
                  'hipster': 1,
                  'vegetarian': 4},
        'Mark': {'travel':1,
                  'experience':2,
                  'cost':3,
                  'indian':4,
                  'mexican':5,
                  'hipster':6,
                  'vegetarian': 7,} ,
        'Kevin': {'travel':9,
                  'experience':8,
                  'cost':7,
                  'indian':6,
                  'mexican':5,
                  'hipster':4,
                  'vegetarian':3 } ,
        'Blake': {'travel':1,
                  'experience':3,
                  'cost':5,
                  'indian':7,
                  'mexican':9,
                  'hipster':2,
                  'vegetarian':4 } ,
        'Jeff': {'travel':2,
                  'experience':4,
                  'cost':6,
                  'indian':8,
                  'mexican':10,
                  'hipster':1,
                  'vegetarian':3 } ,
        'Nikita': {'travel':1,
                  'experience':4,
                  'cost':7,
                  'indian':2,
                  'mexican':5,
                  'hipster':8,
                  'vegetarian': 10} ,
        'Jolie': {'travel':7,
                  'experience':5,
                  'cost':3,
                  'indian':1,
                  'mexican':5,
                  'hipster':8,
                  'vegetarian':2 } ,
        'Cara': {'travel':7,
                  'experience':8,
                  'cost':5,
                  'indian':4,
                  'mexican':4,
                  'hipster':1,
                  'vegetarian':6 }}


#%%
#^ Transform the user data into a matrix(M_people). Keep track of column and row ids.  

# find all the columns and all the rows, sort them    
# columns = sorted(set(key for dictionary in people.values() for key in dictionary))
# rows = sorted(people)
columns = set(key for dictionary in people.values() for key in dictionary)
rows = people

# figure out how wide each column is
col_width = max(max(len(thing) for thing in columns),
                    max(len(thing) for thing in rows)) + 1

# preliminary format string : one column with specific width, right justified
fmt = '{{:>{}}}'.format(col_width)

# format string for all columns plus a 'label' for the row
fmt = fmt * (len(columns) + 1)

# print the header
print(fmt.format('', *columns))

# print the matrix
for row in rows:
    dictionary = people[row]
    s = fmt.format(row, *(dictionary.get(col, 'inf') for col in columns))
    print(s)

# save out matrix
values=[]
for row in rows:
    for col in columns:
        values.append(people[row].get(col, 'inf'))

def slice_per(source, step):
    return [source[i::step] for i in range(step)]

peeps = slice_per(values,7)
peep_matrix = np.asarray(peeps)
peep_matrix = peep_matrix.transpose()

# print out matrix
print("Matrix")
print(peep_matrix)

# data checks
print("Matrix Length:")
print(len(peep_matrix))
print("Matrix Size:")
print(peep_matrix.shape)
print("Matrix Data Types:")
print(peep_matrix.dtype)
print("Matrix Type:")
print(type(peep_matrix))


# %%
#^ Now for restaurants
restaurants  = {'flacos':{'distance' : 1,
                        'novelty' :2,
                        'cost': 3,
                        'average rating':4, 
                        'cuisine':5,
                        'vegetarians':6,
                        'appetizers': 3},
                'tacos':{'distance' : 7,
                        'novelty' :8,
                        'cost': 9,
                        'average rating':10, 
                        'cuisine':9,
                        'vegetarians':8,
                        'appetizers': 4},
                'chickens':{'distance' : 7,
                        'novelty' :6,
                        'cost': 5,
                        'average rating':4, 
                        'cuisine':3,
                        'vegetarians':2,
                        'appetizers': 5},
                'braums':{'distance' : 1,
                        'novelty' :2,
                        'cost': 3,
                        'average rating':4, 
                        'cuisine':5,
                        'vegetarians':6,
                        'appetizers': 6},
                'mcdonalds':{'distance' : 7,
                        'novelty' :8,
                        'cost': 9,
                        'average rating':10, 
                        'cuisine':9,
                        'vegetarians':8,
                        'appetizers': 7},
                'wendys':{'distance' : 7,
                        'novelty' :6,
                        'cost': 5,
                        'average rating':4, 
                        'cuisine':3,
                        'vegetarians':2,
                        'appetizers': 8},
                'ubereats':{'distance' : 1,
                        'novelty' :3,
                        'cost': 5,
                        'average rating':7, 
                        'cuisine':9,
                        'vegetarians':6,
                        'appetizers': 9},
                'lardos':{'distance' : 8,
                        'novelty' :4,
                        'cost': 2,
                        'average rating':1, 
                        'cuisine':5,
                        'vegetarians':3,
                        'appetizers': 10},
                'pancakehouse':{'distance' : 8,
                        'novelty' :7,
                        'cost': 8,
                        'average rating':6, 
                        'cuisine':3,
                        'vegetarians':4,
                        'appetizers': 1},
                'chickenstwin':{'distance' : 5,
                        'novelty' :6,
                        'cost': 8,
                        'average rating':8, 
                        'cuisine':8,
                        'vegetarians':7,
                        'appetizers': 2}}

#%%
#^ Transform the restaurant data into a matrix(M_resturants) use the same column index.

# find all the columns and all the rows, sort them    
columns = set(key for dictionary in restaurants.values() for key in dictionary)
rows = restaurants

# figure out how wide each column is
col_width = max(max(len(thing) for thing in columns),
                    max(len(thing) for thing in rows)) + 1

# preliminary format string : one column with specific width, right justified
fmt = '{{:>{}}}'.format(col_width)

# format string for all columns plus a 'label' for the row
fmt = fmt * (len(columns) + 1)

# print the header
print(fmt.format('', *columns))

# print the matrix
for row in rows:
    dictionary = restaurants[row]
    s = fmt.format(row, *(dictionary.get(col, 'inf') for col in columns))
    print(s)

# save out matrix
values=[]
for row in rows:
    for col in columns:
        values.append(restaurants[row].get(col, 'inf'))

def slice_per(source, step):
    return [source[i::step] for i in range(step)]

rest = slice_per(values,7)
rest_matrix = np.asarray(rest)
rest_matrix = rest_matrix.transpose()

# print out matrix
print("Matrix")
print(rest_matrix)

# data checks
print("Matrix Length:")
print(len(rest_matrix))
print("Matrix Size:")
print(rest_matrix.shape)
print("Matrix Data Types:")
print(rest_matrix.dtype)
print("Matrix Type:")
print(type(rest_matrix))

#%%
#^ Informally describe what a linear combination is and how it will relate to our restaurant matrix.

# A linear combination is the combination of values from different matricies usually by multipication. In our example, the linear combination of people and restaurants can help us determine which restaurants are best suited for each person.


#%%
#^ Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent? 

# I chose myself.
paul = np.matrix(peep_matrix[0])

paul_result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*paul)] for X_row in rest_matrix]

paul_rest = np.sum(paul_result, axis=3)
print("Restaurant Rating for Paul")
print(np.squeeze(np.asarray(paul_rest)))

# each entry represents the multiplication product of my prefences multiplied by each of the restaurant descriptions to determine which restaurant is ideal for me. I actually have two values of 216, meaning that there was a tie in two restaurants. Looking at the index of the original restaurant matrix, those restaurants are 'tacos' and'mcdonalds'. Odd since those two are entirely different, but that's what you get when you make up data!

#%%
#^ Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

M_usr_x_rest = [[sum(b*a for b,a in zip(X_row,Y_col)) for Y_col in zip(*rest_matrix)] for X_row in peep_matrix]

print(np.matrix(M_usr_x_rest))

# This matrix represents each persons ranking of each of the individual restaurants.

#%%
#^ Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?

# This is the final ranking of the restaurants given all user data. The 5th restaurant is the top one (McDonalds).

sum_cols = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*rest_matrix)] for X_row in peep_matrix]

sum_cols = np.sum(M_usr_x_rest, axis=1)
print("Overall Restaurant Ratings")
print(np.squeeze(np.asarray(sum_cols)))

#%%
#^ Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.  

# row sums
rows = len(M_usr_x_rest) 
cols = len(M_usr_x_rest[0])

for i in range(0, rows):  
    sumRow = 0;  
    for j in range(0, cols):  
        sumRow = sumRow + M_usr_x_rest[i][j] 
    print("Sum of " + "#" + str(i+1) +" person: " + str(sumRow))  

M_usr_x_rest_rank = list(map(sum, M_usr_x_rest))
print(M_usr_x_rest_rank)


#%%
#^ Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?

# The main difference is between the two is that one is at a restaurant level and the other is at a person level. Both of them are valuable but in the restaurant result, the user preferences get lost as it's just a basic ranking of restaurants. In the person result, we need to see a bigger picture of the data or else it could be said that we're catering to only certain people.

# I actually got the same result for both.


#%%
#^ How should you preprocess your data to remove this problem. 

# I would normalize everything and potentially re-design it to get more of an equal variance. Depending on the inputs, you could normalize before generating linear combinations.


#%%
#^ Find user profiles that are problematic, explain why?

# I tried to spread out my values when I created the data. One of the potential issues could have been if someone put the same value for everything. Their profile would not really help the ranking system as the values are all equal. If any of the user ratings are outside of some acceptable bounds for the below metrics, I would consider evaulating their efficacy in the model.

avg_peep = np.mean(peep_matrix)
std_peep = np.std(peep_matrix)
var_peep = std_peep / avg_peep

print("Average User Rating: " + str("{:.3f}".format(avg_peep)))
print("Standard Deviation for User Ratings: " + str("{:.3f}".format(std_peep)))
print("Variance for User Ratings: " + str("{:.3f}".format(var_peep)))


#%%
#^ Think of two metrics to compute the disatistifaction with the group.  

# 1. variance between the best and worst ranked restaurant
# 2. cost is going to be the most prohibative factor so a metric around cost varaince would be useful.


#%%
#^ Should you split in two groups today? 

# Splitting into two groups might help a bit but I didn't notice a huge variance in the data. Depending on the personalities of the group, it might be easier to split on a different variable than the final ranking.

# Simple PCA to look at groups
pca = PCA(n_components = 5)
peeps_pca = pca.fit_transform(peep_matrix)
print(pca.explained_variance_)
# There is a huge drop off after 2 variances. We'll run with 2 for the KMeans

peep_kmeans = KMeans(n_clusters = 2, random_state = 123).fit(peeps_pca)
peep_groups = peep_kmeans.predict(peeps_pca)
peep_centers = peep_kmeans.cluster_centers_
peep_labels = peep_kmeans.labels_

print(peep_centers)
print(peep_labels)

labels0 = peeps_pca[peep_groups == 0]
labels1 = peeps_pca[peep_groups == 1]
plt.scatter(labels0[:,0], labels0[:,1], color = "red")
plt.scatter(labels1[:,0], labels1[:,1], color = "black")
plt.show()

# On first glance, these don't look to be the best clustering results but it would work. Since this HW isn't focused on optimizing Kmeans, we won't pursue cleaning this up.

#%%
#^ Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?

# You should adjust by adding a weight or scaling the cost variable. Since cost is now a non-variable in the matrix, it could even be removed.

# Removing the cost variable from the people and restaurant matrix
peep_matrix_no_cost = np.delete(peep_matrix,-1,1)
rest_matrix_no_cost = np.delete(rest_matrix,-1,1)

#peep_matrix_no_cost = np.matrix(peep_matrix_no_cost)

M_usr_x_rest_no_cost = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*rest_matrix_no_cost)] for X_row in peep_matrix_no_cost]

# row sums
rows = len(M_usr_x_rest_no_cost) 
cols = len(M_usr_x_rest_no_cost[0])

for i in range(0, rows):  
    sumRow = 0;  
    for j in range(0, cols):  
        sumRow = sumRow + M_usr_x_rest_no_cost[i][j] 
    print("Restaurant " + "#" + str(i+1) + ": "  + str(sumRow))  

M_usr_x_rest_rank2 = list(map(sum, M_usr_x_rest_no_cost))
print(M_usr_x_rest_rank2)

# Restaurant number 5 is the winner this time... which actually happens to still be McDonalds!

#%%
#^ Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 

# I'd bring in their rankings and weight it by some factor. Probably use factors such as the number of people on their team, similar preferneces, location and any other data I could gather to beef up the dataset.

# We'd also need their actual rankings for the data.. not just the final result of their ordering. We would need the same setup as what we currently have for the rankings.
