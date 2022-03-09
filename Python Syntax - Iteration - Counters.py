#%%
#^ NOTE
#? Click the Run Below option (above) to run the entire py file and view the results in the interactive pane.
#? Ran at 11:12 PM on 5/11 with no errors. Completion time of 17 sec.


#%%
#^ Library Install
from collections import Counter
from itertools import *
from itertools import combinations
from itertools import islice
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
from string import punctuation


#^ Lists
#%%
print(dir(list))

#%%
#? Append
# Used to add a single item to the end of a list. 
# Example: x = [1,2,3,4,5]. What if we wanted to add a 6 to the end of the list? We’d simply do x.append(6). The result could be x = [1,2,3,4,5,6]. It is important to note that it writes this value back to the original list.
x = [1,2,3,4,5]
x.append(6)
print(x)

#%%
#? Extend
# Similar to append, but can add multiple items to the end of the list. 
# Example: x = [Hi, My, Name]. What if we wanted to put our name on the end of the list? y = [is, Paul, Huggins]. Now we can do x.extend(y) and the result would be [Hi, My, Name, is, Paul, Huggins].
x = ['Hi', 'My', 'Name']
y = ['is', 'Paul', 'Huggins']
x.extend(y)
print(x)

#%%
#? Index 
# Used to look up an item in a list and return the index position of that items.
# Example: x = [1,2,3,4,5]. What is the position of the 3? Index = x.index(3). The result would be 2.
x = [1,2,3,4,5]
index = x.index(3)
print(index)

#%%
#? Index 2
# This is just using the index function but applying search criteria on the index search. Useful when we want to search for a value after or between a particular index position.
# Example: x = [1,2,3,4,5,1,2,3,4,5]. What is the position of the 3 AFTER the 4th index is searched? Index = x.index(3,4). The result would be 7. We could also search for the 5 between the 3rd and 7th index. Index = x.index(5,3,7).
x = [1,2,3,4,5,1,2,3,4,5]
index = x.index(3,4)
print(index)

#%%
#? Insert
# Used to add either single or multiple items at a certain index (position). 
# Example: x = [1,2,3,7,8,9,10]. Clearly we’re missing 4-6 and we want to add those numbers after the 3. To do that we can set y = [4,5,6] and then do x.insert(3,y) which will return x = [1,2,3,[4,5,6],7,8,9,10].
x = [1,2,3,7,8,9,10]
y = [4,5,6]
x.insert(3,y)
print(x)

#%%
#? Remove
# Used to remove the 1st occurrence of the item called into the remove statement. The item is removed and not saved off anywhere. 
# Example: x = [1,2,3,4,5]. Let’s drop the 3. x.remove(3)
x = [1,2,3,4,5]
x.remove(3)
print(x)

#%%
#? Pop
# Similar to remove, but this takes into account the index of the item and can store it for later use.
# Example: x = [1,2,3,4,5]. Let’s pull out the 4 and save it somewhere else. Y = x.pop(3).
x= [1,2,3,4,5]
y = x.pop(3)
print(y)

#%%
#? Count
#This command counts the occurrences of the item specified.
# Example: x = [1,1,1,2,2,3,4,5,5,5,5]. How many 5’s are in the list? y = x.count(5)
x = [1,1,1,2,2,3,4,5,5,5,5]
y = x.count(5)
print(y)

#%%
#? Reverse
# Used to simply reverse the order of the items.
# Example: x = [1,2,3,4,5]. y = x.reverse(). The output will be [5,4,3,2,1].
x = [1,2,3,4,5]
x.reverse()
print(x)

#%%
#? Sort
# Used to sort the items in either ascending or descending order. The default is ascending order but descending can be activated by setting reverse=True within the sort() parameters.
# Example: x = [1,5,3,4,2]. y = x.sort(). The output will be [1,2,3,4,5]
x = [1,5,3,4,2]
x.sort()
print(x)

#%%
#? [1]+[1]
# This type of addition adds two numbers (currently in lists) together and puts them in a list. The output will be [1] + [1]
print([1]+[1])

#%%
#? [2]*2
# This creates a 2nd items inside the current [2] list. Resulting in [2,2] list
print([2]*2)

#%%
#? [1,2][1:]
# This pulls the 2nd column, all values. Resulting in a response of 2.
print([1,2][1:])

#%%
#? [x for x in [2,3]]
# This is a simple list comprehension that loop through all the characters in the list. This one will just print [2,3].
print([x for x in [2,3]])

#%%
#? [x for x in [1,2] if x ==1] 
# This loop prints the value when the item in the list equals 1 (== is the syntax for equals here)
print([x for x in [1,2] if x ==1])

#%%
#? [y*2 for x in [[1,2],[3,4]] for y in x] 
# This loop multiplies each value in the list by 2. ie 1*2 = 2, 2*2 = 4, 4*2 = 6, and 4*2 = 8. The y*2 indicates that y is the value in the list and 2 is the value to multiply it by. X is the list that y is referring to.
print([y*2 for x in [[1,2],[3,4]] for y in x])

#%%
#^ Tuples
print(dir(tuple))

#%%
#? Count
# This is very similar to the list count seen above. The function counts the occurrence of a particular value in the tuple.
# Example: x = (1,2,3,4,5). y = x.count(2), the output will be 1 since there is 1 occurrence.
x = (1,2,3,4,5)
y = x.count(2)
print(y)

#%%
#? Index
# This is very similar to the list tuple seen above. It is used to look up an item in a list and return the index position of that items.
# Example: x = (1,2,3,4,5). y = x.index(3), the output will be 2 since the number 2 is in the 3rd index spot.
x = (1,2,3,4,5)
index = x.index(3)
print(index)

#%%
#? Build a dictionary from tuples
# Example: x = ((‘one’, 1), (‘two’,2), (‘three’, 3)). Using the ‘dict’ function we can write these to a dictionary. y = dict((number, letter) for (number, letter) in x)
x = (('one', 1), ('two',2), ('three', 3))
y = dict((number, letter) for (number, letter) in x)
print(y)

#%%
#? Unpacking Tuples
# I haven’t unpacked many tuples in my day, but I generally use the variable method. Here we specify the fields inside of the tuple and then extract the value.
# Example: x = (‘Paul’, ‘Huggins’, 28). To unpack this we set the reverse of x by doing the following: (FirstName, LastName, Age) = x. Now if we call print(FirstName), we get the result ‘Paul’.
x = ("Paul", "Huggins", 28)
(FirstName, LastName, Age) = x
print(FirstName)

#%%
#^ Dictionaries
print(dir(dict))

#%%
#? Creating dictionaries
# Dictionaries are especially useful when dealing with key-value sets. In the a_dict scenario, the keys are ‘I hate’ and ‘You Should’, while the values are ‘you’ and ‘leave’.
# Example: x = {‘FirstName’ : ‘Paul’, ‘LastName’ : ‘Huggins’}. FirstName and LastName are the keys and ‘Paul’ and ‘Huggins’ are the values.
a_dict = {'I hate':'you', 'You should':'leave'}
x = {'FirstName' : 'Paul', 'LastName' : 'Huggins'}
print(x)

#%%
#? Keys
# Like discussed in the previous a_dict example, the keys() function will return a list of all the keys in the dictionary. Great for referencing or doing lookups.
# Example: x = {‘FirstName’ : ‘Paul’, ‘LastName’ : ‘Huggins’}. Calling keys() would return ([‘FirstName’, ‘LastName’})
x = {'FirstName' : 'Paul', 'LastName' : 'Huggins'}
print(x.keys())

#%%
#? Items
# This function returns the dictionaries key & value pairs. Notice how it does not return the ‘:’ originally specified in the dictionary, it returns commas, showing the pairs.
# Example: x = {‘FirstName’ : ‘Paul’, ‘LastName’ : ‘Huggins’}. Calling items() would return dict_items([(‘FirstName’, ‘Paul’), (‘LastName’, ‘Huggins’)])
x = {'FirstName' : 'Paul', 'LastName' : 'Huggins'}
print(x.items())

#%%
#? Values
# I assume we’re referring to the values() function here. This is going to tell us quickly and easily if a particular value is in the dictionary. This looks only at the values, not the keys.
# Example: x = {‘FirstName’ : ‘Paul’, ‘LastName’ : ‘Huggins’}. Calling value() with Paul as the parameter would return TRUE but if we use the parameter Age, it would return FALSE. 
x = {'FirstName' : 'Paul', 'LastName' : 'Huggins'}
x = 'Paul' in x.values()
print(x)

#%%
#? _key()
# ! Skip - see discord

#%%
#? Never in a dict
# ! TODODODODO

#%%
#? Del
# The delete statement deletes an element in the dictionary and cannot be saved to a new variable. The change is applied to the global dictionary.
# Example: x = {‘FirstName’ : ‘Paul’, ‘LastName’ : ‘Huggins’}. Calling del() on the key ‘FirstName’ will delete that key/value pair and leave ‘LastName’. 
x = {'FirstName' : 'Paul', 'LastName' : 'Huggins'}
del x['FirstName']
print(x)

#%%
#? Clear
# The clear statement removes everything from the dictionary. Obviously proceed with caution when doing this…
# Example: x = {‘FirstName’ : ‘Paul’, ‘LastName’ : ‘Huggins’}. Calling clear() will remove everything from the dictionary. 
x = {'FirstName' : 'Paul', 'LastName' : 'Huggins'}
x.clear()
print(x) # returns an empty dictionary

#%%
#^ Sets
print(dir(set))

#%%
#?Add
# Adds element to the list as long as it is not already in the set! If it is already present, it skips it.
x = {1,2,3,4,5}
x.add(5)
print(x)
x.add(6)
print(x)

#%%
#? Clear
# Clears all elements in the set and returns an empty set.
x.clear()

#%%
#? Copy
# Creates a copy of the set. If we simply use the ‘=’ sign we get  a shallow copy of the set… a change made to a ‘copy’ variable will also be made in the original one. It is best to use Copy unless the requirements are such that the original list needs to be updated with changes to the copied one.
x = {1,2,3,4,5}
x = x
x.add(6)
print(x)
print(x)

#%%
#? Difference
# Shows the difference in between two sets. This one is actually really neat, I hadn’t used this one before.
x = {1,2,3,4}
x = {2,3,5}
print(x.difference(x))

#%%
#? Discard
# Drops the specified value from the set. 
x = {1,2,3,4,5}
x.discard(4)
print(x)

#%%
#? Intersection
# Opposite of difference. This one shows the values that are common between sets.
x = {1,2,3,4,5}
x = {1,3,5}
print(x.intersection(x))

#%%
#? Issubset
# Evaluates all items in sets and results in a TRUE/FALSE response if all items are present in the set.
x = {1,2,3,4,5}
x = {1,2,3,4}
print(x.issubset(x))

#%%
#? Union
# Returns a set of unique elements from both sets.
x = {1,2,3,4,5}
x = {1,2,3,4}
print(x.union(x))

#%%
#? Update 
# Updates the set by adding items from other sets.
x = {1,2,3,4,5}
x = {1,2,3,4}
print(x.update(x))

#%%
#^ Strings
print(dir(str))

#%%
#? Capitalize
# Capitalizes the first character in the string and makes all the other characters lowercase
string = 'multi-arm bandit robot attack puppy'
print(string.capitalize())

#%%
#? Casefold
# converts everything to lowercase and symbols are converted to their caseless strings
string = 'I AM NOT GOOD AT LOOPS... Yet'
print(string.casefold())

#%%
#? Encode
# Encodes the string to various codes. Standard ones re utf-8 (default) and ascii.
string = 'pthon vs r'
print(string.encode())

#%%
#? Expandtabs
# Returns a copy with tab characters replaced bu whitespace characters.
# You can specify a tab index in the function to break out specific ones
string = 'Desktop\tSMU\tML2'
tabex = string.expandtabs(2)
print(tabex)

#%%
#? Find
# Returns the index of the 1st occurance of the string inside of the string. If the string is not inside, it returns a -1.
string = 'Well let me just quote the late-great Colonel Sanders, who said...Im too drunk to taste this chicken.'
print(string.find('Colonel Sanders'))
print(string.find('Ricky Bobby'))

#%%
#? Lstrip
# Remvoes spaces left of the string. There is also rstrip which does it on the right side of the string
string = '    Spaces are dumb      '
print(string.lstrip())
print(string.rstrip())

#%%
#? Partition
# Splits the string at the 1st occurance of the specified separator
string = 'separate on ME, not us'
print(string.partition('ME,'))

#%%
#? Split
# Splits the strng at the specified separator
string = 'No no dont, split me!'
print(string.split(','))

#%%
#? Swapcase
# Switches lowercase and uppercase letters
string = 'I WANT TO BE LOWERCASE and i want to be uppercase'
print(string.swapcase())

#%%
#? Zfill
# Pads the string with 0's to the left to make the string the specified length
string = 'Padme'
print(string.zfill(10))

#%%
#^ from collections import Counter
#? Stores elements and their counts in a dictionary. The element is the key and count is the value.
# The example will count and sort the counts of each value by numer.
# ie there are 5 number 2's and 1 number 5. So the 2's will show first in the dictionary with their count as the value.
CoutnerEx = [1,1,2,2,4,5,3,2,3,4,2,1,2,3]
print(Counter(CoutnerEx))

#%%
#^ from itertools import *
# itertools allow us to create iterators that are fast and less memory intensive than other methods.
# I am not as familiar with this topic but want to get better at it.

#%%
#? Count
# this function starts at 5 and and counts up 1 incriment at a time until it hits 13 and then stops. 
for i in count(5): # starting the count at 5
    if i > 13: # when less then 13, add 1
        break # stop at 13
    else:
        print(i) # print results

#%%
#? Cycle
# Simply put, this cycles through the elements until the break is specified
count = 0 # starting the count at 1
for item in cycle('BAM'): # set the string
    if count > 13: # cyle when count of cycles is 13
        break
    print(item) # print results
    count += 1  # add 1 to the cycle count each time

#%%
#? Combinations
# This one is neat!!! It generates all the combinations of the passed elememnts.
print(list(combinations('PYTHON', 2)))


#? I would have loved to spend a day going through all of these in great detail for the HW but I didn't have the time for it. I plan on doing it on my own time.

#%%
#^ Flower Orders 
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']

#%%
#? Build a counter object and use the counter and confirm they have the same values.
x = Counter(flower_orders)
print(sum(x.values())) # 183
print(len(flower_orders)) # 183


#%%
#? Count how many objects have color W in them.
print(sum('W' in x for x in flower_orders))


#%%
#? Make histogram of colors
color_count = [k for v in flower_orders for k in v.split('/')]
color_count = Counter(color_count)
plt.bar(color_count.keys(), color_count.values())
plt.show()


#%%
#? Rank the pairs of colors in each order regardless of how many colors are in an order.
#? FYI - I noticed that my code for this one and the next one are almost identical to Andy's during office hours. Great minds think alike! 
pairs = [k for v in flower_orders for k in combinations(v.split('/'),2)]
pc = Counter(pairs)
for i in pc:
    print(f'{i} : {pc[i]}')


#%%
#? Rank the triplets of colors in each order regardless of how many colors are in an order.
trips = [k for v in flower_orders for k in combinations(v.split('/'),3)]
tc = Counter(trips)
for i in tc:
    print(f'{i} : {tc[i]}')


#%%
#? Make dictionary color for keys and values are what other colors it is ordered with.
# Join everything
flower_orders_join = ','.join(flower_orders)
# Remove spaces
flower_orders_strip = flower_orders_join.replace(" ","")
# Remvoe punctuations
punc_ops = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
for ele in flower_orders_strip:
    if ele in punc_ops:
        flower_orders_strip = flower_orders_strip.replace(ele,'')
# Get a unique list of colors
unique_colors = set(flower_orders_strip)
print(unique_colors)
# initiate dictionary with unique_colors as the key and an empty set as the value to be populated
color_dict = {i : set() for i in unique_colors}
print(color_dict) # here's the source for creating the dict with empty set (https://stackoverflow.com/questions/49358963/how-do-i-initialize-a-dictionary-with-a-list-of-keys-and-values-as-empty-sets-in)

#^ Note for TA!
# Credit to Andy on this part. I was stuck on this part for way too long. Had no idea how to fill in the empty set that I created in the prior step. Below is his code with minor changes to finish it off.
for flower in flower_orders:
    flowerList = flower.split('/')
    for color in flowerList:
        flowerSet = set(flowerList)
        flowerSet.remove(color)
        color_dict[color] = flowerSet | color_dict[color]
for i,x in color_dict.items():
     print(f'{i} : {x}')


#%%
#? Make a graph showing the probability of having an edge between two colors based on how often they co-occur.  (a numpy square matrix)
# Re-used some code from prior questions to generate this
pairs = [k for v in flower_orders for k in combinations(v.split('/'),2)]
color_count = Counter(pairs)
color_sum = sum(color_count.values())
for value in color_count:
    color_count[value] = "{:.0%}".format(round(color_count[value]/color_sum,4))
print(color_count)


#%%
#? Make 10 business questions related to the questions we asked above.
# This question is too ambigious. Are we relating this to a business scneario? Or plain questions about the exact data above? Not sure where exactly you want this to go. My questions are based on the exact data above the some business questions about it.
# 1. What is the inter order ratio of flower types?
# 2. Obviously no date range for this data... so what are we comparing here? This could be daily, weekly, monthly, etc. and it would be nice to see trends.
# 3. Is there any specific reason that the orders are ordered in this particular way?
# 4. What is the use case for the pairs and triplets questions? Where is the ROI in a look like that?
# 5. What is the cost for each flower type? 
# 6. Are there bundling discounts?
# 7. Why are some types paired together more than other pairs?
# 8. Are there any clusters present in the data? Can we aggregate the types into more general types?
# 9. Is there a trend in the count of W's in the data as we progress through the list?
# 10. How does the counter function scale as the dataset gets larger?

#%%
#^ Dead Men
dead_men_tell_tales = ['Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']

#%%
#? Join everything
dead_men_join = ','.join(dead_men_tell_tales)
print(dead_men_join)


#%%
#? Remove spaces
dead_men_strip = dead_men_join.replace(" ","")
print(dead_men_strip)


#%%
#? Occurrence probabilities for letters
# ASSUMPTION: We don't care if the letter is uppercase or lowercase. I'm converting everything to lowercase.
# ASSUMPTION: We don't care about non-alphabetic items (dashes, periods, etc...). I'm removing them
# Trying to get a dict containing the 26 or less actual letter.
# Converting all to lowercase
dead_men_strip_lower = dead_men_strip.casefold()
# standard punctuation options to search for
punc_ops = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
for ele in dead_men_strip_lower:
    if ele in punc_ops:
        dead_men_strip_lower = dead_men_strip_lower.replace(ele,'')
print(dead_men_strip_lower)
# 1st get the count of each letter
letter_dict = dict(Counter(dead_men_strip_lower))
# sort by most frequent to least frequent
letter_dict = dict(sorted(letter_dict.items(), key=lambda x:x[1], reverse=True))
# sum of all letters
letter_sum = len(dead_men_strip)
# divide the count of each letter by the total number of letters and conver to percentages
for key in letter_dict:
    letter_dict[key] = '{:.2%}'.format((letter_dict[key] / letter_sum))
# Print results!
print(letter_dict)
# How about in a prettier format
for key, value in letter_dict.items():
    print(key, ' : ', value)


#%%
#? Tell me transition probabilities for every letter pairs
# Break out text into each letter
composite_list = [dead_men_strip_lower[x:x+1] for x in range(0, len(dead_men_strip_lower),1)]
# duplicate each letter to break up
k = 2 # repeat each letter twice
res =  [ele for ele in composite_list for i in range(k)]
# drop 1st item
res.pop(0)
# drop last item
res.pop()
# group into letter pairs
n = 2
final = [res[i * n:(i + 1) * n] for i in range((len(res) + n - 1) // n )]
final = [tuple(x) for x in final]
# intiate counter & sum length
dead_count = Counter(final)
dead_sum = sum(dead_count.values())
for values in dead_count:
    dead_count[values] = round(dead_count[values]/dead_sum,4)
    #dead_count[values] = "{:.0%}".format(round(dead_count[values]/dead_sum,4)) # can format to % but then that messes up the next chunk so I just made it a normal value.
print(dead_count)


#%%
#? Make a 26x26 graph of 4. in numpy
# list of alphabet
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# empty matrix to fill
empty_list = np.zeros((26,26))

#^ Note for TA!
# Went to office hours but couldn't get this one working on my own. Below is Andy's code and I'm not even sure if I have the right inputs for it. I knew I needed to create an empty array along with an alphabet list but couldn't get it across the finish line.
for i in range(len(alphabet_list)):
    for j in range(len(alphabet_list)):
        if (alphabet_list[i], alphabet_list[j]) in dead_count:
            empty_list[(i,j)] = dead_count[(alphabet_list[i], alphabet_list[j])]
empty_list = [row / row.sum() for row in empty_list if row.sum() > 0]
print(empty_list)