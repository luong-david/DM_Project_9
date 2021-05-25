'''
Turns out the files are too big to really aggregate. I think the best first step
is to aggregate only the restauraunts/bars. 

To do this we need some sort of groupings like
A) clustering on restauraunt types
B) kind of a guess on what things count as restaurants and which don't
'''


#Go through all lines and extract categories, this will be the feature vectors
#for the cluster anaysis. Can also further verify, by looking at number of times 
#'restauraunt' appears in each line. On first glance this looks like a pretty good metric

import os
import re

#Defn
def vectorize_cats(bus_line, word_set):
    #look for categories
    matchObj = re.match(r'.+categories":"([^"]*)"', bus_line)
    if matchObj:
        #get string
        matched_str = matchObj.group(1)
        #split on comma like things (also turn two word terms into own terms)
        split_string = re.split(', |,| ', matched_str)
        #remove special characters e.g. '(' ')'
        for word in split_string:
            clean_string = re.sub('\W+', '', word)
            word_set.add(clean_string.lower())
    else:
        print('no matches found')

#bus_file = open('yelp_dataset/yelp_academic_dataset_business_100.json', 'r')

#Develop dictionary
cat_dict = set()
line_no = 1
for line in bus_file:
    print('Line number: ', line_no)
    line_no+=1
    vectorize_cats(line, cat_dict)
#Now create sparse feature vectors for each item
cat_dict_sorted = sorted(cat_dict)
#for line in bus_file:





