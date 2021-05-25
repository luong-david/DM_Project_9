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

bus_file = open('yelp_dataset/yelp_academic_dataset_business_100.json', 'r')

test_line = bus_file.readline()

matchObj = re.match(r'.+categories":"([^"]*)"', test_line)
if matchObj:
    matched_str = matchObj.group(1)
    print(matched_str)
    split_string = re.split(', |,| ', matched_str)
    word_list = []
    for word in split_string:
        clean_string = re.sub('\W+', '', word)
        word_list.append(clean_string.lower())
    print(word_list)
    
else:
    print('bad regex')


