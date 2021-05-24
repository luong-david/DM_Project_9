#%%
import json
import mmap
import os
import time
#%%
bus_file = open('../yelp_academic_dataset_business.json', 'r')
rev_file = open('../yelp_academic_dataset_review.json', 'r')
feature_file_name = 'feature_file_pt1.json'
feature_file = open(feature_file_name, 'w') #Create and overwrite....

#%%
start_time = time.time()
for i in range(0,5):
    feature_vect = json.loads(bus_file.readline())
    bus_id = feature_vect['business_id']
    review_count = 1
    indices_to_remove = []
    s = mmap.mmap(rev_file.fileno(), 0, access=mmap.ACCESS_READ)
    

    loc = s.find(bytes(bus_id, "utf-8"),s.tell(), -1)
    while loc != -1:
        
        #Bring seek ptr to beginning of file:
        s.seek(loc - 35)
        test_string = s[s.tell(): s.tell() + 35].decode("utf-8")
        while '\n' not in test_string:
            s.seek(-35, os.SEEK_CUR)
            test_string = s[s.tell(): s.tell() + 35].decode("utf-8")
        #Bring seek to beginning of line, by reading to end of 
        #current line
        nully = s.readline()
        rev = s.readline().decode("utf-8")

        rev_key = 'review_%s' % review_count
        review_count+=1
        feature_vect[rev_key] = json.loads(rev)
        #Repeat search
        loc = s.find(bytes(bus_id, "utf-8"),s.tell(), -1)
    with open(feature_file_name, 'a') as outfile:
        json.dump(feature_vect, outfile)
        print('\n', sep='', end='', file=outfile)
        

print(time.time() - start_time)
