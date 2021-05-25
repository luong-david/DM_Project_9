#%%
import json
import mmap
import os
import time
#%%

rev_file = open('yelp_academic_dataset_review.json', 'r')
feature_file_name = 'feature_file_pt2.json'
feature_file = open(feature_file_name, 'w') #Create and overwrite....

#%%
start_time = time.time()
file_index = 1

#Skip ahead - come back and test this.
skip_ahead_index = 3000
for i in range(0, skip_ahead_index):
    bus_file.readline()


for line in bus_file:
    int_time = time.time()
    print('On file ', file_index)
    file_index+=1
    feature_vect = json.loads(line)
    bus_id = feature_vect['business_id']
    review_count = 1
    indices_to_remove = []
    s = mmap.mmap(rev_file.fileno(), 0, access=mmap.ACCESS_READ)
    

    loc = s.find(bytes(bus_id, "utf-8"),s.tell(), -1)
    while loc != -1:
        
        #Bring seek ptr to beginning of file:

        s.seek(loc - 1)
        test_string = s[s.tell(): s.tell() + 5].decode("utf-8")
        while s.tell() != 0 or '\n' not in test_string:
            s.seek(-1, os.SEEK_CUR)
            test_string = s[s.tell(): s.tell() + 5].decode("utf-8")
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
    print('Iteration took ', time.time() - int_time, ' seconds') 

print(time.time() - start_time)
