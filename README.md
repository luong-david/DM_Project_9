# DM_Project_9
 Data Mining Project Group 9 (Yelp Dataset)

## Download the Dataset
Yelp Open Dataset (https://www.yelp.com/dataset (Links to an external site.)) – Repository of Yelp data covering reviews, businesses, users, checking, etc.

Untar and place the following files in the /yelp_dataset directory.

yelp_academic_dataset_business.json

yelp_academic_dataset_checkin.json

yelp_academic_dataset_review.json

yelp_academic_dataset_tip.json

yelp_academic_dataset_user.json

## Running the Data Mining Studies
Open main.py and set the appropriate flags.  Be sure to set the number of data points (nR, nT) at the bottom so that a memory error does not occur.  functions.py contain functions supporting all py files.

To run the business data analysis, set the business flag.  Also, set the classify (this will run functions in task1.py) and the cluster flags (this will run functions in task1_cluster.py).

To run the tips data analysis, set the tip flag (this will run functions in task2.py).  Set the DR list to run the desired dimensionality reduction method or not.  dot files will generate in /plots (create this directory first).  If you have Graphviz or Pydot installed, you can uncomment the code in the plotDecisionTree functions in functions.py to produce pngs.  Otherwise, you can use a free online converter e.g. https://onlineconvertfree.com/ 

## Contacts
Jacob Thomas [jacob.thomas@sjsu.edu]
David Luong [david.luong@sjsu.edu]
Taylor Maurer [taylor.maurer@sjsu.edu]
