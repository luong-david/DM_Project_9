This is meant to be sort of a plan. At least what I'm thinking we could do.
I tried to make this align with the proposal as best as I could, but I think
whatever time we have might change what we're actually capable of doing. 

# 1st - Clustering/Filtering
I think one of the first things that should happen is to cluster the businesses
in order to identify the restauraunts and bars and things. I guess this doesn't
necessarily have to be cluster analysis, but something to remove the non-food
like businesses. Like gyms and things.

[DL] Sounds appropriate to filter down the dataset to just the food-related businesses.

# 2nd - Clustering... again
Once we have the restaurants filtered from the other businesses I was thinking
of trying to understand the natrual groupings that would occur. I think a basic
clustering on the restaurant features would give us the results we want. The hard
part with this task and the #1 is figuring out how to vectorize the business'
categories and attributes. Anyway at the end of this step we'd have several clusters
of similar restaurants with similar attributes. We can average the rating and look 
at the spread of features within each of the clusters to get a first-look at stand-out 
attributes that may be contributors to the overall rating.

[DL] Yes, we should spend some time figuring out how to describe the attributes (binary vs multi class).  Maybe this could be formulated as a classification problem where the star ratings are the different classes?

#3rd - Figuring out what features contribute the most to rating
This can be done in a few different ways the main two I was thinking would be:
## A) Some type of regression/prediction
Using the features of the restaurants basically do either a dimenstioanlity reduction
or regression to identify what features are the bigger contributors. I'm not 100%
sure on how this would work-out.

[DL] Maybe a SVD or PCA dimensionality reduction will help identify the lower dimensions that capture the variance we require.

## B) Use the user reviews 
Using the user reviews do basic NLP looking at word frequency. Do this per each rating
for the review. In theory we could do this for each of the restauraunt clusters found
in step 2. Hopefully there'd be a strong correlation between the words with a higher 
frequency and the attributes of the business. So if a restaurant has 30 reviews, 15 of them
above 3 stars, and the word 'cheap' is said 20 times. We can try to compare that to the
attributes from the restaraunt. Again, I'm not 100% sure how well this would work out
especially because I think we could end up down the NLP rabbit hole fairly quickly!

[DL] Perhaps we can apply text clustering techniques to group restaurants on review similarity?  Finding the number of natural clusters may be a challenge, but we could be intuitivey guided by the kinds of reviews possible (gushing, meh, dissatisfaction, cheap, expensive, haute, etc.).  Or just look for the inflection point in the simiarlity vs K graph.

The idea behind this step is looking deeper at what makes the restaurants successful. Or even
unsuccessful, just the defining features.

# 4th- Clustering once again
Out of step 3 I'd think we'd have some specific features that we're looking for in 
a restaurant. We'd only use those features in a cluster analysis. Then I would hope the 
average rating of the clusters from this step would be more consistent. This would essentially
be verification that those features found from step 3 do indeed impact success of a restaraunt.

# 5th - KNN on location
Lastly, although this doesn't have to happen last we can take the location data provided by yelp
for each restaraunt and find the K nearest neighbors. Then with that infromation we can look at 
the variance of the features within the various neighborhoods. This will look at how common each
food item is. This will be good information to have to compare to the results that we find from
the other steps. 

[DL] Would be interesting to compare K nearest geopgraphic neighbors to the K nearest similar neighbors (on the selected attributes).

I'm hoping we see that certain restaurant types have a higher success and that
multiple of the same restaurant's are grouped together geographically. This would essentially 
support the fact that certain food types are 'easier' to get success in than others. Of course
we may find something completely different, but regardless I think this should be more than enough
to make a project on.

[DL] The data mining may be able to support our hypotheses and/or reveal something not so obvious.  Let's keep an open mind as we explore the data and see what results.