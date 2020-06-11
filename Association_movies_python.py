#install mlxtend
pip install mlxtend
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules#mlxtend for apriori
#Reading data
movies=pd.read_csv(r"D:\Python\New folder\my_movies.csv")
#Dropping columns
movies.drop(["V1", "V2", "V3", "V4","V5"], axis = 1,inplace = True) 
#Applying apriori 
f_movies = apriori(movies, min_support=0.005, max_len=3,use_colnames = True)
print(f_movies)# Most Frequent item sets based on support

#Rules on the basis of confidence
association_rules(f_movies, metric="confidence", min_threshold=0.5)
#Rules on the basis of lift
rules = association_rules(f_book, metric="lift", min_threshold=1)
print (rules)
#Sorting on the basis of support
f_movies.sort_values('support',ascending = False,inplace=True)
