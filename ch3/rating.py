from pyspark import SparkContext

import numpy as np

sc=SparkContext("local[2]","spark rating app")

rating_data=sc.textFile("/usr/bigdata/data/ml-100k/u.data")

print rating_data.first()
print rating_data.count()

rating_fields=rating_data.map(lambda line:line.split("\t"))

num_users=rating_fields.map(lambda fields:fields[0]).distinct().count()

num_movies=rating_fields.map(lambda fields:fields[1]).distinct().count()

print "%d users have rated %d movies" % (num_users, num_movies)

ratings=rating_fields.map(lambda fields:int(fields[2]))

num_ratings=ratings.count()

max_rating=ratings.reduce(lambda x,y:max(x,y))

min_rating=ratings.reduce(lambda x,y:min(x,y))

average_rating=ratings.reduce(lambda x,y:x+y)/num_ratings

print "max rating: %f; min rating: %f; average rating: %f" % (max_rating, min_rating, average_rating)

print ratings.stats()

count_by_rating=ratings.countByValue()

print count_by_rating

user_rating_grouped=rating_fields.map(lambda fields:(fields[0],int(fields[2]))).groupByKey()

user_rating_byuser=user_rating_grouped.map(lambda (k,v):(k,len(v)))

for kv in np.array(user_rating_byuser.take(5)):
    print "user %s has rated %d times" % (kv[0],int(kv[1]))

