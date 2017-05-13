#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

import numpy as np

sc=SparkContext("local[2]","spark recommendation app")

raw_rating_data=sc.textFile("/usr/bigdata/data/ml-100k/u.data")

rating_fields=raw_rating_data.map(lambda line:line.split("\t"))

rating_data=rating_fields.map(lambda fields:(int(fields[0]),int(fields[1]),float(fields[2])))

#print rating_data.first()

ratings=rating_data.map(lambda rating:Rating(rating[0],rating[1],rating[2]))

#print ratings.first()


model=ALS.train(ratings, 50, 10, 0.01)

#print model.userFeatures().count()

#print model.productFeatures().count()

#计算给定用户对指定物品的预期得分
predictedRating=model.predict(789,123)
print "user 789 will rate 123 as %f" %  predictedRating

#为指定用户生成前K个推荐物品
userId=789
K=10

#针对指定用户 找出ta评级最高的k个电影
user_rating=rating_fields.filter(lambda fields:fields[0]==str(userId))
print "user of %d has rated %d" % (userId, user_rating.count())


topKRecommendations=model.recommendProducts(userId, K)

rec=np.array(topKRecommendations)




movie_data=sc.textFile("/usr/bigdata/data/ml-100k/u.item")
movie_data.cache()
movies=movie_data.map(lambda line:line.split("|"))
def findTitleOfMovie(id):
    print "id is:%d" % id
    movie=movies.filter(lambda fields:fields[0]==str(id))
    if movie.count()==0:
        return "NA"
    else:
        return np.array(movie.first())[1]

for r in rec:
    #print "user %d rates movie %d (%s) as %f" % (userId, r[1],findTitleOfMovie(r[1]), r[2])
    print "user %d is predicted to rate movie %d as %f" % (userId, r[1], r[2])



#print findTitleOfMovie(1)
