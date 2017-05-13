from pyspark import SparkContext
import numpy as np

sc=SparkContext("local[2]","spark movie app")

movie_data=sc.textFile("/usr/bigdata/data/ml-100k/u.item")

print "first row of movie data:"
print movie_data.first()

num_movie=movie_data.count()
print "num of movies: %d" % num_movie

def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900


movie_fields=movie_data.map(lambda line:line.split("|"))

movie=movie_fields.filter(lambda fields:fields=="1")
print "the info of movie 1 is:"
print movie.collect()

years=movie_fields.map(lambda fields:fields[2]).map(lambda x:convert_year(x))

years_filtered=years.filter(lambda x:x!=1900)

print "total years: %d; and count of movie not in 1900: %d" % (years.count(), years_filtered.count())

movie_ages=years_filtered.map(lambda year:2017-year).countByValue()

print movie_ages.keys()

print movie_ages.values()

#movie_age_np=np.array(movie_ages)
#for a in movie_age_np:
#    print "age: %d happens %d" % (a[0],a[1])

