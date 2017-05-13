from pyspark import SparkContext


sc=SparkContext("local[2]","user spark app")

user_data=sc.textFile("/usr/bigdata/data/ml-100k/u.user")

#print "total user count: %d" % user_data.count()

user_fields=user_data.map(lambda line: line.split("|"))

num_users=user_fields.map(lambda fields: fields[0]).count()

num_genders=user_fields.map(lambda fields:fields[2]).distinct().count()

num_occupations=user_fields.map(lambda fields:fields[3]).distinct().count()

num_zipcodes=user_fields.map(lambda fields:fields[4]).distinct().count()

print "Users: %d, genders: %d, occupations: %d, zipcodes: %d" % (num_users, num_genders, num_occupations, num_zipcodes)

count_by_occupation1=user_fields.map(lambda fields: (fields[3],1) ).reduceByKey(lambda x,y: x+y)

print "map reduce approach of occupations:"
print dict(count_by_occupation1.collect())

count_by_occupation2=user_fields.map(lambda fields:fields[3]).countByValue()

print "count by value approach of occupations:"
print dict(count_by_occupation2)


