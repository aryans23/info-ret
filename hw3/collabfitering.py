# load the data, then print out the number of ratings, 
# movies and users in each of train and test sets.
# load the data, then print out the number of ratings, 
# movies and users in each of train and test sets.

import pandas as pd
import numpy as np

train_df = pd.read_csv('netflix-dataset/TrainingRatings.txt', header = None, usecols=[0,1,2])
# print(train_df[:10])
train_df.columns = ['mid','uid','r']
print("Number of Movies in train set = " + str(train_df.loc[:,'mid'].unique().size))
print("Number of Users in train set = " + str(train_df.loc[:,'uid'].unique().size))
print("Number of Ratings in train set = " + str(train_df.loc[:,'r'].unique().size))

test_df = pd.read_csv('netflix-dataset/TestingRatings.txt', header = None)
# print(test_df[:10])
test_df.columns = ['mid','uid','r']
print("Number of Movies in test set = " + str(test_df.loc[:,'mid'].unique().size))
print("Number of Users in test set = " + str(test_df.loc[:,'uid'].unique().size))
print("Number of Ratings in test set = " + str(test_df.loc[:,'r'].unique().size))

class Movie:
    def __init__(self):
        self.avg_rating = None
        self.ratings = dict({})

class User:
    def __init__(self):
        self.avg_rating = None
        self.ratings = dict({})
        

user_data = {}
movie_data = {}

train_file = 'netflix-dataset/TrainingRatings.txt'

with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        a = line.rstrip('\n').split(',')
        movie_id = a[0]
        user_id = a[1]
        rating = float(a[2])

        if user_data.get(user_id): 
            user_data[user_id].ratings[movie_id] = rating
        else: 
            user_data[user_id] = User()
            user_data[user_id].ratings[movie_id] = rating


        if movie_data.get(movie_id): 
            movie_data[movie_id].ratings[user_id] = rating

        else: 
            movie_data[movie_id] = Movie()
            movie_data[movie_id].ratings[user_id] = rating

for user_id, val in user_data.items():
    user_data[user_id].avg_rating = np.mean(np.array(val.ratings.values()))
    

for movie_id, val in movie_data.items():
    movie_data[movie_id].avg_rating = np.mean(np.array(val.ratings.values()))
    


test_file = 'netflix-dataset/TestingRatings.txt'
predicted = []
true = []
k = 0.001
i = 0

with open(test_file, 'r') as f:
    lines = f.readlines()
    n = 0 
    mae_sum = 0.0 
    rmse_sum = 0.0 
    weights = {} 

    for line in lines[:5000]:
        print(i)
        i += 1         
        info = line.rstrip('\n').split(',')
        movie_id = info[0]
        user_id = info[1]
        rating = float(info[2])
        if user_id in user_data:
            curr_user = user_data[user_id] 
            avg_rating = curr_user.avg_rating 
            sum = 0.0 
            for new_user_id, train_movie in user_data.items(): 
                if (new_user_id != user_id and movie_id in train_movie.ratings):
                    weight = 0 
                    if (user_id, new_user_id) not in weights and (new_user_id, user_id) not in weights:
                        num = 0
                        d1 = 0
                        d2 = 0
                        for m, rating in train_movie.ratings.items():
                            if m != movie_id and m in curr_user.ratings:
                                num += (curr_user.ratings[m]-curr_user.avg_rating)*(rating-train_movie.avg_rating) 
                        for m, rating in train_movie.ratings.items():
                            if m != movie_id and m in curr_user.ratings:
                                d1 += pow((curr_user.ratings[m]-curr_user.avg_rating),2)
                                d2 += pow((rating-train_movie.avg_rating),2)
                        if (d1 == 0 or d2 == 0):
                            continue
                        weight = num/pow((d1*d2),0.5)
                        weights[(user_id, new_user_id)] = weight
                    else:
                        if (user_id, new_user_id) in weights:
                            weight = weights[(user_id, new_user_id)]
                        elif (new_user_id, user_id) in weights:
                            weight = weights[(new_user_id, user_id)]
                    sum += weight * (train_movie.ratings[movie_id] - train_movie.avg_rating)
            predicted_rating = avg_rating + k * sum
#             print("sum = " + str(sum))
#             print("avg_rating = " + str(avg_rating))
#             print("predicted_rating = " + str(predicted_rating))

            if predicted_rating < 1.0:
                predicted_rating = 1.0
            elif predicted_rating > 5.0:
                predicted_rating = 5.0
            else:
                predicted_rating = round(predicted_rating)

            n += 1

            predicted.append(predicted_rating)
            true.append(rating)

    print(predicted)
    print(true)

# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(true, predicted))

# Root Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(true, predicted)

