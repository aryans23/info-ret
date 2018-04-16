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

training_file = 'netflix-dataset/TrainingRatings.txt'

with open(training_file, 'r') as f:
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
    


testing_set = 'netflix-dataset/TestingRatings.txt'
predicted = []
true = []
k = 0.001
i = 0

with open(testing_set, 'r') as f:
    lines = f.readlines()
    n = 0 
    mae_sum = 0.0 
    rmse_sum = 0.0 
    weights = {} 

    for line in lines[:10]:
        print(i)
        i += 1 
        
        a = line.rstrip('\n').split(',')
        movie_id = a[0]
        user_id = a[1]
        rating = float(a[2])
        
        if user_id in user_data:
            current_user = user_data[user_id] 
            avg_rating = current_user.avg_rating 
            sum = 0.0 

            for key, value in user_data.items(): 
                
                if (key != user_id and movie_id in value.ratings):
                    weight = 0 
                    
                    if (user_id, key) not in weights and (key, user_id) not in weights:
                        
                        num = 0
                        d1 = 0
                        d2 = 0
                        for m, rating in value.ratings.items():
                            
                            if m != movie_id and m in current_user.ratings:
                                num += (current_user.ratings[m]-current_user.avg_rating)*(rating-value.avg_rating) 
                        for m, rating in value.ratings.items():
                            
                            if m != movie_id and m in current_user.ratings:
                                d1 += pow((current_user.ratings[m]-current_user.avg_rating),2)
                                d2 += pow((rating-value.avg_rating),2)
                        if (d1 == 0 or d2 == 0):
                            continue
                        weight = num/pow((d1*d2),0.5)
                            
                        
                        weights[(user_id, key)] = weight
                    else:
                        
                        if (user_id, key) in weights:
                            weight = weights[(user_id, key)]
                        elif (key, user_id) in weights:
                            weight = weights[(key, user_id)]
                    
                    sum += weight * (value.ratings[movie_id] - value.avg_rating)
                    
            calc_rating = avg_rating + k * sum # Here's our predicted rating.
#             print("sum = " + str(sum))
#             print("avg_rating = " + str(avg_rating))
#             print("calc_rating = " + str(calc_rating))

            if calc_rating < 1.0:
                calc_rating = 1.0
            elif calc_rating > 5.0:
                calc_rating = 5.0
            else:
                calc_rating = round(calc_rating)

            n += 1

            predicted.append(calc_rating)
            true.append(rating)

    print(predicted)
    print(true)

# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(true, predicted))

# Root Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(true, predicted)

