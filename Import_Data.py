import numpy as np
# import sys
# np.set_printoptions(threshold=sys.maxsize)

def import_100k():
    file = open("MovieLens_Datasets/ml-100k/u.data", "r")
    data = file.read()
    file.close()

    # Number of Users in 100K dataset is 943, initializing list with 0s
    # Number of Movies in 100K dataset is 1682, initializing list with 0s
    ratings = np.zeros((943,1682))

    data = data.split('\n')[:-1]
    for row in data:
        new_row = row.split('\t')
        # print(int(new_row[2]))
        ratings[int(new_row[0])-1][int(new_row[1])-1] = float(new_row[2])
    return ratings

def import_movie_profiles_100k():
    file = open("MovieLens_Datasets/ml-100k/u.item", "r", encoding="latin1")
    data = file.read()
    file.close()

    # item no. i
    i=0
    profiles = np.zeros((1682,19))
    data = data.split('\n')[:-1]
    for row in data:
        new_row = row[-38:]
        new_row = new_row.split('|')[1:]
        profiles[i] = np.array(new_row)
        i+=1
    return profiles

def import_1m():
    file = open("MovieLens_Datasets/ml-1m/ratings.dat", "r")
    data = file.read()
    file.close()
    # Number of Users in 100K dataset is 943, initializing list with 0s
    # Number of Movies in 100K dataset is 1682, initializing list with 0s
    data = data.split('\n')[:-1]
    ratings = np.zeros((6040,3952))
    for row in data:
        new_row = row.split('::')
        # print(int(new_row[2]))
        ratings[int(new_row[0])-1][int(new_row[1])-1] = float(new_row[2])
    return ratings

def import_movie_profiles_1m():
    genres = ["Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "(no genres listed)"]

    file = open("MovieLens_Datasets/ml-1m/movies.dat", "r", encoding="latin1")
    data = file.read()
    file.close()

    # item no. i
    i=0
    nb_genres = len(genres)
    profiles = np.zeros((3952,nb_genres))
    data = data.split('\n')[:-1]
    for row in data:
        new_row = row.split('::')[2]
        movie_genres = new_row.split('|')
        movie_vector = list()
        for genre in movie_genres:
            movie_vector.append(genres.index(genre))
        targets = np.array([movie_vector]).reshape(-1)
        labels = np.eye(nb_genres)[targets]
        movie_vector = labels.sum(axis=0)
        profiles[i] = np.array(movie_vector)
        i+=1
    return profiles

def import_10m():
    file = open("MovieLens_Datasets/ml-10M100K/ratings.dat", "r")
    data = file.read()
    file.close()
    # Number of Users in 100K dataset is 943, initializing list with 0s
    # Number of Movies in 100K dataset is 1682, initializing list with 0s
    data = data.split('\n')[:-1]
    ratings = np.zeros((71567,65133))
    for row in data:
        new_row = row.split('::')
        ratings[int(new_row[0])-1][int(new_row[1])-1] = float(new_row[2])
    return ratings

def import_movie_profiles_10m():
    genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "IMAX", "(no genres listed)"]

    file = open("MovieLens_Datasets/ml-10M100K/movies.dat", "r", encoding="latin1")
    data = file.read()
    file.close()

    # item no. i
    i=0
    nb_genres = len(genres)
    profiles = np.zeros((65133,nb_genres))
    data = data.split('\n')[:-1]
    for row in data:
        new_row = row.split('::')[2]
        movie_genres = new_row.split('|')
        movie_vector = list()
        for genre in movie_genres:
            movie_vector.append(genres.index(genre))
        targets = np.array([movie_vector]).reshape(-1)
        labels = np.eye(nb_genres)[targets]
        movie_vector = labels.sum(axis=0)
        profiles[i] = np.array(movie_vector)
        i+=1
    return profiles

def import_20m():
    file = open("MovieLens_Datasets/ml-20m/ratings.csv", "r")
    data = file.read()
    file.close()
    # Number of Users in 100K dataset is 943, initializing list with 0s
    # Number of Movies in 100K dataset is 1682, initializing list with 0s
    data = data.split('\n')[1:-1]
    ratings = np.zeros((138493,131262))
    for row in data:
        new_row = row.split(',')
        ratings[int(new_row[0])-1][int(new_row[1])-1] = float(new_row[2])
    return ratings

def import_movie_profiles_20m():
    genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "IMAX", "(no genres listed)"]

    file = open("MovieLens_Datasets/ml-20m/movies.csv", "r", encoding="latin1")
    data = file.read()
    file.close()

    # item no. i
    i=0
    nb_genres = len(genres)
    profiles = np.zeros((131262,nb_genres))
    data = data.split('\n')[1:-1]
    for row in data:
        new_row = row.split(',')[-1]
        movie_genres = new_row.split('|')
        movie_vector = list()
        for genre in movie_genres:
            movie_vector.append(genres.index(genre))
        targets = np.array([movie_vector]).reshape(-1)
        labels = np.eye(nb_genres)[targets]
        movie_vector = labels.sum(axis=0)
        profiles[i] = np.array(movie_vector)
        i+=1
    return profiles

def import_25m():
    file = open("MovieLens_Datasets/ml-25m/ratings.csv", "r")
    data = file.read()
    file.close()
    # Number of Users in 100K dataset is 943, initializing list with 0s
    # Number of Movies in 100K dataset is 1682, initializing list with 0s
    data = data.split('\n')[1:-1]
    ratings = np.zeros((162541,209171))
    for row in data:
        new_row = row.split(',')
        # print(int(new_row[2]))
        ratings[int(new_row[0])-1][int(new_row[1])-1] = float(new_row[2])
    return ratings

def import_movie_profiles_25m():
    genres = ["Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "IMAX", "(no genres listed)"]

    file = open("MovieLens_Datasets/ml-25m/movies.csv", "r", encoding="latin1")
    data = file.read()
    file.close()

    # item no. i
    i=0
    nb_genres = len(genres)
    profiles = np.zeros((131262,nb_genres))
    data = data.split('\n')[1:-1]
    for row in data:
        new_row = row.split(',')[-1]
        movie_genres = new_row.split('|')
        movie_vector = list()
        for genre in movie_genres:
            movie_vector.append(genres.index(genre))
        targets = np.array([movie_vector]).reshape(-1)
        labels = np.eye(nb_genres)[targets]
        movie_vector = labels.sum(axis=0)
        profiles[i] = np.array(movie_vector)
        i+=1
    return profiles
