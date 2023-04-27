from scipy import spatial
import numpy as np
import random
import math
import time

def get_random_rec(u, k, m):
    st = time.time()
    recommendations = list()
    for i in range(u):
        item_ids = [i for i in range(m)]
        user_recommendations = list()
        for j in range(k):
            user_recommendations.append(random.randint(1,m))
        recommendations.append(user_recommendations)
    en = time.time()
    return recommendations, en-st

def get_popular_rec(u, ratings, k):
    st = time.time()
    recommendations = list()
    for i in range(u):
        estimated_ratings = ratings.sum(axis=0)
        user_recommendations = list()
        for r in range(k):
            most_popular = np.argmax(estimated_ratings) +1
            user_recommendations.append(most_popular)
            estimated_ratings[most_popular-1]=-999
        recommendations.append(user_recommendations)
    en = time.time()
    return recommendations, en-st

def get_content_filtered_rec(u, ratings, m_profiles, k, nb_genres):
    st = time.time()
    recommendations = list()
    for user_id in range(1,u+1):
        u1_ratings = ratings[user_id-1]
        rated_items = list()
        u_profile = np.zeros((1,nb_genres))
        for item_no in range(len(u1_ratings)):
            if u1_ratings[item_no] > 0:
                rated_movie_profile = m_profiles[item_no]
                u_profile += rated_movie_profile*u1_ratings[item_no]

        estimated_ratings = np.zeros(len(m_profiles))
        for item_no in range(len(m_profiles)):
            estimated_ratings[item_no] =  5*spatial.distance.cosine(m_profiles[item_no], u_profile)

        u1_ratings[u1_ratings != 0 ] = 1000
        u1_ratings[u1_ratings == 0 ] = 1
        u1_ratings[u1_ratings == 1000 ] = 0
        estimated_ratings = np.multiply(estimated_ratings, u1_ratings)
        user_recommendations = list()
        for r in range(k):
            top_scorer = np.argmax(estimated_ratings)+1
            user_recommendations.append(top_scorer)
            estimated_ratings[top_scorer-1] = -999
        recommendations.append(user_recommendations)
    en = time.time()
    return recommendations, en-st

def get_user_based_collab_filtered_rec(ratings, u, k, m):
    st = time.time()
    # u is the number of most similar user to take into consideration
    # k is the number of recommendations required
    # m is the number of movies in the dataset

    # Calculating the similarity between user choices, using cosine similarity
    recommendations = list()
    for user_id in range(1, u+1):
        u1_ratings = ratings[user_id-1]
        cosine_distances = [0 for x in range(ratings.shape[1])]
        for u2_id in range(ratings.shape[0]):
            if not (u2_id == user_id):
                u2_ratings = ratings[u2_id-1]
                cosine_distances[u2_id-1] = spatial.distance.cosine(u1_ratings, u2_ratings)

        estimated_ratings = np.zeros((u,m))
        similarity_score_summation = 0
        for i in range(u):
            user_id = np.argmax(cosine_distances) + 1
            similarity_score = cosine_distances[user_id-1]
            cosine_distances.remove(similarity_score)
            similarity_score_summation += similarity_score
            estimated_ratings[i] = np.multiply(ratings[user_id+1], similarity_score)

        estimated_ratings = estimated_ratings.sum(axis=0)
        estimated_ratings = estimated_ratings/similarity_score_summation

        u1_ratings[u1_ratings != 0 ] = 1000
        u1_ratings[u1_ratings == 0 ] = 1
        u1_ratings[u1_ratings == 1000 ] = 0
        estimated_ratings = np.multiply(estimated_ratings, u1_ratings)

        user_recommendations = list()
        for r in range(k):
            top_scorer = np.argmax(estimated_ratings)+1
            user_recommendations.append(top_scorer)
            estimated_ratings[top_scorer-1] = -999
        recommendations.append(user_recommendations)
    en = time.time()
    return recommendations, en-st

def get_item_based_collab_filtered_rec(u, ratings, n, k, m):
    st = time.time()
    # n is the number of most similar user to take into consideration
    # k is the number of recommendations required
    # m is the number of movies in the dataset

    # Calculating the similarity between user choices, using cosine similarity
    recommendations = list()
    Trans_ratings = ratings.T
    for user_id in range(1,u+1):
        u1_ratings = ratings[user_id-1]
        rated_items = list()
        for item_no in range(m):
            if u1_ratings[item_no] > 0:
                rated_items.append(item_no+1)
        estimated_ratings = np.zeros(m)
        similarity_score_summation = np.zeros(m)
        for i in rated_items:
            i1_vector = Trans_ratings[i-1]
            cosine_distances = [1 for x in range(m)]
            for i2_id in range(1,m+1):
                if not (i2_id == i):
                    i2_vector = Trans_ratings[i2_id-1]
                    cosine_distances[i2_id-1] = spatial.distance.cosine(i1_vector, i2_vector)

            for j in range(n):
                item_id = np.argmin(cosine_distances) + 1
                similarity_score = 1 - cosine_distances[item_id-1]
                cosine_distances.remove(1-similarity_score)
                estimated_ratings[item_id-1] += similarity_score*u1_ratings[i-1]
                similarity_score_summation[item_id-1] += similarity_score

        similarity_score_summation[ similarity_score_summation==0 ] = 1
        estimated_ratings = estimated_ratings/similarity_score_summation

        user_recommendations = list()
        for r in range(k):
            top_scorer = np.argmax(estimated_ratings)+1
            user_recommendations.append(top_scorer)
            estimated_ratings[top_scorer-1] = -999
        recommendations.append(user_recommendations)
    en = time.time()
    return recommendations, en-st

def get_category_based_rec(u, ratings, m_profiles, k, nb_genres):
    st = time.time()
    for user_id in range(1,u+1):
        u1_ratings = ratings[user_id-1]
        rated_items = list()
        u_profile = np.zeros(m_profiles.shape[1])
        for item_no in range(len(u1_ratings)):
            if u1_ratings[item_no] > 0:
                rated_movie_profile = m_profiles[item_no]
                u_profile += rated_movie_profile*u1_ratings[item_no]
                rated_items.append(item_no+1)

        ranked_u_pro_cat_ids = list()
        ranked_u_pro_cat_scores = np.zeros(nb_genres)
        for i in range(len(u_profile)):
            item_no = np.argmax(u_profile) + 1
            ranked_u_pro_cat_ids.append(item_no)
            ranked_u_pro_cat_scores[i] = u_profile[item_no-1]
            u_profile[item_no-1] = -999

        m_profiles = m_profiles.T
        ranked_u_pro_cat_scores = np.divide(ranked_u_pro_cat_scores, ranked_u_pro_cat_scores.sum())
        pass_to_next = 0
        recommended_items = list()
        for cat_no in range(1, len(ranked_u_pro_cat_ids)+1):
            num_selected = int(math.ceil(ranked_u_pro_cat_scores[cat_no-1]*k + pass_to_next))
            # getting items in this category
            m_profile = m_profiles[cat_no-1]
            category_items = [x+1 for x in range(len(m_profile)) if(m_profile[x]==1 and x+1 not in rated_items)]
            if(len(category_items)>=num_selected):
                pass_to_next = 0
                for j in range(num_selected):
                    rec_item = random.choice(category_items)
                    category_items.remove(rec_item)
                    recommended_items.append(rec_item)
            else:
                recommended_items += category_items
                pass_to_next = num_selected - len(category_items)
        recommendations.append(recommended_items[:k])
    en = time.time()
    return recommendations, en-st
