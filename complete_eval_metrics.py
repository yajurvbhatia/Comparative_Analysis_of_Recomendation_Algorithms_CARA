from scipy import spatial
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def avg_precision_recall(recommendations, ratings, threshold = 0):
    avg_precision = 0
    avg_recall = 0
    for i in range(1, len(recommendations)+1):
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        true_negatives = 0

        precision = 0
        recall = 0
        actual_ratings = ratings[i-1]
        # print("actual_ratings of the user are\n", actual_ratings, "\n")
        # print("recommendations of the user are\n", recommendations[i-1], "\n")
        for item_no in range(1, len(actual_ratings)+1):
            if (actual_ratings[item_no-1] > threshold):
                # print("Actual preference of item no ", item_no)
                if item_no in recommendations[i-1]:
                    # print("\t item recommended")
                    true_positives +=1
                else:
                    # print("\t item not recommended")
                    false_negatives +=1
            else:
                # print("Actual dislikeness of item no ", item_no)
                if item_no in recommendations[i-1]:
                    # print("\t item recommended")
                    false_positives +=1
                else:
                    # print("\t item not recommended")
                    true_negatives +=1
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        avg_precision += precision
        avg_recall += recall

    return avg_precision/len(recommendations), avg_recall/len(recommendations)


def avg_coverage(recommendations, item_info):
    # c is the total categories in the dataset
    avg_coverage = 0
    for user_recommendation in recommendations:
        coverage = 0
        xord_item_infos = np.zeros((len(user_recommendation), item_info.shape[1]))
        i=0
        for recommendation in user_recommendation:
            xord_item_infos[i] = item_info[recommendation-1]
            i+=1
        xord_item_infos = xord_item_infos.sum(axis=0)
        xord_item_infos[xord_item_infos!=0] = 1
        no_of_categories_recommended = xord_item_infos.sum()

        coverage = (no_of_categories_recommended/item_info.shape[1])*100
        avg_coverage += coverage
    return avg_coverage/len(recommendations)


def personalization(recommendations):
    personalization = 0
    for i_1 in range(1,len(recommendations)+1):
        inv_jaccard_coeff = 0
        for i_2 in range(1,len(recommendations)+1):
            if not(i_1 == i_2):
                intersection = len(list(set(recommendations[i_1-1]).intersection(recommendations[i_2-1])))
                union = (len(recommendations[i_1-1]) + len(recommendations[i_2-1])) - intersection
                inv_jaccard_coeff += 1-float(intersection/union)
        inv_jaccard_coeff = inv_jaccard_coeff/len(recommendations)
        personalization += inv_jaccard_coeff
    return personalization/len(recommendations)


def avg_intra_list_sim(recommendations, item_info):
    avg_il_sim = 0
    for user_recommendation in recommendations:
        il_sim = 0
        sim = 1
        for recommended_item in user_recommendation:
            i1_vector = item_info[recommended_item-1]
            for other_rec_item in user_recommendation:
                if not (other_rec_item == recommended_item):
                    i2_vector = item_info[other_rec_item-1]
                    sim += 1-spatial.distance.cosine(i1_vector, i2_vector)
            sim = sim/len(user_recommendation)
            il_sim += sim
        il_sim = il_sim/len(user_recommendation)
        avg_il_sim += il_sim
    return avg_il_sim/len(recommendations)
