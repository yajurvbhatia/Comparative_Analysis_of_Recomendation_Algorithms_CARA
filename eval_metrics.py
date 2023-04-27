from scipy import spatial
import numpy as np

def precision_recall(recommendations, actual_ratings, threshold = 2):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    precision = 0
    recall = 0
    for item_no in range(1, len(actual_ratings)+1):
        if (actual_ratings[item_no-1] > threshold):
            # print("Actual preference of item no ", item_no)
            if item_no in recommendations:
                # print("\t item recommended")
                true_positives +=1
            else:
                # print("\t item not recommended")
                false_negatives +=1
        else:
            # print("Actual dislikeness of item no ", item_no)
            if item_no in recommendations:
                # print("\t item recommended")
                false_positives +=1
            else:
                # print("\t item not recommended")
                true_negatives +=1
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)

    return precision, recall


def coverage(recommendations, item_info):
    # c is the total categories in the dataset
    cvrg = 0
    xord_item_infos = np.zeros((len(recommendations), item_info.shape[1]))
    i=0
    for recommendation in recommendations:
        xord_item_infos[i] = item_info[recommendation-1]
        i+=1
    xord_item_infos = xord_item_infos.sum(axis=0)
    xord_item_infos[xord_item_infos!=0] = 1
    no_of_categories_recommended = xord_item_infos.sum()

    cvrg = (no_of_categories_recommended/item_info.shape[1])*100

    return cvrg


def intra_list_sim(recommendations, item_info):
    il_sim = 0
    sim = 0
    for recommended_item in recommendations:
        i1_vector = item_info[recommended_item-1]
        for other_rec_item in recommendations:
            if not (other_rec_item == recommended_item):
                i2_vector = item_info[recommended_item-1]
                sim += 1-spatial.distance.cosine(i1_vector, i2_vector)
        sim = sim/len(recommendations)
        il_sim +=sim
    return il_sim/len(recommendations)
