# -*- coding: utf8 -*-
import math
from numbers import Number


def calc_distance_euclidean(vector1, vector2):
    # !!! all passed vector elements to this method must be float values !!!

    # validate combarability
    if len(vector1) != len(vector2):
        raise AttributeError("Compared vectors have different number of arguments!")

    # init differences vector
    per_element_distances = [0] * len(vector1)
    
    # compute (each vector element) difference for RMSE (for euclidean distance)
    for index, (value1, value2) in enumerate(zip(vector1, vector2)):
        per_element_distances[index] = value1 - value2

    # compute RMSE (root mean squared error)
    return math.sqrt(sum([val**2 for val in per_element_distances]) / len(per_element_distances))


def calc_k_distance(k, vector, dataset):
    # TODO: consider caching for more efficient re-computation
    distances = {}
    for vector2 in dataset:
        distance = calc_distance_euclidean(vector1=vector, vector2=vector2)
        if distance in distances:
            distances[distance].append(vector2)
        else:
            distances[distance] = [vector2]
    distances = sorted(distances.items())
    neighbours = []
    [neighbours.extend(n[1]) for n in distances[:k]]
    k_distance = distances[k - 1][0] if len(distances) >= k else distances[-1][0]
    return k_distance, neighbours


def calc_k_reachability_distance(k, vector1, vector2, dataset):
    (k_distance, neighbours) = calc_k_distance(k, vector2, dataset)
    return max(k_distance, calc_distance_euclidean(vector1=vector1, vector2=vector2))


def calc_local_reachability_density(k, vector, dataset):
    (k_distance, neighbours) = calc_k_distance(k=k, vector=vector, dataset=dataset)
    reachability_distances = [0] * len(neighbours)
    for index, neighbour in enumerate(neighbours):
        reachability_distances[index] = calc_k_reachability_distance(k=k, vector1=vector, vector2=neighbour, dataset=dataset)
    if sum(reachability_distances) == 0:
        # TODO: vector is identical with its neighbors, consider fixing this case!
        # returning 'inf' to note that this vector has an issue
        return float("inf")
    else:
        return len(neighbours) / sum(reachability_distances)


def calc_local_outlier_factor(k, vector, dataset):
    (k_distance, neighbours) = calc_k_distance(k=k, vector=vector, dataset=dataset)
    vector_lrd = calc_local_reachability_density(k=k, vector=vector, dataset=dataset)
    lrd_ratios = [0] * len(neighbours)
    
    for index, neighbour in enumerate(neighbours):
        tmp_dataset_without_neighbor = set(dataset)
        tmp_dataset_without_neighbor.discard(neighbour)
        neighbour_lrd = calc_local_reachability_density(k=k, vector=neighbour, dataset=tmp_dataset_without_neighbor)
        lrd_ratios[index] = neighbour_lrd / vector_lrd
    
    return sum(lrd_ratios) / len(neighbours)


def calc_lof_boundary(vector, dataset):
    return 0


def detect_outliers(k, dataset):
    tmp_dataset = dataset
    outliers = []

    for index, vector in enumerate(tmp_dataset):
        tmp_dataset_without_vector = list(tmp_dataset)
        tmp_dataset_without_vector.remove(vector)
        LOF = calc_local_outlier_factor(k, vector, tmp_dataset_without_vector)
        
        # TODO: add threshold in LOF boundary calculation
        # According to the source: https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
        # Theorem 1: LOF(p) <= lof_boundary(p): direct_max_distance(p) / indirect_min_distance(p)
        # But it needs to be implemented after distance calculation caching is implemented
        if LOF > calc_lof_boundary(vector, dataset):
            outliers.append({"LOF": LOF, "vector": vector, "index": index})
    outliers.sort(key=lambda lof_key: lof_key["LOF"], reverse=True)
    return outliers


if __name__ == "__main__":
    # dataset = (
    #     (31, 340, 56, 15),
    #     (26, 376, 80, 3.7),
    #     (26, 380, 72, 5.1),
    #     (25, 368, 75, 3.9),
    #     (25, 370, 68, 5.5)
    # )
    dataset = (
        (0, 0, 0),
        (-0.07858112, 0.04371652, 0.08754063),
        (-0.071033, 0.04098192, 0.06599712),
        (-0.068174735, 0.026259795, 0.06360817),
        (-0.062882856, 0.030693293, 0.07797909)
    )
    for outlier in detect_outliers(3, dataset):
        print(outlier)
