

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
import torch_clustering
from munkres import Munkres
def evaluate(
    model: torch.nn.Module,
    data_loader_test: DataLoader,
    device: torch.device,
    args=None,
):
    N = args.n_sample_test
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        features_all = torch.zeros(args.n_views, N, args.embed_dim).to(device)
        labels_all = torch.zeros(N, dtype=torch.long).to(device)
        labels_all_re = torch.zeros(N, dtype=torch.long).to(device)

        for indexs, samples, labels, labels2, id1, id2, label_align in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            labels2 = labels2.to(device, non_blocking=True)
            features, res_idx, _= model.extract_feature(samples)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]

            labels_all[indexs] = labels
            labels_all_re[indexs] = labels2[res_idx]

        features_cat = features_all.permute(1, 0, 2).reshape(N, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1)
        kmeans_label = run_k_means_pytorch(features_cat,args).cpu().numpy()
        labels_all_cpu = np.asarray(labels_all.cpu())

    nmi, ari, f, acc = calculate_metrics(labels_all_cpu, kmeans_label)
    car = labels_all - labels_all_re
    car = (car == 0).sum().item() / len(car)

    result = {'k-means'  :   {'nmi': nmi, 'ari': ari, 'f': f, 'acc': acc, 'car': car}}
    return result

def run_k_means_pytorch(feature, args, random_state=0, return_centroids=False, verbose=False):
    kwargs = {
        #'cosine' else 'euclidean',
        'metric': 'cosine',
        'distributed': False,
        'random_state': random_state,
        'n_clusters': args.n_classes,
        'verbose': verbose
    }

    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    pseudo_labels = clustering_model.fit_predict(feature)
    if return_centroids:
        centroids = clustering_model.cluster_centers_
        return pseudo_labels, centroids
    else:
        return pseudo_labels

def calculate_metrics(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)

    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters, return_dict=False):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_cluster:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    # compute accuracy based on optimal 1:1 use_assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    if return_dict:
        return y_pred, kmeans_to_true_cluster_labels
    else:
        return y_pred
