import json
import os
from typing import List, Dict, Union

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm


def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    clip_model: torch.nn.Module,
    index_features: torch.Tensor,
    index_names: List,
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    # Move the features to the device
    device = 'cpu'
    print(f'image shape (#image * n_view * n_patch * dim): {index_features.shape}')
    print(f'text shape (n_caption * #text * dim): {predicted_features.shape}')
    index_features = torch.nn.functional.normalize(index_features, dim=-1)
    predicted_features = torch.nn.functional.normalize(predicted_features, dim=-1)
    predicted_features = predicted_features.permute(1,0,2)
    predicted_features_mean = (predicted_features[:,:-1].mean(dim=1) + predicted_features[:,-1]) / 2
    predicted_features_mean = torch.nn.functional.normalize(predicted_features_mean, dim=-1)
    # We have 3 distance metrics to compare
    # 1. Cosine distance between the predicted features and the index features
    distances = 1 - predicted_features_mean.to(device) @ index_features[:,0, 0].to(device).T * clip_model.logit_scale.exp().cpu()
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }

    print('Base:', output_metrics)

    lamda = 0.5
    predicted_features[:,0] = predicted_features[:,0] * lamda + predicted_features[:,-1] * (1-lamda)
    predicted_features[:,1] = predicted_features[:,1] * lamda + predicted_features[:,-1] * (1-lamda)
    predicted_features[:,2] = predicted_features[:,2] * lamda + predicted_features[:,-1] * (1-lamda)
    predicted_features[:,3] = predicted_features[:,3] * lamda + predicted_features[:,-1] * (1-lamda)
    predicted_features[:,4] = predicted_features[:,4] * lamda + predicted_features[:,-1] * (1-lamda)
    predicted_features = torch.nn.functional.normalize(predicted_features, dim=-1)

    scores_total = None
    batch_size = 100
    for i in range(0, predicted_features.size(0), batch_size):
        batch_predicted_features = predicted_features[i:i + batch_size]
        sim = torch.einsum('cmd,bnd->cbmn', batch_predicted_features[:,:5].to(device), index_features[:,:,0].to(device)) * clip_model.logit_scale.exp().cpu()
        sim = sim.contiguous().view(-1, sim.size(2), sim.size(3))
        wdist = 1 - sim
        kk = torch.exp(-wdist / 1.0)
        aa = torch.ones(kk.size(0), kk.size(1)).to(device) / kk.size(1)
        bb = torch.ones(kk.size(0), kk.size(2)).to(device) / kk.size(2)
        scores = -torch.sum(Sinkhorn(kk, aa, bb) * sim, dim=[1, 2]).contiguous().view(batch_predicted_features.size(0), index_features.size(0))
        if scores_total is None:
            scores_total = scores
        else:
            scores_total = torch.cat((scores_total, scores))

    sorted_indices = torch.argsort(scores_total, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    output_metrics1 = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    print('AWT: ', output_metrics1)

    scores_total = None
    batch_size = 100
    for i in range(0, predicted_features.size(0), batch_size):
        batch_predicted_features = predicted_features[i:i + batch_size]
        sim = torch.einsum('cmd,bnd->cbmn', batch_predicted_features[:,:5].to(device), index_features[:,0].to(device)) * clip_model.logit_scale.exp().cpu()
        sim = sim.contiguous().view(-1, sim.size(2), sim.size(3))
        wdist = 1 - sim
        kk = torch.exp(-wdist / 1.0)
        aa = torch.ones(kk.size(0), kk.size(1)).to(device) / kk.size(1)
        bb = torch.ones(kk.size(0), kk.size(2)).to(device) / kk.size(2)
        scores = -torch.sum(Sinkhorn(kk, aa, bb) * sim, dim=[1, 2]).contiguous().view(batch_predicted_features.size(0), index_features.size(0))
        if scores_total is None:
            scores_total = scores
        else:
            scores_total = torch.cat((scores_total, scores))
    
    sorted_indices = torch.argsort(scores_total, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))   
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    output_metrics2 = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    print('PLOT: ', output_metrics2)

    scores_total = None
    batch_size = 100
    for i in range(0, predicted_features.size(0), batch_size):
        batch_predicted_features = predicted_features[i:i + batch_size]
        sim = torch.einsum('cmd,bnd->cbmn', batch_predicted_features[:,:5].to(device), index_features[:,:,0].to(device)) * clip_model.logit_scale.exp().cpu()
        sim = sim.contiguous().view(-1, sim.size(2), sim.size(3))
        forward_pi = torch.softmax(sim / 0.1, dim=1)
        backward_pi = torch.softmax(sim / 0.1, dim=2)
        forward_cost = (forward_pi * sim).sum(dim=1).mean(1)
        backward_cost = (backward_pi * sim).sum(dim=2).mean(1) 
        scores = -(forward_cost + backward_cost).contiguous().view(batch_predicted_features.size(0), index_features.size(0))
        if scores_total is None:
            scores_total = scores
        else:
            scores_total = torch.cat((scores_total, scores))

    sorted_indices = torch.argsort(scores_total, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))   
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    output_metrics3 = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    print('CT: ', output_metrics3)


    return output_metrics


@torch.no_grad()
def fiq_v0(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    # Move the features to the device
    index_features = torch.nn.functional.normalize(index_features).to(device)
    predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    return output_metrics
    

@torch.no_grad()
def cirr(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    clip_model: torch.nn.Module,
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    split: str='val',    
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Computes Recall@1, 5, 10 and 50. If given a test set, will generate submittable file.
    """   
    # Put on device.

    device = 'cpu'
    print(f'image shape (#image * n_view * n_patch * dim): {index_features.shape}')
    print(f'text shape (n_caption * #text * dim): {predicted_features.shape}')
    index_features = torch.nn.functional.normalize(index_features, dim=-1)
    predicted_features = torch.nn.functional.normalize(predicted_features, dim=-1)
    predicted_features = predicted_features.permute(1,0,2)
    predicted_features_mean = (predicted_features[:,:-1].mean(dim=1) + predicted_features[:,-1]) / 2
    predicted_features_mean = torch.nn.functional.normalize(predicted_features_mean, dim=-1)
    # We have 3 distance metrics to compare
    # 1. Cosine distance between the predicted features and the index features
    distances = 1 - predicted_features_mean.to(device) @ index_features[:,0, 0].to(device).T * clip_model.logit_scale.exp().cpu()

    # We have 3 distance metrics to compare
    # 1. Cosine distance between the predicted features and the index features
    distances = 1 - predicted_features_mean @ index_features[:,0, 0].T


    print(distances.shape, predicted_features_mean.shape, index_features.shape)
    # distances = 1 - predicted_features @ index_features.T
    if distances.ndim == 3:
        # If there are multiple features per instance, we average.
        distances = distances.mean(dim=1)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    resize = len(sorted_index_names) if split == 'test' else len(target_names)
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(resize, -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    # Compute the subset predictions and ground-truth labels
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    if split == 'test':
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
        for pair_id, prediction in zip(query_ids, sorted_index_names):
            pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
        for pair_id, prediction in zip(query_ids, sorted_group_names):
            pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)

        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'cirr')
        os.makedirs(submissions_folder_path, exist_ok=True)

        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(os.path.join(submissions_folder_path, f"subset_{preload_dict['test']}"), 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        return None
            
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))    
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {f'recall@{key}': (torch.sum(labels[:, :key]) / len(labels)).item() * 100 for key in [1, 5, 10, 50]}
    output_metrics.update({f'group_recall@{key}': (torch.sum(group_labels[:, :key]) / len(group_labels)).item() * 100 for key in [1, 2, 3]})

    return output_metrics


@torch.no_grad()
def circo(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List,
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names.
    Computes mAP@5, 10, 25 and 50. If test-split, generates submittable file.
    """
    # Load the model
    # Put on device.
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)
    
    ### Compute Test Submission in case of test split.
    if split == 'test':
        print('Generating test submission file!')
        similarity = predicted_features @ index_features.T
        if similarity.ndim == 3:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=1)                    
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        # Return prediction dict to submit.
        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        
        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'circo')
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)        
        return None
    
    ### Directly compute metrics when using validation split.
    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
        
    for predicted_feature, target_name, sub_targets in tqdm.tqdm(zip(predicted_features, target_names, targets), total=len(predicted_features), desc='Computing Metric.'):
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        if similarity.ndim == 2:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=0)
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        assert target_name == sub_targets[0], f"Target name not in GTs {target_name} {sub_targets}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        
        for key in retrievals:
            recalls[key].append(float(torch.sum(single_gt_labels[:key])))

    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}
    output_metrics.update({f'recall@{key}': np.mean(item) * 100 for key, item in recalls.items()})
    return output_metrics


@torch.no_grad()
def genecis(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    index_features: torch.Tensor, 
    index_ranks: List,
    topk: List[int] = [1, 2, 3],
    **kwargs    
) -> Dict[str, float]:
    
    predicted_features = torch.nn.functional.normalize(predicted_features.float(), dim=-1).to(device)
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1).to(device)
    
    # Compute similarity
    if predicted_features.ndim == 3:
        similarities = predicted_features.bmm(index_features.permute(0,2,1)).mean(dim=1)
    else:
        similarities = (predicted_features[:, None, :] * index_features).sum(dim=-1)

    # # Sort the similarities in ascending order (closest example is the predicted sample)
    _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

    # Compute recall at K
    if isinstance(index_ranks, list):
        index_ranks = torch.stack(index_ranks)
    index_ranks = index_ranks.to(device)
    
    output_metrics = {f'R@{k}': get_recall(sort_idxs[:, :k], index_ranks) * 100 for k in topk}

    return output_metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
    













