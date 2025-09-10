
import torch


def get_sample_contrastive_loss(sample, sources, margin=1.0):
    """
    Computes the contrastive loss for a given sample batch using a pairwise distance matrix.

    Args:
        sample (Tensor): A tensor of shape (N, D), where N is the number of fragments, D is the feature dimension.
        sources (Tensor): A tensor of shape (N,) containing the image source ID for each fragment.
        margin (float): The margin used for dissimilar pairs in contrastive loss.

    Returns:
        Tensor: A scalar tensor representing the contrastive loss for the sample batch.
    """

    n_fragments_per_sample = sample.shape[0]

    # Compute pairwise Euclidean distances between all samples
    pairwise_distances = torch.cdist(sample, sample, p=2.0)

    # Create similarity label matrix (1.0 for same source, 0.0 otherwise)
    similarity_labels = torch.zeros(n_fragments_per_sample, n_fragments_per_sample)
    # bool matrix to keep track of source equality
    equ_mat = (sources.unsqueeze(1) == sources.unsqueeze(0))
    # set fragment pairs from same source to 1 in similarity matrix
    similarity_labels[equ_mat] = 1.0
    similarity_labels = similarity_labels.to(sample.device)

    # contrastive loss computation
    positive_pairs_loss = torch.mul(similarity_labels, torch.square(pairwise_distances))
    negative_pairs_loss = torch.mul(1.0 - similarity_labels,
                                   torch.square(torch.maximum(margin - pairwise_distances,
                                                              torch.tensor(0.0))))

    loss_matrix = 0.5 * (positive_pairs_loss + negative_pairs_loss)

    # we compute mean over all fragment pairs in sample batch
    sample_loss = torch.mean(loss_matrix)

    return sample_loss