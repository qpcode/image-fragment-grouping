import os
import argparse
import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

import datasets
from data_loader import ImageFragmentsDataset
from image_fragmentation_models import LinearImageFragmentModel, ConvulationalImageFragmentModel
import cluster_metrics
from loss import get_sample_contrastive_loss


TRAINED_MODEL_DIR = 'TRAINED_MODEL_DIR'

def evaluate_image_fragment_model(image_fragment_model, test_dataloader,
                                  num_test_batches, device, num_images_per_sample=10):
    """ Evaluates the provided image_fragment_model on the given test dataset """

    # set model to evaluation model
    image_fragment_model.eval()

    # initialize metric collectors
    test_losses = []
    test_recalls = []
    test_precisions = []
    print("Starting model evaluation...")
    for batch_idx, (x_test, source_test) in enumerate(test_dataloader):

        #  termination check
        if batch_idx >= num_test_batches:
            break

        # move data to device
        x_test = x_test.to(device)
        # forward pass to get fragment features
        features = image_fragment_model(x_test)

        batch_size = features.shape[0]
        batch_losses = []
        batch_recalls = []
        batch_precisions = []

        for si in range(batch_size):

            sample_loss = get_sample_contrastive_loss(features[si, :], source_test[si, :])
            batch_losses.append(sample_loss.detach().cpu().numpy())

            kmeans = KMeans(n_clusters=num_images_per_sample, random_state=42, n_init='auto')
            cluster_assignments = kmeans.fit_predict(features[si, :].detach().cpu().numpy())
            sample_cluster_metrics = cluster_metrics.compute_pairwise_agreement_metrics(
                source_test[si, :].detach().cpu().numpy(),
                cluster_assignments)
            batch_recalls.append(sample_cluster_metrics['recall'])
            batch_precisions.append(sample_cluster_metrics['precision'])

        test_losses.extend(batch_losses)
        test_recalls.extend(batch_recalls)
        test_precisions.extend(batch_precisions)

    # Calculate final metrics
    mean_test_loss = np.mean(test_losses) if test_losses else 0.0
    mean_test_recall = np.mean(test_recalls) if test_recalls else 0.0
    mean_test_precision = np.mean(test_precisions) if test_precisions else 0.0

    return {'mean_test_loss': mean_test_loss,
            'mean_test_recall': mean_test_recall,
            'mean_test_precision': mean_test_precision}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval script.")

    parser.add_argument('--eval-file-path', type=str, help='Path to data folder.')
    parser.add_argument('--fragment-size', type=int, default=16, help='fragment size.')
    parser.add_argument('--batch-size', type=int, default=128, help='fragment size.')
    parser.add_argument('--feature-dimension', type=int, default=64, help='fragment size.')
    parser.add_argument('--model-type', choices=['linear', 'conv'], required=True,
                        help='Choose between "linear" or "conv"')
    parser.add_argument('--epoch-num', type=int, default=5, help='epoch number of trained model to be used for eval.')

    # Parse the command-line arguments
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'linear':
        image_fragment_model = LinearImageFragmentModel(args.fragment_size, args.feature_dimension)
    elif args.model_type == 'conv':
        image_fragment_model = ConvulationalImageFragmentModel(args.fragment_size, args.feature_dimension)
    model_file_path = os.path.join(TRAINED_MODEL_DIR,
                                   f'image_fragment_model_{args.model_type}_'
                                   f'fd_{args.feature_dimension}_'
                                   f'epoch_{args.epoch_num}.pth')
    if not os.path.exists(model_file_path):
        print(f"Model file {model_file_path} does not exist. Check that valid model params are provided.")
    if torch.cuda.is_available():
        state_dict = torch.load(model_file_path)
    else:
        state_dict = torch.load(model_file_path, map_location=torch.device('cpu'))
    image_fragment_model.load_state_dict(state_dict)
    image_fragment_model = image_fragment_model.to(device)

    NUM_IMAGES_PER_SAMPLE = 10
    DS = datasets.Imagenet64Eval(args.eval_file_path)

    for frag_augmentation_level in np.arange(0, 1.04, 0.04):
        image_dg = DS.datagen_cls(NUM_IMAGES_PER_SAMPLE)

        iterable_dataset = ImageFragmentsDataset(image_dg, fragment_size=args.fragment_size, frag_augmentation_level=float(frag_augmentation_level))
        test_dataloader = DataLoader(iterable_dataset, batch_size=args.batch_size, num_workers=0)
        num_test_batches = DS.test_size // args.batch_size

        test_metrics = evaluate_image_fragment_model(image_fragment_model,
                                                 test_dataloader,
                                                 num_test_batches,
                                                 device,
                                                 num_images_per_sample=NUM_IMAGES_PER_SAMPLE)
    
        print("Test metrics:")
        print(test_metrics)

        import csv
        with open(f'{args.model_type}_frag_aug.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([frag_augmentation_level, test_metrics['mean_test_recall'], test_metrics['mean_test_precision']])
    
                        
