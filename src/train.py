import os
import argparse
import torch
import torch.optim as optim
from collections import deque
import numpy as np
import wandb

import data_loader
from image_fragmentation_models import LinearImageFragmentModel, ConvulationalImageFragmentModel
from loss import get_sample_contrastive_loss
import evaluate


TRAINED_MODEL_DIR = 'TRAINED_MODEL_DIR'
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)
WANDB_LOG_DIR = 'WANDB_LOGS'


def print_and_log_test_metrics(epoch_ind, test_metrics):
    print(f"Epoch {epoch_ind}, mean-test-loss: {test_metrics['mean_test_loss']}")
    print(f"Epoch {epoch_ind}, mean-test-recall: {test_metrics['mean_test_recall']}")
    print(f"Epoch {epoch_ind}, mean-test-precision: {test_metrics['mean_test_precision']}")

    wandb.log({'epoch_ind': epoch_ind,
               'test-loss': test_metrics['mean_test_loss'],
               'test-recall': test_metrics['mean_test_recall'],
               'test-precision': test_metrics['mean_test_precision']})
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script.")

    parser.add_argument('--path-to-data-folder', type=str, required=True, help='Path to data folder.')
    parser.add_argument('--fragment-size', type=int, default=16, help='fragment size.')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size for training.')
    parser.add_argument('--max-epochs', type=int, default=5, help='number of epochs for training.')
    parser.add_argument('--feature-dimension', type=int, default=64, help='out feature dimension.')
    parser.add_argument('--model-type', choices=['linear', 'conv'], required=True,
                        help='Choose between "linear" or "conv"')

    # Parse the command-line arguments
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'linear':
        image_fragment_model = LinearImageFragmentModel(args.fragment_size, args.feature_dimension)
    elif args.model_type == 'conv':
        image_fragment_model = ConvulationalImageFragmentModel(args.fragment_size, args.feature_dimension)
    image_fragment_model = image_fragment_model.to(device)

    optimizer = optim.SGD(image_fragment_model.parameters(), lr=0.01)

    # train dataset loader
    train_dl,num_train_batches = data_loader.get_data_loader(args.path_to_data_folder, 'train', args.batch_size, fragment_size=args.fragment_size)

    # test dataset loader
    test_dl, num_test_batches = data_loader.get_data_loader(args.path_to_data_folder, 'test', 1, fragment_size=args.fragment_size)

    all_losses = deque(maxlen=num_train_batches)

    batch_ind = 0
    epoch_ind = 0

    wandb.init(project='image-fragment-mapping',
               dir=WANDB_LOG_DIR,
               name=f'image_fragment_model_{args.model_type}_fd_{args.feature_dimension}',
               config={'model_type': args.model_type,
                       'feature_dimension': args.feature_dimension,
                       'fragment_size': args.fragment_size,
                       'batch_size': args.batch_size,
                       'max_epochs': args.max_epochs})

    # doing initial eval before training starts
    test_metrics = \
        evaluate.evaluate_image_fragment_model(image_fragment_model,
                                               test_dl,
                                               num_test_batches,
                                               device)
    print_and_log_test_metrics(epoch_ind, test_metrics)

    for x_batch, source_batch in train_dl:

        image_fragment_model.train()

        x_batch = x_batch.to(device)

        features = image_fragment_model(x_batch)

        batch_size = features.shape[0]
        sample_losses = []
        for si in range(batch_size):
            sample_loss = get_sample_contrastive_loss(features[si,:], source_batch[si, :])
            sample_losses.append(sample_loss)
        combined_loss = torch.stack(sample_losses)
        batch_mean_loss = torch.mean(combined_loss)
        all_losses.append(batch_mean_loss.detach().cpu().numpy())

        mean_train_loss = np.mean(all_losses)
        print(f"Epoch: {epoch_ind}, batch-ind = {batch_ind}/{num_train_batches}")
        print(f"mean-train-loss: {mean_train_loss}")

        wandb.log({'epoch_ind': epoch_ind,
                   'train-loss': mean_train_loss})

        optimizer.zero_grad()  # Clear gradients
        batch_mean_loss.backward()  # Compute gradients
        optimizer.step()

        batch_ind += 1
        if batch_ind >= num_train_batches:

            # update epoch and batch-numbers
            epoch_ind += 1
            batch_ind = 0

            test_metrics = \
                evaluate.evaluate_image_fragment_model(image_fragment_model,
                                                       test_dl,
                                                       num_test_batches,
                                                       device)
            print_and_log_test_metrics(epoch_ind, test_metrics)

            print("Saving new pytorch model")
            model_file_path = os.path.join(TRAINED_MODEL_DIR,
                                           f'image_fragment_model_{args.model_type}_'
                                           f'fd_{args.feature_dimension}_'
                                           f'epoch_{epoch_ind}.pth')
            torch.save(image_fragment_model.state_dict(), model_file_path)

        if epoch_ind >= args.max_epochs:
            break

    wandb.finish()
