
import argparse
import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf

import datasets

# keeping DS global so that train and test datasets
# are not loaded each time for each dataloader
DS = None


def init_augmentor(augmentation_level):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=augmentation_level*20,
        width_shift_range=augmentation_level*0.1,
        height_shift_range=augmentation_level*0.1,
        # shear_range=10,
        # zoom_range=0.1,
        channel_shift_range=augmentation_level*0.2,
        fill_mode="reflect",
    )


class ImageFragmentsDataset(IterableDataset):
    """
    An Iterable Dataset where each item has the shuffled image fragments from 10 images and
    the corresponding image source ids
    """
    def __init__(self, image_data_generator, image_size=64, fragment_size=16, num_images_per_sample=10, frag_augmentation_level=0.0):
        self.image_data_generator = image_data_generator
        self.image_size = image_size
        self.fragment_size = fragment_size
        self.num_fragments_per_axis = self.image_size // self.fragment_size
        self.num_images_per_sample = num_images_per_sample

        assert 0.0 <= frag_augmentation_level <= 1.0

        self.frag_augmentation_level = frag_augmentation_level
        if self.frag_augmentation_level > 0.0:
            self.augmentor = init_augmentor(self.frag_augmentation_level)

    def __iter__(self):
        for x, _ in self.image_data_generator:
            x = torch.from_numpy(x)
            num_fragments_per_image = self.num_fragments_per_axis * self.num_fragments_per_axis

            # assign the image source id for each image fragment
            img_source_tensor = torch.zeros(self.num_images_per_sample * num_fragments_per_image)
            for img_ind in range(self.num_images_per_sample):
                img_source_tensor[img_ind * num_fragments_per_image: (img_ind + 1) * num_fragments_per_image] = img_ind

            # fragment each image
            fragmented_x = self.fragment(x)
            fragmented_x = fragmented_x.view(self.num_images_per_sample * num_fragments_per_image, -1)

            if self.frag_augmentation_level > 0.0:
                fragmented_x = fragmented_x.reshape(shape=(fragmented_x.shape[0], self.fragment_size, self.fragment_size, 3)).numpy()
                fragmented_x = next(self.augmentor.flow(fragmented_x, batch_size=self.num_images_per_sample * num_fragments_per_image)) #augmented fragments
                fragmented_x = torch.from_numpy(fragmented_x)
                fragmented_x = fragmented_x.reshape(shape=(fragmented_x.shape[0], self.fragment_size*self.fragment_size*3))

            # now shuffle both the image fragments and image source ids
            shuffled_fragment_indices = torch.randperm(self.num_images_per_sample * num_fragments_per_image)
            shuffled_fragmented_x = fragmented_x.index_select(0, shuffled_fragment_indices)
            shuffled_img_source_tensor = img_source_tensor.index_select(0, shuffled_fragment_indices)
            yield shuffled_fragmented_x, shuffled_img_source_tensor

    def fragment(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, fragment_size**2 *3)
        """
        p = self.fragment_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x


def get_data_loader(data_path, data_type, batch_size, image_size=64,fragment_size=16, num_images_per_sample=10):
    """ returns a train/test pytorch dataloader based on the input data_type """
    assert data_type in ['train', 'test']
    global DS
    if DS is None:
        DS = datasets.Imagenet64(data_path)
    if data_type == 'train':
        image_dg = DS.datagen_cls(num_images_per_sample, ds='train', augmentation=True)
        data_size = DS.train_size // num_images_per_sample
    elif data_type == 'test':
        image_dg = DS.datagen_cls(num_images_per_sample, ds='test', augmentation=False)
        data_size = DS.test_size // num_images_per_sample

    iterable_dataset = ImageFragmentsDataset(image_dg,
                                             image_size=image_size,
                                             fragment_size=fragment_size,
                                             num_images_per_sample=num_images_per_sample)
    dataloader = DataLoader(iterable_dataset, batch_size=batch_size, num_workers=0)

    return dataloader, data_size // batch_size

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataloader test.")

    parser.add_argument('--path-to-data-folder', type=str, help='Path to data folder.')

    # Parse the command-line arguments
    args = parser.parse_args()

    dl, _ = get_data_loader(args.path_to_data_folder, 'train', 4)

    for x_batch, y_batch in dl:
        print(x_batch.shape, y_batch.shape)
