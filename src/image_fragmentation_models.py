""" Pytorch model definitions that acts on individual image fragments """
import torch.nn as nn


class LinearImageFragmentModel(nn.Module):
    def __init__(self, fragment_size, out_features):
        super(LinearImageFragmentModel, self).__init__()
        self.fragment_size = fragment_size
        self.out_features = out_features
        in_features = self.fragment_size * self.fragment_size * 3
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class ConvulationalImageFragmentModel(nn.Module):
    def __init__(self, fragment_size, out_features):
        super(ConvulationalImageFragmentModel, self).__init__()
        self.fragment_size = fragment_size
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_features, kernel_size=5, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=5, padding=0)

    def forward(self, x):
        batch_size = x.shape[0]
        num_img_fragments = x.shape[1]
        x = x.reshape(shape=(x.shape[0] * x.shape[1], x.shape[2]))
        x = x.reshape(shape=(x.shape[0], self.fragment_size, self.fragment_size, 3))
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.mean(dim=(2, 3)) # global mean pooling
        x = x.reshape(shape=(batch_size, num_img_fragments, self.out_features))
        return x