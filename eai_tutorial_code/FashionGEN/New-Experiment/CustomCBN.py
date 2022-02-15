import torch
import torch.nn as nn


class CondBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, batch_size):
        super(CondBatchNorm2d, self).__init__()
        self.embedder = nn.Embedding(num_classes, 1)
        self.batch_norm = nn.BatchNorm2d(num_features, affine=True)
        self.eps = 1.0e-5
        # MLP for mean
        self.fc_gamma = nn.Sequential(
            nn.Linear(batch_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_features),
        )
        # MLP for Variance
        self.fc_beta = nn.Sequential(
            nn.Linear(batch_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_features),
        )

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def forward(self, x, y):
        label_embedding = self.embedder(y).view(-1)

        # Calculate mean and variance from the input feature map
        feature_mean = torch.mean(x)
        feature_variance = torch.var(x)

        # Calculate the mean and variance of the word embedding
        beta_mlp = self.fc_beta(label_embedding)
        gamma_mlp = self.fc_gamma(label_embedding)

        delta_beta = torch.mean(beta_mlp)
        delta_gamma = torch.var(gamma_mlp)

        # Final beta and gamma
        total_beta = torch.add(feature_mean, delta_beta)
        total_gamma = torch.add(feature_variance, delta_gamma)
        m = feature_variance + self.eps
        step_1 = (x-feature_mean)/torch.sqrt(torch.tensor(m))
        out = torch.mul(step_1, total_gamma) + total_beta

        return out



