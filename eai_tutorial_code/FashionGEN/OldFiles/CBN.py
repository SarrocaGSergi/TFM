import torch
import torch.nn as nn


class CBN(nn.Module):

    def __init__(self, mlp_input, mlp_hidden, out_size, batch_size, channels, height, width, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.mlp_input = mlp_input # size of the lstm emb which is input to MLP
        self.mlp_hidden = mlp_hidden # size of hidden layer of MLP
        self.out_size = out_size # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.mlp_input, self.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden, self.out_size),
            ).cuda()

        self.fc_beta = nn.Sequential(
            nn.Linear(self.mlp_input, self.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden, self.out_size),
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def create_cbn_input(self, emb_size):

        if self.use_betas:
            delta_betas = self.fc_beta(emb_size)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(emb_size)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()

        return delta_betas, delta_gammas

    def forward(self, feature, emb_size):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape

        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(emb_size)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, emb_size