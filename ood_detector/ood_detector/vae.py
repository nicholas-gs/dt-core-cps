#!/usr/bin/env python3

import torch

BETA = 1

CNN_ARCH = [
    {
        'in_channels': 3,
        'out_channels': 4,
        'k_conv': 3,
        'k_pool': 2,
        'activation': torch.nn.LeakyReLU(),
    },
    {
        'in_channels': 4,
        'out_channels': 8,
        'k_conv': 3,
        'k_pool': 2,
        'activation': torch.nn.LeakyReLU(),
    },
    {
        'in_channels': 8,
        'out_channels': 16,
        'k_conv': 3,
        'k_pool': 2,
        'activation': torch.nn.LeakyReLU(),
    },
    {
        'in_channels': 16,
        'out_channels': 32,
        'k_conv': 3,
        'k_pool': 2,
        'activation': torch.nn.LeakyReLU(),
    },
    {
        'in_channels': 32,
        'out_channels': 64,
        'k_conv': 3,
        'k_pool': 2,
        'activation': torch.nn.LeakyReLU(),
    },
]


class EncConvBlock(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, k_conv, k_pool, activation
    ):
        super().__init__()
        #print(f'enc_in: {in_channels}')
        #print(f'enc_out: {out_channels}')
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, k_conv, padding=k_conv // 2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.pool = torch.nn.MaxPool2d(
            k_pool, return_indices=True, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.pool(x)


class DecConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_conv, k_pool, activation):
        super().__init__()
        #print(f'dec_in: {in_channels}')
        #print(f'dec_out: {out_channels}')
        self.unpool = torch.nn.MaxUnpool2d(k_pool)
        self.conv_t = torch.nn.ConvTranspose2d(
            in_channels, out_channels, k_conv, padding=k_conv // 2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x, indices, out_size):
        x = self.unpool(x, indices, output_size=out_size)
        x = self.conv_t(x)
        x = self.batch_norm(x)
        return self.activation(x)


class FcLayer(torch.nn.Module):

    def __init__(self, in_neurons, out_neurons, activation):
        super().__init__()
        #print(f'in: {in_neurons}')
        #print(f'out: {out_neurons}')
        self.linear = torch.nn.Linear(in_neurons, out_neurons)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)


class CnnEncoder(torch.nn.Module):

    def __init__(self, conv_blocks, fc_layers):
        super().__init__()
        self.conv_blocks = torch.nn.ModuleList(
            [EncConvBlock(**c) for c in conv_blocks])
        self.fc_layers = torch.nn.ModuleList([FcLayer(**f) for f in fc_layers])

    def forward(self, x):
        indices_l = []
        out_sizes = []
        for block in self.conv_blocks:
            out_sizes.append(x.shape)
            x, indices = block(x)
            #print(x.shape)
            indices_l.append(indices)
        out_sizes.append(x.shape)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x[:, :x.shape[-1] // 2], x[:, x.shape[-1] // 2:], \
            indices_l, out_sizes


class CnnDecoder(torch.nn.Module):

    def __init__(self, fc_layers, conv_blocks):
        super().__init__()
        self.fc_layers = torch.nn.ModuleList(
            [FcLayer(**f) for f in fc_layers])
        self.conv_blocks = torch.nn.ModuleList(
            [DecConvBlock(**c) for c in conv_blocks])

    def forward(self, x, indices_l, out_sizes):
        for layer in self.fc_layers:
            #print(x.shape)
            x = layer(x)
        x = torch.reshape(x, out_sizes.pop())
        for block in self.conv_blocks:
            x = block(x, indices_l.pop(), out_sizes.pop())
        return x


class Vae(torch.nn.Module):
    def __init__(self, conv_blocks, fc_layers, beta=1, learning_rate=1e-5):
        super().__init__()
        self.beta = beta
        self.learning_rate = learning_rate
        self.n_latent = fc_layers[-1]['out_neurons'] // 2
        dec_conv_blocks = []
        dec_fc_layers = [
            {
                'in_neurons': fc_layers[-1]['out_neurons'] // 2,
                'out_neurons': fc_layers[-1]['in_neurons'],
                'activation': fc_layers[-1]['activation'],
            }
        ]
        for layer in fc_layers[-2::-1]:
            dec_fc_layers.append(
                {
                    'in_neurons': layer['out_neurons'],
                    'out_neurons': layer['in_neurons'],
                    'activation': layer['activation'],
                }
            )
        for block in conv_blocks[::-1]:
            dec_conv_blocks.append(
                {
                    'in_channels': block['out_channels'],
                    'out_channels': block['in_channels'],
                    'k_conv': block['k_conv'],
                    'k_pool': block['k_pool'],
                    'activation': block['activation']
                }
            )
        dec_conv_blocks[-1]['activation'] = torch.nn.Sigmoid()
        self.encoder = CnnEncoder(conv_blocks, fc_layers)
        self.decoder = CnnDecoder(dec_fc_layers, dec_conv_blocks)


    def forward(self, x):
        mu, logvar, indices, out_sizes = self.encoder(x)
        stdev = torch.exp(logvar / 2)
        eps = torch.randn_like(stdev)
        z = mu + stdev * eps
        return self.decoder(z, indices, out_sizes), mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        loss = mse_loss + self.beta * kl_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        loss = mse_loss + self.beta * kl_loss
        self.log('val_loss', loss)
        #self.log('y', y)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        _, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        self.log('dkl', kl_loss)
        return kl_loss, y


    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        x_hat, mu, logvar = self.forward(x)
        return x_hat, mu, logvar, y
