import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate=0.25):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = drop_rate
        self.gru1 = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.act1 = nn.LeakyReLU(0.2)
        self.gru2 = nn.GRU(self.hidden_dim * 2, self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.mlp = nn.Linear(self.hidden_dim * 2, 2)
        self.act2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        gru1_out, gru1_h = self.gru1(x)
        drop_out = self.dropout(gru1_out)
        act1_out = self.act1(drop_out)
        gru2_out, gru2_h = self.gru2(act1_out)
        mlp_out = self.mlp(gru2_out)
        out = self.act2(mlp_out)

        return out


class Discriminator(nn.Module):
    def __init__(self, model_size, input_dim, hidden_dim, drop_rate=0.25):
        super(Discriminator, self).__init__()
        self.model_size = model_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = drop_rate
        self.gru1 = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.act1 = nn.LeakyReLU(0.2)
        self.out_layer = nn.Linear(self.model_size * self.hidden_dim * 2, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        gru1_out, gru1_h = self.gru1(x)
        drop_out = self.dropout(gru1_out)
        act1_out = self.act1(drop_out)
        mlp_in = act1_out.reshape(act1_out.shape[0], -1)
        mlp_out = self.out_layer(mlp_in)
        out = self.act2(mlp_out)

        return out


if __name__ == '__main__':
    a = torch.randn(8, 64, 2)
    generator = Generator(2, 16, 0.2)
    b = generator(a)
    print(b.shape)
    discriminator = Discriminator(64, 2, 16, 0.2)
    c = discriminator(b)
    print(c.shape)
