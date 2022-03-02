import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from gan import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--model_size", type=int, default=2, help="num of time step")
parser.add_argument("--input_dim", type=int, default=2, help="dim of model input")
parser.add_argument("--hidden_dim", type=int, default=2, help="dim of model hidden")
parser.add_argument("--dropout_rate", type=float, default=0.25, help="rate of dropout")
parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate of G")
parser.add_argument("--lr_d", type=float, default=0.0002, help="adam: learning rate of D")
parser.add_argument("--b1_g", type=float, default=0.5, help="adam: decay of first order momentum of gradient of G")
parser.add_argument("--b1_d", type=float, default=0.5, help="adam: decay of first order momentum of gradient of D")
parser.add_argument("--b2_g", type=float, default=0.999, help="adam: decay of first order momentum of gradient of G")
parser.add_argument("--b2_d", type=float, default=0.999, help="adam: decay of first order momentum of gradient of D")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples")
arg = parser.parse_args()
print(arg)


cuda = True if torch.cuda.is_available() else False

random_seed = 999

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


def init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        nn.init.orthogonal_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.constant_(m.bias_hh_l0, 0)
        nn.init.constant_(m.bias_ih_l0, 0)


def plot(g_list, d_list, save_path):
    num_epoch = list(range(len(g_list)))
    plt.plot(num_epoch, g_list, color="orange")
    plt.plot(num_epoch, d_list)
    plt.title("generator and discriminator loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["generator", "discriminator"], loc="upper right")
    plt.savefig(save_path)


def save_points(gen_points, save_path):
    batch_size = gen_points.shape[0]
    reshape_points = gen_points.reshape(batch_size, -1)
    array_points = reshape_points.data.cpu().numpy()
    with open(save_path, 'w') as f:
        for points in array_points:
            save_points = []
            for point in points:
                save_points.append(str(round(point, 1)))
            f.write(','.join(save_points) + '\n')


# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(input_dim=arg.input_dim, hidden_dim=arg.hidden_dim, drop_rate=arg.dropout_rate)
discriminator = Discriminator(model_size=arg.model_size, input_dim=arg.input_dim, hidden_dim=arg.hidden_dim,
                              drop_rate=arg.dropout_rate)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(init_params)
discriminator.apply(init_params)

# Dataset loader
train_feature_data_dx_dy = 'features/train_raw_features_dx_dy.csv'
df_train_feature_dx_dy = pd.read_csv(train_feature_data_dx_dy, header=None)
train_feature_array = df_train_feature_dx_dy.values
train_feature_x = train_feature_array[:, 4:-1]
train_feature_x = train_feature_x.reshape(-1, arg.model_size, arg.input_dim)  # [sample_num, 64, 2]
features_train = torch.tensor(train_feature_x)  # [sample_num, 64, 2]

train_label_data_dx_dy = 'equidistant_actions/train_equidistant.csv'
df_train_label_dx_dy = pd.read_csv(train_label_data_dx_dy, header=None)
train_label_array = df_train_label_dx_dy.values
train_label_x = train_label_array.reshape(-1, arg.model_size, arg.input_dim)  # [sample_num, 64, 2]
label_train = torch.tensor(train_label_x)  # [sample_num, 64, 2]

train_dataset = Data.TensorDataset(label_train, features_train)
data_iter_train = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=arg.lr_g, betas=(arg.b1_g, arg.b2_g))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=arg.lr_d, betas=(arg.b1_d, arg.b2_d))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# save loss
g_loss_list = []
d_loss_list = []

# training
for epoch in range(arg.n_epochs):
    sum_g_loss = 0
    sum_d_loss = 0
    for i, (label, feature) in enumerate(data_iter_train):
        # Adversarial ground truths
        valid = Variable(Tensor(feature.shape[0], 1).fill_(1.0), requires_grad=False)  # [32, 1]
        fake = Variable(Tensor(feature.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_points = Variable(feature.type(Tensor))
        label_input = label.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (feature.shape[0], arg.model_size, arg.input_dim))))  # [32, 64, 2]

        label_z = label_input + z
        # Generate a batch of points
        gen_points = generator(label_z)  # [32, 64, 2]

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_points), valid)
        sum_g_loss += g_loss.item()

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_points), valid)
        fake_loss = adversarial_loss(discriminator(gen_points.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        sum_d_loss += d_loss.item()

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, arg.n_epochs, i, len(data_iter_train), d_loss.item(), g_loss.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(data_iter_train) + i
        if batches_done % arg.sample_interval == 0:
            save_points(gen_points, "results/dx_dy/points_%d.csv" % batches_done)
            # save_points(gen_points, "results/dx_dy_dt/points_%d.csv" % batches_done)

    sum_g_loss = sum_g_loss / len(data_iter_train)
    sum_d_loss = sum_d_loss / len(data_iter_train)
    g_loss_list.append(sum_g_loss)
    d_loss_list.append(sum_d_loss)

save_path = 'plot_loss/dx_dy/loss.png'

# plot loss
plot(g_loss_list, d_loss_list, save_path)

# save models
torch.save(generator.state_dict(), 'trained_model/dx_dy/generator.bin')
torch.save(discriminator.state_dict(), 'trained_model/dx_dy/discriminator.bin')
