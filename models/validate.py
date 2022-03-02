import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from gan import Generator


parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=int, default=2, help="size of model ")
parser.add_argument("--input_dim", type=int, default=2, help="dim of model input")
parser.add_argument("--hidden_dim", type=int, default=2, help="dim of model hidden")
arg = parser.parse_args()

output_file = 'generated_actions'

try:
    os.mkdir(output_file)
except OSError:
    print('Directory %s already exists' % output_file)
else:
    print('Successfully created the directory %s' % output_file)

tic = time.perf_counter()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

random_seed = 999

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

model_path = 'trained_model/dx_dy/generator.bin'
generator = Generator(input_dim=arg.input_dim, hidden_dim=arg.hidden_dim)
generator.load_state_dict(torch.load(model_path))

if cuda:
    generator.cuda()

# test datasets path
test_feature_data_dx_dy = 'features/test_raw_features_dx_dy.csv'
df_test_feature_dx_dy = pd.read_csv(test_feature_data_dx_dy, header=None)
test_feature_array = df_test_feature_dx_dy.values
test_feature_x = test_feature_array[:, 4:-1]
test_feature_x = test_feature_x.reshape(-1, arg.model_size, arg.input_dim)  # [sample_num, 64, 2]
features_test = torch.tensor(test_feature_x)  # [sample_num, 64, 2]

test_label_data_dx_dy = 'equidistant_actions/test_equidistant.csv'
df_test_label_dx_dy = pd.read_csv(test_label_data_dx_dy, header=None)
test_label_array = df_test_label_dx_dy.values
test_label_x = test_label_array.reshape(-1, arg.model_size, arg.input_dim)  # [sample_num, 64, 2]
label_test = torch.tensor(test_label_x)  # [sample_num, 64, 2]

test_dataset = Data.TensorDataset(label_test, features_test)
data_iter_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

generator.eval()
with torch.no_grad():
    # generate actions
    df = pd.DataFrame()
    for i, (label, feature) in enumerate(data_iter_test):
        z = Tensor(np.random.normal(0, 1, (1, arg.model_size, arg.input_dim)))
        label_input = label.type(Tensor)
        real_points = feature.type(Tensor)

        label_z = label_input + z
        df_generated = generator(label_z)  # [batch_size, 64, 2]
        df_generated = df_generated.data.cpu().numpy()

        dim1, dim2, dim3 = df_generated.shape
        df_generated = df_generated.reshape(dim1, dim2 * dim3)
        df_generated = np.round(df_generated, decimals=1)
        df_generated = pd.DataFrame(data=df_generated)
        df_generated[df_generated == 0.] = 0.
        df = pd.concat([df, df_generated], axis=0)
    df.to_csv(os.path.join(output_file, 'test_actions_dx_dy.csv'), index=False, header=False)

toc = time.perf_counter()
print(f"Execution time: {toc - tic:0.4f} seconds")
