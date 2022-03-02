import numpy as np


len_lst = []
data = np.load('human50.npz', allow_pickle=True)
for i in range(len(data['human50'])):
    for tra in data['human50'][i]:
        len_lst.append(len(tra))

len_sum = 0
len1 = 0
len2 = 0
len3 = 0
len4 = 0

for l in len_lst:
    if l < 50:
        len1 += 1
    elif l < 64:
        len2 += 1
    elif l < 90:
        len3 += 1
    else:
        len4 += 1
    len_sum += 1

print('len sum: ', len_sum)
print('less than 50: ', len1/len_sum)
print('less than 64: ', len2/len_sum)
print('less than 90: ', len3/len_sum)
print('more than 90: ', len4/len_sum)