"""
generate a new dataset with the values we need.
It will substitute values outside thresold by new randomized values.
It shouldn't touch the other values. After doing this, use the predictor to calculate new
success_rate and compare!!
"""


import numpy as np




def replace_random(x, min_val, max_val):
    x = x.astype('float')
    x[x>max_val]=np.nan
    x[x<min_val]=np.nan
    idxs = np.argwhere(np.isnan(x))
    x[idxs] = np.random.uniform(min_val, max_val, idxs.shape)
    return x


if __name__=='__main__':
    x = np.random.uniform(-10, 10, (10,10))
    min_val, max_val = 2,8
    x = replace_random(x, min_val, max_val)
    print(x)

    