import subprocess
import pandas as pd
from io import StringIO
import numpy as np
import torch 
import random


def get_free_gpu(num=1):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf8')),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df = gpu_df.sort_values(by='memory.free', ascending=False)
    print('GPU usage:\n{}'.format(gpu_df))
    if len(gpu_df) < num:
        raise RuntimeError('No enough GPU')
    free_gpus = []
    for i in range(num):
        print('Returning GPU{} with {} free MiB'.format(gpu_df.index[i], gpu_df.iloc[i]['memory.free']))
        free_gpus.append(gpu_df.index[i])
    return ','.join(str(x) for x in free_gpus)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def normalization_minmax (data, parameters=None):
    '''Normalize data in [0, 1] range for each user.
    norm_x = (x-min)/(max-min)
    
    Args:
        - data: shape[user_num, feature_num]
    
    Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each user for renormalization
    '''
    print('normalize')
    # Parameters
    no, _ = data.shape
    norm_data = data.copy()
    # labels = norm_data[:,0]
    # norm_data[:,0] = np.nan
    
    if parameters is None:
        min_val = np.zeros(no)
        max_val = np.zeros(no)
        # For each user
        for i in range(no):
            # print(i)
            min_val[i] = np.nanmin(norm_data[i][1:])
            norm_data[i][1:] = norm_data[i][1:] - min_val[i]                # x - min
            max_val[i] = np.nanmax(norm_data[i][1:])                    # max-min
            norm_data[i][1:] = norm_data[i][1:] / (max_val[i] + 1e-6)   # +1e-6是为了避免0值在分母 
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                        'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        # For each dimension
        for i in range(no):
            norm_data[i][1:] = norm_data[i][1:] - min_val[i]
            norm_data[i][1:] = norm_data[i][1:] / (max_val[i] + 1e-6)  
        
        norm_parameters = parameters    
        
    # norm_data[:,0] = labels
    return norm_data, norm_parameters

def renormalization_minmax (norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.
    
    Args:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each user for renormalization
    
    Returns:
        - renorm_data: renormalized original data
    '''
    
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    no, _ = norm_data.shape
    renorm_data = norm_data.copy()
        
    for i in range(no):
        renorm_data[i][1:] = renorm_data[i][1:] * (max_val[i] + 1e-6)   
        renorm_data[i][1:] = renorm_data[i][1:] + min_val[i]
        
    return renorm_data

def normalization_zsocre (data, parameters=None):
    '''对每一个用户归一化为均值为0，方差为1的正态分布.
    norm_x = (x-mean)/std

    Args:
        - data: shape[user_num, feature_num]
    
    Returns:
        - norm_data: normalized data
        - norm_parameters: mean_val, std_val for each user for renormalization
    '''
    print('normalize')
    # Parameters
    no, _ = data.shape
    norm_data = data.copy()
    
    if parameters is None:
        mean_val = np.zeros(no)
        std_val = np.zeros(no)
        # For each user
        for i in range(no):
            # print(i)
            mean_val[i] = np.nanmean(norm_data[i][1:])
            std_val[i] = np.nanstd(norm_data[i][1:])
            norm_data[i][1:] = (norm_data[i][1:] - mean_val[i]) / (std_val[i] + 1e-6)  # +1e-6是为了避免0值在分母 
        # Return norm_parameters for renormalization
        norm_parameters = {'mean_val': mean_val,
                        'std_val': std_val}

    else:
        mean_val = parameters['mean_val']
        std_val = parameters['std_val']
        # For each dimension
        for i in range(no):
            norm_data[i][1:] = (norm_data[i][1:] - mean_val[i]) / (std_val[i] + 1e-6)  
        
        norm_parameters = parameters    
        
    return norm_data, norm_parameters

def processNan(data):

    print('process Nan value!')
    
    return np.nan_to_num(data, 0)

def processOutlier(data):
    '''
    use 3-sigma to process Outlier 
    the rule:
        xi = min(avg(X) + 2std(X), xi)
    '''
    print('process Outlier!')
    no, dim = data.shape
    for i in range(no):
        avg = np.mean(data[i][1:])
        std = np.std(data[i][1:])
        # std = np.sqrt(np.sum((data[i]-avg)**2)/dim)
        data[i][1:][data[i][1:]>avg + 2*std] = avg + 2*std
    
    return data

def creat2Dfeature(data, mask, lttb, period):
    '''
    transfer data to 2D feature image according to peroid
    
    Args:
        data: shape[no, 1035]
        mask: shape[no, 1035]
        lttb: shape[no, 500/600/700/800/900/1000]
        peroid: int

    Return: 
        2Dfeature:shape[no, 2, dim//peroid, peroid]
    '''
    no, dim = data.shape
    assert dim==1035
    data = np.concatenate((data, np.zeros((no, 1))), axis=1)
    mask = np.concatenate((mask, np.ones((no, 1))), axis=1)
    if lttb is not None:
        temp = np.ones((no, dim+1))
        for i in range(no):
            temp[i][lttb[i]] = 0
        lttb = temp.reshape(no, 1036//period, period)

    return np.stack((data.reshape(no, 1036//period, period), mask.reshape(no, 1036//period, period)), axis=1), \
            lttb

def creat2Crops(data, lttb, dropout=0.2):
    '''
    Create two dropout of the same data

    Args:
        data: [N, 2, H, W]
        lttb: [N, H, W]
        num: int
        dropout: float
    
    Return:
        new_data: [N, 2, 2, H, W]
    '''
    x, mask = data[:,0,:], data[:,1,:]  # mark, 1 mean missed
    N, H, W = mask.shape
    temp_mask1 = (np.random.uniform(0., 1., size=(N, H, W)) > dropout) * 1
    # temp_mask2 = (np.random.uniform(0., 1., size=(N, H, W)) > 0.1) * 1
    new_x1 = x * temp_mask1
    # new_x2 = x * temp_mask2
    mask1 = (1-mask) * temp_mask1
    # mask2 = 1-((1-mask) * temp_mask2)
    data1 = np.stack((new_x1, mask1), axis=1)   # [N, 2, H, W]
    # data2 = np.stack((new_x2, mask2), axis=1) # [N, 2, H, W]
    new_x2 = x * (1-lttb)
    mask2 = (1-mask)*(1-lttb)
    data2 = np.stack((new_x2, mask2), axis=1)   # [N, 2, H, W]
    new_data = np.stack((data1, data2), axis=1) # [N, 2, 2, H, W]
    # new_data = np.stack((data, data1), axis=1)
    return new_data


def process_conv_kernel(data):
    b, c, n, f = data.shape
    data_tamp = np.zeros((b, c, n+2, f+2))
    data_tamp[:,:, :n, :f] = data
    data = data_tamp
    data1 = np.zeros((b, c, n*3, f*3))
    data2 = np.zeros((b, c, n*3, f*3))
    
    for i in range(n):
        for j in range(f):
            data1[:, :, i*3, j*3:(j+1)*3] = 2*data[:, :, i, j:j+3] - data[:, :, i+1, j:j+3] - data[:, :, i+2, j:j+3]
            data1[:, :, i*3+1, j*3:(j+1)*3] = 2*data[:, :, i+1, j:j+3] - data[:, :, i, j:j+3] - data[:, :, i+2, j:j+3]
            data1[:, :, i*3+2, j*3:(j+1)*3] = 2*data[:, :, i+2, j:j+3] - data[:, :, i, j:j+3] - data[:, :, i+1, j:j+3]

            data2[:, :, i*3:(i+1)*3, j*3] = 2*data[:, :, i:i+3, j] - data[:, :, i:i+3, j+1] - data[:, :, i:i+3, j+2]
            data2[:, :, i*3:(i+1)*3, j*3+1] = 2*data[:, :, i:i+3, j+1] - data[:, :, i:i+3, j] - data[:, :, i:i+3, j+2]
            data2[:, :, i*3:(i+1)*3, j*3+2] = 2*data[:, :, i:i+3, j+2] - data[:, :, i:i+3, j] - data[:, :, i:i+3, j+1]

    return data1+data2

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



