import torch
from torch.utils.data import Dataset
from os import path
from pandas import read_csv, DataFrame
import numpy as np
from utils import *
import random

class SGCC(Dataset):
    def __init__(self, 
                train_test_rate, 
                local_rank, 
                train=True, 
                distributed=False,
                dropout = 0.2,
                lttb = None):
        '''
        Args:
            local_rank: data load to the device
            train: whether train or test dataset
            distributed: whether to use the distributed training
        '''
        self.train = train

        if train:
            abnormal_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal.csv'
            normal_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal.csv'
            abnormal_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_m.csv'
            normal_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_m.csv'
            abnormal_l_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l'+str(lttb)+'.csv'
            normal_l_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l'+str(lttb)+'.csv'
            if not path.exists(abnormal_path): 
                if distributed:
                    if local_rank==0:   # 让进程0准备数据
                        preparedataset(train_test_rate)
                    torch.distributed.barrier()
                else:
                    preparedataset(train_test_rate)
            print('**load train dataset ')
        else:
            abnormal_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_abnormal.csv'
            normal_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_normal.csv'
            abnormal_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_abnormal_m.csv'
            normal_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_normal_m.csv'
            print('**load test dataset ')
        
        abnormal_data = read_csv(abnormal_path).values
        normal_data = read_csv(normal_path).values
        abnormal_m = read_csv(abnormal_m_path).values
        normal_m = read_csv(normal_m_path).values
        if self.train:
            abnormal_l = read_csv(abnormal_l_path).values
            normal_l = read_csv(normal_l_path).values
        else:
            abnormal_l = None
            normal_l = None

        self.abnormal_y = abnormal_data[:, 0]
        abnormal_x = abnormal_data[:, 1:]
        self.normal_y = normal_data[:, 0]
        normal_x = normal_data[:, 1:]

        self.abnormal_no, self.timesize = abnormal_x.shape
        self.normal_no = self.normal_y.shape[0]
        self.no = self.abnormal_no + self.normal_no
        
        self.abnormal_x2d, abnormal_l2d = creat2Dfeature(abnormal_x, abnormal_m, abnormal_l, 7)
        self.normal_x2d, normal_l2d = creat2Dfeature(normal_x, normal_m, normal_l, 7)
        if self.train: 
            self.abnormal_x2d_new = creat2Crops(self.abnormal_x2d, abnormal_l2d, dropout)
            self.normal_x2d_new = creat2Crops(self.normal_x2d, normal_l2d, dropout)
             
    def __len__(self):
        return self.no

    def __getitem__(self, index):
        if self.train:
            if index % 2 == 0:
                index_new = random.randint(0, self.normal_no - 1)
                Y = self.normal_y[index_new]
                X2D = self.normal_x2d[index_new]
                X2D_new = self.normal_x2d_new[index_new]
            else:
                index_new = random.randint(0, self.abnormal_no - 1)
                Y = self.abnormal_y[index_new]
                X2D = self.abnormal_x2d[index_new]
                X2D_new = self.abnormal_x2d_new[index_new]
        else:
            if index < self.normal_no:
                Y = self.normal_y[index]
                X2D = self.normal_x2d[index]
                X2D_new = self.normal_x2d_new[index]
            else:
                index -= self.normal_no
                Y = self.abnormal_y[index]
                X2D = self.abnormal_x2d[index]
                X2D_new = self.abnormal_x2d_new[index]

        if self.train:
            return (torch.tensor(Y.reshape(1, -1), dtype=torch.long), 
                    torch.Tensor(X2D), torch.Tensor(X2D_new))
        else:
            return (torch.tensor(Y.reshape(1, -1), dtype=torch.long),
                    torch.Tensor(X2D))

    def featuresize(self):
        return self.timesize
            

def preparedataset(train_test_rate):
    '''
    准备数据集
    '''
    print('     Prepare data...')
    datas = read_csv('data/SGCC/data.csv')
    mask = read_csv('data/SGCC/mask.csv')
    nanindex = read_csv('data/SGCC/nanindex.csv').values[:,0]
    lttb5 = read_csv('data/SGCC/lttb_500.csv')
    lttb6 = read_csv('data/SGCC/lttb_600.csv')
    lttb7 = read_csv('data/SGCC/lttb_700.csv')
    lttb8 = read_csv('data/SGCC/lttb_800.csv')
    lttb9 = read_csv('data/SGCC/lttb_900.csv')
    lttb10 = read_csv('data/SGCC/lttb_1000.csv')
    
    datas.drop(nanindex, inplace=True)  # 去除nuinay<5的
    mask.drop(nanindex, inplace=True)
    lttb5.drop(nanindex, inplace=True)
    lttb6.drop(nanindex, inplace=True)
    lttb7.drop(nanindex, inplace=True)
    lttb8.drop(nanindex, inplace=True)
    lttb9.drop(nanindex, inplace=True)
    lttb10.drop(nanindex, inplace=True)
    y_x = datas.values.astype(np.float32)  # datas: 0:label, 1~:data
    m = mask.values[:,1:].astype(np.float32)
    l5 = lttb5.values.astype(int)
    l6 = lttb6.values.astype(int)
    l7 = lttb7.values.astype(int)
    l8 = lttb8.values.astype(int)
    l9 = lttb9.values.astype(int)
    l10 = lttb10.values.astype(int)
    
    # Data Preprocess
    y_x = processOutlier(y_x)
    y_x, _ = normalization_zsocre(y_x)
    y_x = processNan(y_x)
    
    # data split
    pos = y_x[y_x[:,0]==1]
    neg = y_x[y_x[:,0]==0]
    pos_m = m[y_x[:,0]==1]
    neg_m = m[y_x[:,0]==0]
    pos_l5 = l5[y_x[:,0]==1]
    neg_l5 = l5[y_x[:,0]==0]
    pos_l6 = l6[y_x[:,0]==1]
    neg_l6 = l6[y_x[:,0]==0]
    pos_l7 = l7[y_x[:,0]==1]
    neg_l7 = l7[y_x[:,0]==0]
    pos_l8 = l8[y_x[:,0]==1]
    neg_l8 = l8[y_x[:,0]==0]
    pos_l9 = l9[y_x[:,0]==1]
    neg_l9 = l9[y_x[:,0]==0]
    pos_l10 = l10[y_x[:,0]==1]
    neg_l10 = l10[y_x[:,0]==0]
    print('the number of abnormal: ', len(pos))
    print('the number of normal: ', len(neg))
    print('\n')

    train_pos_index = np.random.uniform(0., 1., size=len(pos)) < train_test_rate
    train_pos = pos[train_pos_index]
    test_pos = pos[~train_pos_index]
    train_pos_m = pos_m[train_pos_index]
    test_pos_m = pos_m[~train_pos_index]
    train_pos_l5 = pos_l5[train_pos_index]
    train_pos_l6 = pos_l6[train_pos_index]
    train_pos_l7 = pos_l7[train_pos_index]
    train_pos_l8 = pos_l8[train_pos_index]
    train_pos_l9 = pos_l9[train_pos_index]
    train_pos_l10 = pos_l10[train_pos_index]
    print('the number of train abnormal: ', len(train_pos))
    print('the number of test abnormal: ', len(test_pos))
    print('\n')

    train_neg_index = np.random.uniform(0., 1., size=len(neg)) < train_test_rate
    train_neg = neg[train_neg_index]
    test_neg = neg[~train_neg_index]
    train_neg_m = neg_m[train_neg_index]
    test_neg_m = neg_m[~train_neg_index]
    train_neg_l5 = neg_l5[train_neg_index]
    train_neg_l6 = neg_l6[train_neg_index]
    train_neg_l7 = neg_l7[train_neg_index]
    train_neg_l8 = neg_l8[train_neg_index]
    train_neg_l9 = neg_l9[train_neg_index]
    train_neg_l10 = neg_l10[train_neg_index]
    print('the number of train normal: ', len(train_neg))
    print('the number of test normal: ', len(test_neg))
    print('\n')

    train_pos_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal.csv'
    train_neg_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal.csv'
    test_pos_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_abnormal.csv'
    test_neg_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_normal.csv'
    train_pos_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_m.csv'
    train_neg_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_m.csv'
    test_pos_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_abnormal_m.csv'
    test_neg_m_path = 'data/SGCC/processed/'+str(train_test_rate)+'/test_normal_m.csv'
    train_pos_l5_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l5.csv'
    train_neg_l5_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l5.csv'
    train_pos_l6_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l6.csv'
    train_neg_l6_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l6.csv'
    train_pos_l7_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l7.csv'
    train_neg_l7_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l7.csv'
    train_pos_l8_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l8.csv'
    train_neg_l8_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l8.csv'
    train_pos_l9_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l9.csv'
    train_neg_l9_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l9.csv'
    train_pos_l10_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_abnormal_l10.csv'
    train_neg_l10_path = 'data/SGCC/processed/'+str(train_test_rate)+'/train_normal_l10.csv'
    DataFrame(train_pos).to_csv(train_pos_path, index=False)
    DataFrame(train_neg).to_csv(train_neg_path, index=False)
    DataFrame(test_pos).to_csv(test_pos_path, index=False)
    DataFrame(test_neg).to_csv(test_neg_path, index=False)
    DataFrame(train_pos_m).to_csv(train_pos_m_path, index=False)
    DataFrame(train_neg_m).to_csv(train_neg_m_path, index=False)
    DataFrame(test_pos_m).to_csv(test_pos_m_path, index=False)
    DataFrame(test_neg_m).to_csv(test_neg_m_path, index=False)
    DataFrame(train_pos_l5).to_csv(train_pos_l5_path, index=False)
    DataFrame(train_neg_l5).to_csv(train_neg_l5_path, index=False)
    DataFrame(train_pos_l6).to_csv(train_pos_l6_path, index=False)
    DataFrame(train_neg_l6).to_csv(train_neg_l6_path, index=False)
    DataFrame(train_pos_l7).to_csv(train_pos_l7_path, index=False)
    DataFrame(train_neg_l7).to_csv(train_neg_l7_path, index=False)
    DataFrame(train_pos_l8).to_csv(train_pos_l8_path, index=False)
    DataFrame(train_neg_l8).to_csv(train_neg_l8_path, index=False)
    DataFrame(train_pos_l9).to_csv(train_pos_l9_path, index=False)
    DataFrame(train_neg_l9).to_csv(train_neg_l9_path, index=False)
    DataFrame(train_pos_l10).to_csv(train_pos_l10_path, index=False)
    DataFrame(train_neg_l10).to_csv(train_neg_l10_path, index=False)

    print('   Prepare data done! ')

