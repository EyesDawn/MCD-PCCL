import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
def load_UCR(dataset):
    train_file = os.path.join('../../ts_data/UCRArchive_2018/', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('../../ts_data/UCRArchive_2018/', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'../../ts_data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'../../ts_data/Multivariate_arff/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    print(train_X.shape)
    return train_X, train_y, test_X, test_y

def load_HEI(dataset,mode="train",dataset_root="../../CA-TCC/data"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"{dataset_root}/{dataset}/"
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    if len(train_X.shape)==2:
        train_X = train_X.unsqueeze(1)
    train_X = torch.transpose(train_X,2,1)
    #train_X = train_X[:, ::3, :]
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    if len(test.shape)==2:
        test = test.unsqueeze(1)
    test_X = torch.transpose(test, 1, 2)
    #test_X = test_X[:, ::3, :]
    test_y = test_['labels']

    
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    print("train_X.shape",train_X.shape)
    print("test_X.shape",test_X.shape)

    return train_X, train_y, test_X, test_y

def load_HEI_fft(dataset,mode="train",dataset_root="../../CA-TCC/data"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"{dataset_root}/{dataset}/"

    print(data_path)
    print(mode != "train",mode)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")

    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    if len(train_X.shape)==2:
        train_X = train_X.unsqueeze(1)
    train_X = torch.transpose(train_X,2,1)
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test_X = test_['samples']
    if len(test_X.shape)==2:
        test_X = test_X.unsqueeze(1)
    test_X = torch.transpose(test_X, 1, 2)
    print(test_X.shape)
    test_y = test_['labels']
    print(test_y)
    # scaler = StandardScaler()
    # scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    # train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    # test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    # train_X = torch.from_numpy(train_X)
    # test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    print("train_X.shape",train_X.shape)
    print("train_X_fft.shape",train_X.shape)
    return [train_X, train_X_fft], train_y, [test_X,test_X_fft], test_y


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data

from torch.utils.data import Dataset

class TwoViewloader(Dataset):
    """
    Return the dataitem and corresponding index
    The batch of the loader: A list
        - [B, L, 1] (For univariate time series)
        - [B]: The corresponding index in the train_set tensors

    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample_tem = self.data[0][index]
        sample_fre = self.data[1][index]


        return index, sample_tem, sample_fre

    def __len__(self):
        return len(self.data[0])

class ThreeViewloader(Dataset):
    """
    Return the dataitem and corresponding index
    The batch of the loader: A list
        - [B, L, 1] (For univariate time series)
        - [B]: The corresponding index in the train_set tensors

    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample_tem = self.data[0][index]
        sample_fre = self.data[1][index]
        sample_sea = self.data[2][index]


        return index, sample_tem, sample_fre,sample_sea

    def __len__(self):
        return len(self.data[0])


def load_tri_view(dataset="ISRUC",mode="train",decompose_mode="seasonal"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"../../CA-TCC/data/{dataset}/"
        
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    train_X = torch.transpose(train_X,2,1)
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    if decompose_mode == "seasonal":
        if mode!="train":
            train_sea_ = torch.load(data_path + f"train_{mode}_sea.pt")
        else:
            train_sea_ = torch.load(data_path + "train_sea.pt")
        test_sea_ = torch.load(data_path + "test_sea.pt")
        train_X_sea = train_sea_['samples']
        test_X_sea = test_sea_['samples']
        
        # if mode!="train":
        #     train_trend_ = torch.load(data_path + f"train_{mode}_trend.pt")
        # else:
        #     train_trend_ = torch.load(data_path + "train_trend.pt")
        # test_trend_ = torch.load(data_path + "test_trend.pt")
        # train_X_trend = train_trend_['samples']
        # test_X_trend = test_trend_['samples']

        # if mode!="train":
        #     train_resid_ = torch.load(data_path + f"train_{mode}_resid.pt")
        # else:
        #     train_resid_ = torch.load(data_path + "train_resid.pt")
        # test_resid_ = torch.load(data_path + "test_resid.pt")
        # train_X_resid = train_resid_['samples']
        # test_X_resid = test_resid_['samples']

    elif decompose_mode == "trend":
        if mode!="train":
            train_trend_ = torch.load(data_path + f"train_{mode}_trend.pt")
        else:
            train_trend_ = torch.load(data_path + "train_trend.pt")
        test_trend_ = torch.load(data_path + "test_trend.pt")
        train_X_sea = train_trend_['samples']
        test_X_sea = test_trend_['samples']
    elif decompose_mode == "resid":
        if mode!="train":
            train_resid_ = torch.load(data_path + f"train_{mode}_resid.pt")
        else:
            train_resid_ = torch.load(data_path + "train_resid.pt")
        test_resid_ = torch.load(data_path + "test_resid.pt")
        train_X_sea = train_resid_['samples']
        test_X_sea = test_resid_['samples']
    
    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
        # 频域
    print(type(train_X))
    print(train_X.shape)
    # train_X = torch.from_numpy(train_X)
    # test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    print("train_X.shape",train_X.shape)
    print("train_X_fft.shape",train_X.shape)
    print("train_X_sea.shape",train_X_sea.shape)
    #return [train_X,train_X_sea, train_X_fft], train_y, [test_X,test_X_sea,test_X_fft], test_y
    return [train_X,train_X_sea, train_X_fft], train_y, [test_X,test_X_sea,test_X_fft], test_y
