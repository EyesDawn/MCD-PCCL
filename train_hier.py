import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from datetime import datetime as timesnow
from ts2vec_hier import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import logging
def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback
def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--warmup', default=0.5, type=float, help='Warmup epoch before using co-training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--data_perc', type=str, default="train", help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--model_path', type=str, default=None, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--ma_gamma', default=0.9999, type=float, help='The moving average parameter for prototype updating')
    parser.add_argument('--temperature', default=0.1, type=float, help='softmax temperature of InfoNCE')
    args = parser.parse_args()
        
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    run_dir = 'exp_log/dual_ld/' + args.dataset + '/' + name_with_datetime(args.run_name+'_'+args.data_perc)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file_name = os.path.join(run_dir, f"logs_{timesnow.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.dataset}')
    logger.debug(f'Mode:    {args.run_name}')
    logger.debug(f'Data Percent:    {args.data_perc}')
    logger.debug("=" * 45)
    logger.debug(args)
    args.num_cluster='6'
    # args.warmup = int(args.epochs*0.5)
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_HEI_fft(args.dataset,dataset_root="/workspace/CA-TCC/data/UCR")
        args.num_cluster =str(np.concatenate((train_labels,test_labels),axis=0).max() + 1)
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_HEI_fft(args.dataset,dataset_root="/workspace/CA-TCC/data/UEA")
        args.num_cluster = str(np.concatenate((train_labels,test_labels),axis=0).max() + 1)
        
    elif args.loader in ['HAR','Epilepsy','ISRUC']:
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_HEI_fft(args.dataset)
        args.num_cluster = str(np.concatenate((train_labels,test_labels),axis=0).max() + 1)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            
            train_data[0] = data_dropout(train_data[0], args.irregular)
            test_data[0] = data_dropout(test_data[0], args.irregular)
            train_data[1] = data_dropout(train_data[1], args.irregular)
            test_data[1] = data_dropout(test_data[1], args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        args=args
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)


    
    t = time.time()
    
    model = TS2Vec(
        input_dims=train_data[0].shape[-1],
        device=device,
        **config
    )
    if not args.eval :

    
        loss_log = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True,
            logger=logger
        )
        model.save(f'{run_dir}/model.pkl')
        args.eval = True
    else:
        model.load(args.model_path)

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            if args.loader == 'UCR':
                train_data, train_labels, test_data, test_labels = datautils.load_HEI_fft(args.dataset,args.data_perc,dataset_root="/workspace/CA-TCC/data/UCR")
                
            elif args.loader == 'UEA':
                train_data, train_labels, test_data, test_labels = datautils.load_HEI_fft(args.dataset,args.data_perc,dataset_root="/workspace/CA-TCC/data/UEA")
                
            elif args.loader in ['HAR','Epilepsy','ISRUC']:
                train_data, train_labels, test_data, test_labels = datautils.load_HEI_fft(args.dataset,args.data_perc)
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        logger.debug("================== result ======================")
        for metric ,val in eval_res.items():
            logger.debug(f"==== {metric} : {val}")
        logger.debug("================== result ======================")
        print('Evaluation result:', eval_res)

    print("Finished.")
