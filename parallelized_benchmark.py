import torch
import torch.nn as nn

import numpy as np
from ts2 import TemperatureScaling
from Calibrators import VectorScaling, MatrixScaling
from mix_match_calib import ets_calibrate
from parameterized_temp_scaling import ParameterizedTemperatureScaling

from netcal.binning import HistogramBinning

from pycalib.metrics import ECE
from eq_bin_ece import ECE_eq_bin
from ECE_kde import ece_kde_binary



import pickle
import os

from scipy.special import softmax

""
import warnings
import argparse
import pandas as pd
import random
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# methods 0 for ts 1 for vs 2 for ETS 3 for ETS mse 4 for PTS 5 for mono_vs 6 for mono_vs_inv 7 for ms 8 for HB 9 for DIAG
parser.add_argument("--method", required=True, type=int, help="calibration method to benchmark")

parser.add_argument("--seed", required=True, type=int, help="seed for running ")

args = parser.parse_args()



def unpickle_probs(file, verbose = 0):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)  # unpickle the content
        
    if verbose:    
        print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels
        
    return ((y_probs_val, y_val), (y_probs_test, y_test))

class logger:
    def __init__(self, algo_list, datasets, log_name, dirname):
        self.df = pd.DataFrame(columns=algo_list, index=datasets)
        self.n_algo = len(algo_list)
        self.n_dataset = len(datasets)
        self.log_name = log_name
        self.dir_name = dirname
    def log(self, dataset, results):
        # check if the dataset is in the 
        self.df.loc[dataset] = results
        self.df.to_csv(f"{self.dir_name}/{self.log_name}.csv")



def measure_error(probs, labels, num_bin=15):
    ece = ECE(labels, probs, ece_full = False, bins=num_bin)
    
    pw_ece,prob_start = ECE_eq_bin(probs,labels, bin_size=1000)
    # need to take the one hot encoding of the labels
    num_classes = np.max(labels) + 1  # Assuming labels start from 0
    one_hot_test = np.eye(num_classes)[labels]
    ece_kde = ece_kde_binary(probs, one_hot_test)
    return ece, ece_kde, np.array(pw_ece).sum()


from Mono_cali import MCCT_I, MCCT


models = ['resnet50', 'eff_net', 'resnet152', 'vitb16',  'inception_v3', 'resnet_wide110', 'densenet40_c10',
            'lenet5_c10', 'resnet_wide32_c10', 'resnet110_c10',
             'resnet_wide32_c100', 'resnet110_c100']
folder_path = "./../imagenet_logits/"

save_dir_name = "./results2"

methods_list = ["TS", "VS", "ETS", "ETS_mse", "PTS", "mono_vs", "mono_vs_inv", "HB","DIAG"]

method = args.method
random_seed = args.seed

my_log = logger(["ECE", "KDE-ECE", "EQ-bin_ECE"], models, f"{methods_list[method]}_seed{random_seed}", save_dir_name)



for a in range(len(models)):
    ((logits_vali, vali_lables), (logits, test_lables)) = unpickle_probs(f"{folder_path}{models[a]}_logits.p")
    vali_lables = vali_lables.flatten()
    test_lables = test_lables.flatten()
    
    num_classes = np.max(test_lables) + 1  # Assuming labels start from 0
    one_hot_test = np.eye(num_classes)[test_lables]
    one_hot_vali = np.eye(num_classes)[vali_lables]
    print(f"for {models[a]}")

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    
    # print("------------------------------Uncalibrated------------------------------------")
   
    ece_ori, ece_kde_ori, eq_ece_ori = measure_error(softmax(logits, axis=1), test_lables)
    
    # print("-----------------------------------Mono VS------------------------------------")

    # there are so many bugs in PBMC that I am not going to use it
    # methods 0 for ts 1 for vs 2 for ETS 3 for ETS mse 4 for PTS 5 for mono_vs 6 for mono_vs_inv 7 for ms 8 for HB 9 for DIAG

    match method:
        case 0:
            # TS
            TS = TemperatureScaling()
            TS.fit(logits_vali, one_hot_vali)
            TS_result = TS.predict(logits)
            ece_ts, ece_kde_ts, eq_ece_ts = measure_error(TS_result, test_lables)
            my_log.log(models[a], [ece_ts, ece_kde_ts, eq_ece_ts])

        case 1:
            # VS
            VS = VectorScaling(num_label=len(np.unique(vali_lables)))
            VS.fit(logits_vali, vali_lables)
            VS_result = VS.calibrate(logits)
            VS_result = softmax(VS_result, axis=1)
            ece_vs, ece_kde_vs, eq_ece_vs = measure_error(VS_result, test_lables)
            my_log.log(models[a], [ece_vs, ece_kde_vs, eq_ece_vs])


        case 2:
            ETS_res = ets_calibrate(logits_vali,one_hot_vali,logits, one_hot_vali.shape[1], 'ce')
            ece_ets, ece_kde_ets, eq_ece_ets = measure_error(ETS_res, test_lables)
            my_log.log(models[a], [ece_ets, ece_kde_ets, eq_ece_ets])


        case 3:
            ETS_res_mse = ets_calibrate(logits_vali,one_hot_vali,logits, one_hot_vali.shape[1], 'mse')
            ece_ets_mse, ece_kde_ets_mse, eq_ece_ets_mse = measure_error(ETS_res_mse, test_lables)
            my_log.log(models[a], [ece_ets_mse, ece_kde_ets_mse, eq_ece_ets_mse])

        case 4:
            # parametized TS pytorch implementation. Same set up as the paper
            PTS_calibrator = ParameterizedTemperatureScaling(
                    epochs=200, # stepsize = 100,000
                    lr=0.00005,
                    batch_size=1000,
                    nlayers=2,
                    n_nodes=5,
                    length_logits=one_hot_vali.shape[1],
                    top_k_logits=10
            )
            #hard label
            PTS_calibrator.tune(logits_vali, vali_lables)
            PTS_res = PTS_calibrator.calibrate(logits)
            ece_pts, ece_kde_pts, eq_ece_pts = measure_error(PTS_res, test_lables)
            my_log.log(models[a], [ece_pts, ece_kde_pts, eq_ece_pts])


        case 5:
            mcct = MCCT(topk=num_classes, maxiter=100, bounds=False, filter=False)
            mcct.fit(logits_vali, one_hot_vali)
            mcct_res = mcct.predict(logits)
            ece_mcct, ece_kde_mcct, eq_ece_mcct = measure_error(mcct_res, test_lables)
            my_log.log(models[a], [ece_mcct, ece_kde_mcct, eq_ece_mcct])

        case 6:
            mcct_i = MCCT_I(topk=num_classes, maxiter=100, bounds=False, filter=False)
            mcct_i.fit(logits_vali, one_hot_vali)
            mcct_i_res = mcct_i.predict(logits)
            ece_mi, ece_kde_mi, eq_ece_mi = measure_error(mcct_i_res, test_lables)
            my_log.log(models[a], [ece_mi, ece_kde_mi, eq_ece_mi])
        case 7:
            hb = HistogramBinning()
            hb.fit(softmax(logits_vali, axis=1), vali_lables)
            hb_result = hb.transform(softmax(logits, axis=1))
            ece_HB, ece_kde_HB, eq_ece_hb = measure_error(hb_result, test_lables)
            my_log.log(models[a], [ece_HB, ece_kde_HB, eq_ece_hb])
        case 8:
            from DIAG import diag
            from torch.utils.data import DataLoader, TensorDataset
            # init the model
            device = "cuda:0"
            loss_fn = nn.CrossEntropyLoss()
            model = diag.MonotonicModel(num_hiddens = [10,10], conditioned = False, add_condition_to_integrand = False, nb_steps=30, device=device, num_classes=num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0)
            model.train()
            # validation set dataloader
            logits_vali_tensor = torch.tensor(logits_vali, dtype=torch.float32)
            vali_labels_tensor = torch.tensor(vali_lables, dtype=torch.long)
            dataset = TensorDataset(logits_vali_tensor, vali_labels_tensor)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            # testing set dataloader
            logits_test_tensor = torch.tensor(logits, dtype=torch.float32)
            test_labels_tensor = torch.tensor(test_lables, dtype=torch.long)
            test_dataset = TensorDataset(logits_test_tensor, test_labels_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            # 15 epoch as the original inplementation
            from torch.autograd import Variable

            for epoch in range(15):
                for i, (train_batch, labels_batch) in enumerate(dataloader):
                    # move to GPU if available
                    
                    train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
                    # convert to torch Variables
                    train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

                    # compute model output and loss (outputs are logits)
                    output_batch = model(train_batch)
                    loss = loss_fn(output_batch, labels_batch)
                    optimizer.zero_grad()
                    loss.backward()

                    # performs updates using calculated gradients
                    optimizer.step()

            results = []
            for i, (test_batch, test_labels_batch) in enumerate(test_dataloader):
                # move to GPU if available
                
                test_batch = test_batch.to(device)

                result = model.forward(test_batch)
                softmax_res = softmax(result.detach().cpu().numpy(), axis=1)
                results.append(softmax_res)

            DIAG_result = np.concatenate(results, axis=0)

            ece_DIAG, ece_kde_DIAG, eq_ece_DIAG = measure_error(DIAG_result, test_lables)
            my_log.log(models[a], [ece_DIAG, ece_kde_DIAG, eq_ece_DIAG])
        case _:
            print("wrong calibrator")
            exit()