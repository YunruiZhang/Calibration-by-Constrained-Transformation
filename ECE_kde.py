#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:58:58 2019

@author: zhang64
"""


import torch
import numpy as np
import torch.nn.parallel

from KDEpy import FFTKDE


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return np.concatenate((d, 2*xmax-d))
    else:
        return d


def ece_kde_binary(p,label,p_int=None,order=1):

    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p,1e-256,1-1e-256)
    p_int = np.clip(p_int,1e-256,1-1e-256)
    
    
    x_int = np.linspace(-0.6, 1.6, num=2**14)
    
    
    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    label_index = np.array([np.where(r==1)[0][0] for r in label])
    with torch.no_grad():
        if p.shape[1] !=2:
            p_new = torch.from_numpy(p)
            p_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
        else:
            p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index
                
    method = 'triweight'
    
    dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
    kbw = np.std(p_b.numpy())*(N*2)**-0.2
    kbw = np.std(dconf_1)*(N*2)**-0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1
    
    
    p_int = p_int/np.sum(p_int,1)[:,None]
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1,1))
        if p_int.shape[1]!=2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i,pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i,1]

    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    
    if p.shape[1] !=2: # top label (confidence)
        perc = np.mean(label_binary)
    else: # or joint calibration for binary cases
        perc = np.mean(label_index)
            
    integral = np.zeros(x_int.shape)
    reliability= np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]
        if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu)==False:
                integral[i] = np.abs(conf-accu)**order*pp2[i]  
                reliability[i] = accu
        else:
            if i>1:
                integral[i] = integral[i-1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
