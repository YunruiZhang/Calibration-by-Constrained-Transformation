import numpy as np
from ts2 import TemperatureScaling
from scipy.special import softmax

def create_bin_array(n, bin_size):
    bin_array = []
    bin_array.append(0)
    current_sum = 0
    
    # Add bin_size until just before the last bin
    while current_sum + bin_size * 2 <= n:
        current_sum += bin_size
        bin_array.append(current_sum)
    
    # Calculate the last bin: bin_size + remainder
    remainder = n - current_sum
    last_bin = current_sum + remainder
    bin_array.append(last_bin)
    
    return np.array(bin_array)




def _compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    # print(f"{conf_thresh_lower} and {conf_thresh_upper}")
    """
    # Computes accuracy and average confidence for bin

    Parameters
    ==========
    conf_thresh_lower (float):
        Lower Threshold of confidence interval
    conf_thresh_upper (float):
        Upper Threshold of confidence interval
    conf (numpy.ndarray):
        list of confidences
    pred (numpy.ndarray):
        list of predictions
    true (numpy.ndarray):
        list of true labels
    Returns
    =======
    (accuracy, avg_conf, len_bin) :
        accuracy of bin, confidence of bin and number of elements in bin.
    """
    # filtered_tuples = [x for x in zip(pred, true, conf)
    #                    if (x[2] > conf_thresh_lower or conf_thresh_lower == 0)
    #                    and (x[2] <= conf_thresh_upper)]
    
    filtered_tuples = (pred[conf_thresh_lower:conf_thresh_upper],true[conf_thresh_lower:conf_thresh_upper], conf[conf_thresh_lower:conf_thresh_upper])
    
    # How many correct labels
    # correct = len([x for x in filtered_tuples if x[0] == x[1]])
    correct = (filtered_tuples[0] == filtered_tuples[1]).sum()
    # How many elements falls into given bin
    # len_bin = len(filtered_tuples)
    len_bin = len(filtered_tuples[0])
    # Avg confidence of BIN
    # avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin
    avg_conf = filtered_tuples[2].mean()
    # accuracy of BIN
    accuracy = float(correct)/len_bin

    return accuracy, avg_conf, len_bin, filtered_tuples[2][0]

# weight by the width of the bin?
def ECE_eq_bin(conf, true, bin_size=100):

    """
    Expected Calibration Error

    Parameters
    ==========
    conf (numpy.ndarray):
        list of confidences
    true (numpy.ndarray):
        list of true labels
    bin_size: (int):
        size of each bin

    Returns
    =======
    ece: expected calibration error
    """
    # top label ECE
    pred = np.argmax(conf, axis=1)
    conf = np.max(conf, axis=1) 
    
    # sort the conf and pred and True accordingly
    sorted_indices = np.argsort(conf)

    conf_sorted = conf[sorted_indices]
    pred_sorted = pred[sorted_indices]
    true_sorted = true[sorted_indices]
    

    n = len(conf)
    upper_bounds = create_bin_array(n, bin_size)
    ece = []  # Starting error
    prob_start = []
    for i in range(1,len(upper_bounds)):  # Find accur. and confidences per bin
        # print(conf_thresh)
        # this need to be fixed this - binsize thing does not work for the last bin
        acc, avg_conf, len_bin,p_start = _compute_acc_bin(upper_bounds[i-1],
                                                  upper_bounds[i], conf_sorted, pred_sorted,
                                                  true_sorted)
        # return filtered_tuples
        # ece += np.abs(acc-avg_conf)/len_bin  # Add weigthed difference to ECE
        ece.append(np.abs(avg_conf-acc))
        prob_start.append(p_start)

    return ece,prob_start


