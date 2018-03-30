
import sys
import time
import numpy as np
import os

from sklearn.metrics import roc_auc_score


# Parameters for progress_bar Init
TOTAL_BAR_LENGTH = 65.

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

last_time = time.time()
begin_time = last_time


def ensure_dir(path):
    import pathlib
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 

def progress_bar(current, total, msg=None):
    ''' print current result of train, valid
    
    Args:
        current (int): current batch idx
        total (int): total number of batch idx
        msg(str): loss and acc
    '''

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()



def format_time(seconds):
    ''' calculate and formating time 

    Args:
        seconds (float): time
    '''

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def stats(outputs, targets):
    ''' Using outputs and targets list, calculate true positive,
        false positive, true negative, false negative, accuracy, 
        recall, specificity, precision, F1 Score, AUC, best Threshold.
        And return them

    Args:
        outputs (numpy array): net outputs list
        targets (numpy array): correct result list

    '''
    
    num = len(np.arange(0,1.005,0.005))

    correct = [0] * num
    tp = [0] * num
    tn = [0] * num
    fp = [0] * num
    fn = [0] * num
    recall = [0] * num
    specificity = [0] * num

    outputs_num = outputs.shape[0]
    for i, threshold in enumerate(np.arange(0, 1.005, 0.005)):
            
        threshold = np.ones(outputs_num) * (1-threshold)
        _outputs = outputs + threshold
        _outputs = np.floor(_outputs)

        tp[i] = (_outputs*targets).sum()
        tn[i] = np.where((_outputs+targets)==0, 1, 0).sum()
        fp[i] = np.floor(((_outputs-targets)*0.5 + 0.5)).sum()
        fn[i] = np.floor(((-_outputs+targets)*0.5 + 0.5)).sum()
        correct[i] += (tp[i] + tn[i])

    thres_cost = fp[0]+fn[0]
    thres_idx = 0

    for i in range(num):
        recall[i] = tp[i] / (tp[i]+fn[i])
        specificity[i] = tn[i] / (fp[i]+tn[i])
        if thres_cost > (fp[i]+fn[i]):
            thres_cost = fp[i]+fn[i]
            thres_idx = i

    correct = correct[thres_idx]
    tp = tp[thres_idx]
    tn = tn[thres_idx]
    fp = fp[thres_idx]
    fn = fn[thres_idx]
    recall = (tp+1e-7)/(tp+fn+1e-7)
    precision = (tp+1e-7)/(tp+fp+1e-7)
    specificity = (tn+1e-7)/(fp+tn+1e-7)
    f1_score = 2.*precision*recall/(precision+recall+1e-7)
    auc = roc_auc_score(targets, outputs) 
    threshold = thres_idx * 0.005

    return correct, tp, tn, fp, fn, recall, precision, specificity, f1_score,auc,threshold



