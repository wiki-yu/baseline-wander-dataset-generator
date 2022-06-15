import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle


class Qt_DB_Heartbeats(object):
    def __init__(self) -> None:
         pass
    def func1():
        pass


qt_path = 'data/qt-database-1.0.0/'
new_fs = 360

files_path = glob.glob(qt_path + "/*.dat")
qt_db_signlas = dict()
file_name = None

for file_path in files_path:
    # Reading signals
    path_parts = file_path.split(".dat")
    # print(path_parts)
    file_name = path_parts[0].split("/")[-1]
    print("current file name: ", file_name)
    signal, fields = wfdb.rdsamp(path_parts[0])
    sig_len = len(signal)
    # print("signal length: ", sig_len)
    # print("current sampling frequency:", fields['fs'])

    # Reading annotations
    # wfdb.rdann(record_name, extension, sampfrom=0, sampto=None, shift_samps=False, pn_dir=None, return_label_elements=['symbol'], summarize_labels=False)
    ann = wfdb.rdann(path_parts[0], "pu1")
    ann_type = ann.symbol
    ann_samples = ann.sample
    # print("len ann_type: ", len(ann_type))
    # print("len ann_samples: ", len(ann_samples))
    # print("ann_type: ", ann_type)
    # print("ann_samples: ", ann_samples)

    # Obtaining P wave start positions
    ann_type = np.array(ann_type)
    idx1 = ann_type == "p"
    # print("p wave idx: ", idx1)
    # idx1:  [False  True False ... False False False]

    p_idx = ann_samples[idx1]
    # print("p wave index: ", p_idx)
    # p_idx1:  [194   1058   1190   1632 ...]

    idx2 = ann_type == "("
    s_idx = ann_samples[idx2]

    idx3 = ann_type == "N"
    r_idx = ann_samples[idx3]

    p_idx_arr = np.zeros(len(p_idx))

    for i in range(len(p_idx_arr)):
        arr = np.where(p_idx[i] > s_idx)
        arr = arr[0]
        p_idx_arr[i] = arr[-1]
    
    p_idx_arr = p_idx_arr.astype(np.int64)
    p_start = s_idx[p_idx_arr]
    # print("p wave start index: ", p_start)

    p_start = p_start - int(0.04 * fields['fs'])
    signal_1ch = signal[0:sig_len, 0]

    heartbeats_raw = list()
    for k in range(len(p_start) - 1):
        remove = (r_idx > p_start[k] & (r_idx < p_start[k]))
        if np.sum(remove) < 2:
            heartbeats_raw.append(signal_1ch[p_start[k]: p_start[k + 1]])
    
    heartbeats = list()
    for k in range(len(heartbeats_raw)):
        L = math.ceil(len(heartbeats_raw[k] * new_fs / fields["fs"]))
        norm_heartbeat = list(reversed(heartbeats_raw[k]) + list(heartbeats_raw[k]) + list(reversed(heartbeats_raw[k])))
        res = resample_poly(norm_heartbeat, new_fs, fields['fs'])
        res = res[L-1: 2*L-1 ]
        heartbeats.append(res)
    
    qt_db_signlas[file_name] = heartbeats


with open("data/qt_db_heartbeats.pkl", "wb") as output:
        pickle.dump(qt_db_signlas, output)
print("MIT QT DB saved as pickle file!")















