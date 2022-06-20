import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math
import _pickle as pickle

from utils.visualization import plot_segments


class Qt_DB_Heartbeats(object):
    def __init__(self, qt_path):
         self.qt_path = qt_path
         self.qt_db_signlas = dict()
  
    def create_qt_heartbeats(self):
        new_fs = 360

        files_path = glob.glob(self.qt_path + "/*.dat")
        # qt_db_signlas = dict()
        file_name = None

        for file_path in files_path:
            # Reading signals
            path_parts = file_path.split(".dat")
            print(path_parts)
            # file_name = path_parts[0].split("/")[-1]  # linux 
            file_name = path_parts[0].split("/")[-1]  # windows
            print("current file name: ", file_name)
            signal, fields = wfdb.rdsamp(path_parts[0])
            sig_len = len(signal)
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

            ind = np.zeros(len(p_idx))

            for j in range(len(p_idx)):
                arr = np.where(p_idx[j] > s_idx)
                arr = arr[0]
                ind[j] = arr[-1]
            
            ind = ind.astype(np.int64)
            p_start = s_idx[ind]
            # print("p wave start index: ", p_start)

            p_start = p_start - int(0.04 * fields['fs'])
            signal_1ch = signal[0:sig_len, 0]

            heartbeats_raw = list()
            for k in range(len(p_start) - 1):
                remove = (r_idx > p_start[k]) & (r_idx < p_start[k + 1])
                if np.sum(remove) < 2:
                    heartbeats_raw.append(signal_1ch[p_start[k]: p_start[k + 1]])
    
            heartbeats = list()
            for k in range(len(heartbeats_raw)):
                L = math.ceil(len(heartbeats_raw[k]) * new_fs / fields["fs"])
                norm_heartbeat = list(reversed(heartbeats_raw[k])) + list(heartbeats_raw[k]) + list(reversed(heartbeats_raw[k]))
                res = resample_poly(norm_heartbeat, new_fs, fields['fs'])
                res = res[L-1:2*L-1]
                heartbeats.append(res)
            self.qt_db_signlas[file_name] = heartbeats
   

    def plot_heartbeats(self, record_name):
        heartbeats = self.qt_db_signlas[record_name]
        plot_segments(heartbeats, "test")



    def save_pkl(self):
        with open("data/qt_db_heartbeats.pkl", "wb") as output:
                pickle.dump(self.qt_db_signlas, output)
        print("MIT QT DB saved as pickle file!")















