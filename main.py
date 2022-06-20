import numpy as np
import _pickle as pickle
from nstdb_noise import Nst_DB_Noise
from qt_db_heartbeats import Qt_DB_Heartbeats

def main():
    # qt_path = 'data/qt-database-1.0.0/'
    # heartbeats_generator = Qt_DB_Heartbeats(qt_path)
    # heartbeats_generator.create_qt_heartbeats()
    # heartbeats_generator.save_pkl()
    # record_name = "sel100"
    # heartbeats_generator.plot_heartbeats(record_name)

    nstdb_path = 'data/mit-bih-noise-stress-test-database-1.0.0/bw'
    blw_noise_generator = Nst_DB_Noise(nstdb_path)
    blw_noise_generator.create_bwl_noise()

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/NoiseBWL.pkl', 'rb') as input:
        nstd = pickle.load(input)

    print("qtdb shape: ", np.shape(qtdb))
    print("nstd shape: ", np.shape(nstd))

    noise_channel1 = nstd[:, 0]
    noise_channel2 = nstd[:, 1]

    noise_test = np.concatenate((noise_channel1[0: int(noise_channel1.shape[0] * 0.13)], noise_channel2[0: int(noise_channel2.shape[0] * 0.13)]))
    noise_train = np.concatenate((noise_channel1[int(noise_channel1.shape[0] * 0.13):-1],noise_channel2[int(noise_channel2.shape[0] * 0.13):-1]))
    print(np.shape(noise_train), np.shape(noise_test))

    beats_train = []
    beats_test = []
    test_set = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 'sel16420','sel16795', 
    'sele0106','sele0121','sel32', 'sel49', 'sel14046','sel15814',]

    qtdb_keys = list(qtdb.keys())
    print("qtdb len: ", len(qtdb_keys))

    skip_beats = 0
    samples = 512
    for i in range(len(qtdb_keys)):
        record_name = qtdb_keys[i]

        for b in qtdb[record_name]:
            b_np = np.zeros[samples]
            b_sq = np.array(b)   

            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue
                
            b_np[init_padding: b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if record_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)






if __name__ == "__main__":
    main()














