import numpy as np
import _pickle as pickle

def create_train_test_data():
    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Generate train & test dataset
    # Load QT Database
    with open('data/qt_db_heartbeats.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/blw_noise.pkl', 'rb') as input:
        nstd = pickle.load(input)

    print("qtdb shape: ", np.shape(qtdb))
    print("nstd shape: ", np.shape(nstd))

    noise_channel1 = nstd[:, 0]
    noise_channel2 = nstd[:, 1]

    noise_test = np.concatenate((noise_channel1[0: int(noise_channel1.shape[0] * 0.13)], noise_channel2[0: int(noise_channel2.shape[0] * 0.13)]))
    noise_train = np.concatenate((noise_channel1[int(noise_channel1.shape[0] * 0.13):-1],noise_channel2[int(noise_channel2.shape[0] * 0.13):-1]))
    print(np.shape(noise_train), np.shape(noise_test))

    heartbeats_train = []
    heartbeats_test = []
    test_set = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 'sel16420','sel16795', 
    'sele0106','sele0121','sel32', 'sel49', 'sel14046','sel15814',]

    qtdb_keys = list(qtdb.keys())
    print("qtdb len: ", len(qtdb_keys))

    skip_beats = 0
    samples = 512
    for i in range(len(qtdb_keys)):
        record_name = qtdb_keys[i]
        print(record_name)
        # print(len(qtdb[record_name]))  # heartbeat amount in this record

        for heartbeat in qtdb[record_name]:
            # print(len(heartbeat))

            hb_normalized = np.zeros(samples)
            hb_actual = np.array(heartbeat)   

            init_padding = 16
            if hb_actual.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue
                
            hb_normalized[init_padding: hb_actual.shape[0] + init_padding] = hb_actual - (hb_actual[0] + hb_actual[-1]) / 2

            if record_name in test_set:
                heartbeats_test.append(hb_normalized)
            else:
                heartbeats_train.append(hb_normalized)
        print("skip_beats: ", skip_beats)

    noise_index = 0
    sn_train = []
    sn_test = []
    rnd_train = np.random.randint(low=20, high=200, size=len(heartbeats_train)) / 100
    for i in range(len(heartbeats_train)):
        noise = noise_train[noise_index: noise_index + samples]
        heartbeat_max_value = np.max(heartbeats_train[i]) - np.min(heartbeats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        ase = noise_max_value / heartbeat_max_value
        alpha = rnd_train[i] / ase
        signal_noise = heartbeats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(heartbeats_test)) / 100
    for i in range(len(heartbeats_test)):
        noise = noise_test[noise_index: noise_index + samples]
        heartbeat_max_value = np.max(heartbeats_test[i]) - np.min(heartbeats_test[i])
        ase = noise_max_value /heartbeat_max_value
        alpha = rnd_test[i] / ase
        signal_noise = heartbeats_test[i] + alpha * noise
        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
            noise_index = 0

    X_train = np.array(sn_train)
    y_train = np.array(heartbeats_train)

    X_test = np.array(sn_test)
    y_test = np.array(heartbeats_test)

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    return X_train, y_train, X_train, y_test

