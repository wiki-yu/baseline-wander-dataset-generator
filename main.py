import numpy as np
import _pickle as pickle
from nstdb_noise import Nst_DB_Noise
from qt_db_heartbeats import Qt_DB_Heartbeats
from utils.visualization import plot_segments, generate_table
from train_test_data import create_train_test_data
from datetime import datetime

from digital_filters import FIR_test_Dataset
from digital_filters import IIR_test_Dataset
from eval_metrics import SSD, MAD, PRD, COS_SIM


def main():
    # Generate hearbeat data from QT
    qt_path = 'data/qt-database-1.0.0/'
    heartbeats_generator = Qt_DB_Heartbeats(qt_path)
    heartbeats_generator.create_qt_heartbeats()
    heartbeats_generator.save_pkl()
    record_name = "sel100"
    heartbeats_generator.plot_heartbeats(record_name)

    # Generate noise data from NSTDB
    nstdb_path = 'data/mit-bih-noise-stress-test-database-1.0.0/bw'
    blw_noise_generator = Nst_DB_Noise(nstdb_path)
    blw_noise_generator.create_bwl_noise()

    X_train, y_train, X_test, y_test = create_train_test_data()
    Dataset = [X_train, y_train, X_test, y_test]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print('Dataset ready to use.')

    plot_segments(X_train, "noise data")
    plot_segments(y_train, "ground truth")

    # Classical Filters
    # FIR
    print('Running FIR fiter on the test set. This will take a while (2h)...')
    # start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
    # end_test = datetime.now()
    # train_time_list.append(0)
    # test_time_list.append(end_test - start_test)

    test_results_FIR = [X_test_f, y_test_f, y_filter]

    # Save FIR filter results
    with open('test_results_FIR.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_FIR, output)
    print('Results from experiment FIR filter saved')

    # IIR
    print('Running IIR fiter on the test set. This will take a while (25 mins)...')
    # start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
    # end_test = datetime.now()
    # train_time_list.append(0)
    # test_time_list.append(end_test - start_test)

    test_results_IIR = [X_test_f, y_test_f, y_filter]

    # Save IIR filter results
    with open('test_results_IIR.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_IIR, output)
    print('Results from experiment IIR filter saved')

    # Saving timing list
    # timing = [train_time_list, test_time_list]
    # with open('timing.pkl', 'wb') as output:  # Overwrites any existing file.
    #     pickle.dump(timing, output)
    # print('Timing saved')
    
    # Load Result FIR Filter
    with open('test_results_FIR.pkl', 'rb') as input:
        test_FIR = pickle.load(input)

    # Load Result IIR Filter
    with open('test_results_IIR.pkl', 'rb') as input:
        test_IIR = pickle.load(input)

    # FIR Filtering Metrics
    [X_test, y_test, y_filter] = test_FIR

    SSD_values_FIR = SSD(y_test, y_filter)
    MAD_values_FIR = MAD(y_test, y_filter)
    PRD_values_FIR = PRD(y_test, y_filter)
    COS_SIM_values_FIR = COS_SIM(y_test, y_filter)

    # IIR Filtering Metrics (Best)
    [X_test, y_test, y_filter] = test_IIR

    SSD_values_IIR = SSD(y_test, y_filter)
    MAD_values_IIR = MAD(y_test, y_filter)
    PRD_values_IIR = PRD(y_test, y_filter)
    COS_SIM_values_IIR = COS_SIM(y_test, y_filter)

    ####### LOAD EXPERIMENTS #######

    # # Load timing
    # with open('timing.pkl', 'rb') as input:
    #     timing = pickle.load(input)
    #     [train_time_list, test_time_list] = timing

    # Load Result FIR Filter
    with open('test_results_FIR.pkl', 'rb') as input:
        test_FIR = pickle.load(input)

    # Load Result IIR Filter
    with open('test_results_IIR.pkl', 'rb') as input:
        test_IIR = pickle.load(input)

    ####### Calculate Metrics #######

    print('Calculating metrics ...')

    
    # Digital Filtering

    # FIR Filtering Metrics
    [X_test, y_test, y_filter] = test_FIR
    SSD_values_FIR = SSD(y_test, y_filter)
    MAD_values_FIR = MAD(y_test, y_filter)
    PRD_values_FIR = PRD(y_test, y_filter)
    COS_SIM_values_FIR = COS_SIM(y_test, y_filter)

    # IIR Filtering Metrics (Best)
    [X_test, y_test, y_filter] = test_IIR
    SSD_values_IIR = SSD(y_test, y_filter)
    MAD_values_IIR = MAD(y_test, y_filter)
    PRD_values_IIR = PRD(y_test, y_filter)
    COS_SIM_values_IIR = COS_SIM(y_test, y_filter)

    ####### Results Visualization #######

    SSD_all = [SSD_values_FIR,
            SSD_values_IIR,
            ]

    MAD_all = [MAD_values_FIR,
            MAD_values_IIR,
            ]

    PRD_all = [PRD_values_FIR,
            PRD_values_IIR,
            ]

    CORR_all = [COS_SIM_values_FIR,
                COS_SIM_values_IIR,
                ]

    # Exp_names = ['FIR Filter', 'IIR Filter'] + dl_experiments
    Exp_names = ['FIR Filter', 'IIR Filter']


    metrics = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    metric_values = [SSD_all, MAD_all, PRD_all, CORR_all]

    # Metrics table
    generate_table(metrics, metric_values, Exp_names)

    # # Timing table
    # timing_var = ['training', 'test']
    # generate_table_time(timing_var, timing, Exp_names, gpu=True)


if __name__ == "__main__":
    main()














