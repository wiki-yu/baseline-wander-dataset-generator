import numpy as np
import wfdb
import _pickle as pickle

class Nst_DB_Noise(object):
    def __init__(self, nstdb_path):
        self.nstdb_path = nstdb_path

    def create_bwl_noise(self):
        file_path = self.nstdb_path
        signals, fields = wfdb.rdsamp(file_path)

        for key in fields:
            print(key, fields[key])
        
        np.save("data/blw_noise", signals)
        with open("data/blw_noise.pkl", "wb") as output:
            pickle.dump(signals, output)
        print("MIT-BITH noise stress test database BLW noise save as pickle")
