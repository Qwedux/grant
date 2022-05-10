import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.metrics
import pickle
import lzma
import pathlib
from datetime import datetime

class Dataset_processor:
    def __init__(self, mode:str) -> None:
        self.mode_ = mode
        self.data_scaler_ = None
        self.target_scaler_ = None
        self.max_videne_prasknutie_ = None
    
    def preprocess_dataset(self, input_file):
        loaded_datafile = pd.read_csv(input_file, header = 0, delimiter = " ")
        self.max_videne_prasknutie_ = np.max(loaded_datafile.values[:,6])

        if self.mode_ == 'classifier':
            data = loaded_datafile.values[:,:5]
            targets = loaded_datafile.values[:,6]
            true_prasknute = self.preprocess_targets(targets)

            self.data_scaler_ = sklearn.preprocessing.MinMaxScaler()
            data = self.data_scaler_.fit_transform(data)

            return sklearn.model_selection.train_test_split(data, true_prasknute, test_size=0.2, random_state=42)
        
    def transform_data(self, data) -> np.ndarray:
        return self.data_scaler_.transform(data)
    
    def preprocess_targets(self, targets) -> np.ndarray:
        if self.mode_ == 'classifier':
            return np.array((targets >= (0.95*self.max_videne_prasknutie_))*1)
    
    def unscale_targets_regressor(self, scaled_targets):
        return np.exp(self.target_scaler_.inverse_transform(scaled_targets))

    def save(self, folder:str="data_processors/", filename=None) -> None:
        if filename is not None:
            with lzma.open(pathlib.Path(folder).joinpath(filename), "wb") as processor_file:
                pickle.dump([self.mode_, self.data_scaler_, self.target_scaler_, self.max_videne_prasknutie_], processor_file)
        else:
            with lzma.open(pathlib.Path(folder).joinpath(datetime.now().strftime("data_proc" + '_%Y_%m_%d_%H_%M_%S.obj')), "wb") as processor_file:
                pickle.dump([self.mode_, self.data_scaler_, self.target_scaler_, self.max_videne_prasknutie_], processor_file)

    def load(self, folder:str="data_processors/", filename=None) -> None:
        try:
            with lzma.open(pathlib.Path(folder).joinpath(filename), "rb") as processor_file:
                [self.mode_, self.data_scaler_, self.target_scaler_, self.max_videne_prasknutie_] = pickle.load(processor_file)
        except:
            print("Invalid filename or folder path.")

    def compute_metrics(self, targets, predicted):
        if self.mode_ == 'classifier':
            print("TP: ", np.sum(targets * predicted))
            print("FN: ", np.sum((targets != predicted) * targets))
            print("FP: ", np.sum((targets != predicted) * predicted))
            print("TN: ", np.sum((targets == predicted) * (1-targets)))

            print("precision: ", sklearn.metrics.precision_score(targets, predicted))
            print("recall:    ", sklearn.metrics.recall_score(targets, predicted))
            print("f1:        ", sklearn.metrics.f1_score(targets, predicted))
            print("accuracy:  ", sklearn.metrics.accuracy_score(targets, predicted))