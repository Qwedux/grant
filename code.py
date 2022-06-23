import pandas as pd
import numpy as np
import pickle
import lzma
import sklearn
import sklearn.neural_network
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics

from collections import namedtuple

Save_container = namedtuple("Save_container", ["info","objects"])
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def save_to(path, info, *objects):
    with lzma.open(path, "wb") as f:
        pickle.dump(Save_container(info, objects), f)

def load_from(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)

def load_dataset_from_file(input_path, mode:str, make_test_set=True):
    loaded_datafile = pd.read_csv(input_path, header = 0, delimiter = " ")
    max_videne_prasknutie = np.max(loaded_datafile.values[:,6])

    if mode == 'classifier':
        # nacitanie trenovacich a testovacich dat
        data = loaded_datafile.values[:,:5]
        targets = loaded_datafile.values[:,6]

        # we consider every one with size at least 0.95 of maximum as successful
        true_prasknute = np.array((targets >= (0.95*max_videne_prasknutie))*1)

        data_scaler = sklearn.preprocessing.MinMaxScaler()
        data = data_scaler.fit_transform(data)

        if make_test_set:
            return {
                "data": sklearn.model_selection.train_test_split(data, true_prasknute, test_size=0.2, random_state=42),
                "scalers": [data_scaler]
            }
        else:
            return {
                "data": (data, true_prasknute),
                "scalers": [data_scaler]
            }
    
    elif mode == 'regressor':
        # regressor predicts numerical values instead of target class
        data = loaded_datafile.values[:,:5]
        targets = loaded_datafile.values[:,6:7]
        
        data_scaler = sklearn.preprocessing.MinMaxScaler()
        data = data_scaler.fit_transform(data)

        targets_scaler = sklearn.preprocessing.MinMaxScaler()
        targets = targets_scaler.fit_transform(targets)[:,0]

        if make_test_set:
            return {
                "data": sklearn.model_selection.train_test_split(data, targets, test_size=0.2, random_state=42),
                "scalers": [data_scaler, targets_scaler]
            }
        else:
            return {
                "data": (data, targets),
                "scalers": [data_scaler, targets_scaler]
            }

    else:
        raise Exception('mozne hodnoty mode su "classifier", "regressor"')

def compute_metrics(mode:str, targets, predicted):
    if mode == 'classifier':
        print("TP: ", np.sum(targets * predicted))
        print("FN: ", np.sum((targets != predicted) * targets))
        print("FP: ", np.sum((targets != predicted) * predicted))
        print("TN: ", np.sum((targets == predicted) * (1-targets)))

        print("precision: ", sklearn.metrics.precision_score(targets, predicted))
        print("recall:    ", sklearn.metrics.recall_score(targets, predicted))
        print("f1:        ", sklearn.metrics.f1_score(targets, predicted))
        print("accuracy:  ", sklearn.metrics.accuracy_score(targets, predicted))
    if mode == 'regressor':
        raise Exception("please binarize the data before computing metrics")

def train_MLP_classifier(dataset:dict):
    '''train MLP classifier for given dataset
    - `dataset["data"]` is a tuple containing train (possibly test) data and targets
    - `dataset["scalers"]` is a list of scalers used for preprocessing raw data
    '''

    # all possible parameters with brief explanation can be found here:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # dolezite parametre su:
    # - alpha = sila regularizacie, cim je vecsia tym mensie detaily sa vedia rozlisit
    # - hidden_layer_sizes = cim su vecsie, tym viacej sa toho vie MLP naucit
    # - solver = pouzival by som 'adam' alebo 'lbfgs', obe davaju dobre vysledky
    model = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=(32, 32, 32, 32),
        activation='tanh',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=600,
        random_state=69,
        tol=0.0001,
        max_fun=15000
    )

    if len(dataset["data"]) == 4:
        # dataset contains test data and targets
        train_data, test_data, train_target, test_target = dataset["data"]
        model.fit(train_data, train_target)

        threshold=0.5
        tipnute_prasknute = (model.predict_proba(test_data)[:, 1] > threshold)*1
        compute_metrics(mode="classifier", targets=test_target, predicted=tipnute_prasknute)

    else:
        # there are no test data and targets in the dataset
        train_data, train_target = dataset["data"]
        model.fit(train_data, train_target)

    save_to("MLP_a_dataset",
        "polozka model obsahuje natrenovany mlp classifier a polozka dataset obsahuje trenovacie,\
        (pripadne testovacie) data, scaler na data, mode datasetu je classifier",
        {"model":model, "dataset":dataset, "dataset_mode":"classifier"}
    )

    return model, dataset

def train_and_save_random_forest(dataset:dict):
    '''train random forest classifier for given dataset
    - `dataset["data"]` is a tuple containing train (possibly test) data and targets
    - `dataset["scalers"]` is a list of scalers used for preprocessing raw data
    '''

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # The most imporatant parameters are n_estimators, max-depth, min_samples_split, min_samples_leaf
    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=69
    )

    if len(dataset["data"]) == 4:
        # dataset contains test data and targets
        train_data, test_data, train_target, test_target = dataset["data"]
        model.fit(train_data, train_target)

        threshold=0.5
        tipnute_prasknute = (model.predict_proba(test_data)[:, 1] > threshold)*1
        compute_metrics(mode="classifier", targets=test_target, predicted=tipnute_prasknute)

    else:
        # there are no test data and targets in the dataset
        train_data, train_target = dataset["data"]
        model.fit(train_data, train_target)

    save_to("Random_Forest_a_dataset",
        "polozka model obsahuje natrenovany random forest classifier a polozka dataset obsahuje trenovacie,\
        (pripadne testovacie) data a scaler na data, mode datasetu je classifier",
        {"model":model, "dataset":dataset, "dataset_mode":"classifier"}
    )

    return model, dataset

def train_and_save_gradient_boosting_classifier(dataset:dict):
    '''train random forest classifier for given dataset
    - `dataset["data"]` is a tuple containing train (possibly test) data and targets
    - `dataset["scalers"]` is a list of scalers used for preprocessing raw data
    '''

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    # The most imporatant parameters are n_estimators, max-depth, min_samples_split, min_samples_leaf
    model = sklearn.ensemble.GradientBoostingClassifier(
        loss='exponential',
        learning_rate=0.1,
        n_estimators=200,
        criterion='friedman_mse',
        min_samples_split=2,
        max_depth=3,
        min_samples_leaf=1,
        random_state=69,
        tol=1e-4
    )

    if len(dataset["data"]) == 4:
        # dataset contains test data and targets
        train_data, test_data, train_target, test_target = dataset["data"]
        model.fit(train_data, train_target)

        threshold=0.5
        tipnute_prasknute = (model.predict_proba(test_data)[:, 1] > threshold)*1
        compute_metrics(mode="classifier", targets=test_target, predicted=tipnute_prasknute)

    else:
        # there are no test data and targets in the dataset
        train_data, train_target = dataset["data"]
        model.fit(train_data, train_target)

    save_to("Gradient_boosting_classifier_a_dataset",
        "polozka model obsahuje natrenovany gradient boosting classifier a polozka dataset obsahuje trenovacie,\
        (pripadne testovacie) data a scaler na data, mode datasetu je classifier",
        {"model":model, "dataset":dataset, "dataset_mode":"classifier"}
    )

    return model, dataset

def train_and_save_MLP_regressor(dataset:dict):
    '''train MLP regressor for given dataset
    - `dataset["data"]` is a tuple containing train (possibly test) data and targets
    - `dataset["scalers"]` is a list of scalers used for preprocessing raw data
    '''

    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    # The most imporatant parameters are hidden_layer_sizes, tol, alpha
    # podobne ako u MLP classifier by som ako solver pouzival 'adam' alebo 'lbfgs',
    # hidden_layer_sizes a alpha funguju rovnako, ale tento krat treba davat pozor aj na parameter
    # tol, kedze treba zarucit konvergenciu na viac desatinnych miest aby to malo zmysel

    # pozn. tento sa trenuje dlhsie ako ostatne, moze to trvat aj niekolko minut. ma vsak tu vyhodu, ze
    # predpoveda ciselne hodnoty, teda sa da pouzit na nejaku dalsiu analizu dat
    model = sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=(64, 64, 64, 64),
        activation='tanh',
        solver='lbfgs',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=10000,
        random_state=69,
        tol=1e-7,
        max_fun=50000
    )

    if len(dataset["data"]) == 4:
        # dataset contains test data and targets
        train_data, test_data, train_target, test_target = dataset["data"]
        model.fit(train_data, train_target)
    
        train_target_unscaled = dataset['scalers'][1].inverse_transform(train_target.reshape(-1,1)).reshape(-1)
        test_target_unscaled = dataset['scalers'][1].inverse_transform(test_target.reshape(-1,1)).reshape(-1)
        # binarizovanie predikcii na zratanie metrik
        max_videne_prasknutie = max(np.max(train_target_unscaled), np.max(test_target_unscaled))
        binarizovane_targets = (test_target_unscaled > 0.95 * max_videne_prasknutie)*1
        tipnute_prasknute = (dataset['scalers'][1].inverse_transform(model.predict(test_data).reshape(-1,1)).reshape(-1) > 0.95 * max_videne_prasknutie)*1
        compute_metrics(mode="classifier", targets=binarizovane_targets, predicted=tipnute_prasknute)

    else:
        # there are no test data and targets in the dataset
        train_data, train_target = dataset["data"]
        model.fit(train_data, train_target)

    save_to("MLP_regressor_a_dataset",
        "polozka model obsahuje natrenovany MLP regressor a polozka dataset obsahuje trenovacie,\
        (pripadne testovacie) data a scaler na data, targets, mode datasetu je regressor",
        {"model":model, "dataset":dataset, "dataset_mode":"regressor"}
    )

    return model, dataset

def threshold_curve(model, dataset, mode:str, save_path='figure'):
    if mode == 'classifier':
        thresholds = []
        TPs = []
        FNs = []
        FPs = []
        _, test_data, _, test_target = dataset["data"]
        for threshold in range(1,100):

            tipnute_prasknute = (model.predict_proba(test_data)[:, 1] > threshold/100)*1
            TPs.append(np.sum(test_target * tipnute_prasknute))
            FNs.append(np.sum((test_target != tipnute_prasknute) * test_target))
            FPs.append(np.sum((test_target != tipnute_prasknute) * tipnute_prasknute))
            thresholds.append(threshold / 100)
        
        import matplotlib.pyplot as plt

        plt.ylim(0, 60)
        plt.plot(thresholds, TPs, 'r')
        plt.plot(thresholds, FNs, 'g')
        plt.plot(thresholds, FPs, 'b')

        # naming the x axis
        plt.xlabel('threshold')
        # naming the y axis
        plt.ylabel('TP(red), FN(green), FP(blue)')

        plt.savefig(save_path)
    
    elif mode == 'regressor':
        thresholds = []
        TPs = []
        FNs = []
        FPs = []
        _, test_data, train_target, test_target = dataset["data"]

        test_predictions = dataset['scalers'][1].inverse_transform(model.predict(test_data).reshape(-1,1)).reshape(-1)
        test_target      = dataset['scalers'][1].inverse_transform(test_target.reshape(-1,1)).reshape(-1)
        train_target     = dataset['scalers'][1].inverse_transform(train_target.reshape(-1,1)).reshape(-1)

        indices_to_sort  = np.argsort(test_target)
        test_predictions = test_predictions[indices_to_sort]
        test_target      = test_target[indices_to_sort]
        
        max_videne_prasknutie = max(np.max(train_target), np.max(test_target))

        for threshold in test_target[-100:]:
            TPs.append(np.sum(( test_target >= 0.95*max_videne_prasknutie) *  (test_predictions >= threshold)))
            FNs.append(np.sum(((test_target >= 0.95*max_videne_prasknutie) != (test_predictions >= threshold)) * (test_target >= 0.95*max_videne_prasknutie)))
            FPs.append(np.sum(((test_target >= 0.95*max_videne_prasknutie) != (test_predictions >= threshold)) * (test_predictions >= threshold)))
            thresholds.append(threshold)

        import matplotlib.pyplot as plt

        plt.ylim(0, 60)
        plt.plot(thresholds, TPs, 'r')
        plt.plot(thresholds, FNs, 'g')
        plt.plot(thresholds, FPs, 'b')

        
        # naming the x axis
        plt.xlabel('threshold')
        # naming the y axis
        plt.ylabel('TP(red), FN(green), FP(blue)')

        # plt.show()
        plt.savefig(save_path)

def classifier_predict_datapoint(model, threshold, datapoint, data_scaler):
    return (model.predict_proba(data_scaler.transform(datapoint))[:, 1] > threshold)*1

def regressor_predict_datapoint(model, datapoint, data_scaler, target_scaler):
    '''vracia velkost prasknutej plochy ako naskalovane cislo, aby z toho bolo nieco rozumne,
    tak sa musi naskalovat spet rovnakym scalerom ako ten ktorim sa skaloval model'''
    return target_scaler.inverse_transform(model.predict(data_scaler.transform(datapoint)).reshape(-1,1)).reshape(-1)

# ---EXAMPLES---
# save_to("test", "testing save of 3 objects", "aba", [1,2,30], 5)
# loaded = load_from("test")
# loaded.info
# loaded.objects # ("aba", [1,2,30], 5)

# dataset_classification = load_dataset_from_file(input_path="ellipticalmodelgener-gridsearch.dat", mode='classifier')
# train_MLP_classifier(dataset_classification)
# train_and_save_random_forest(dataset_classification)
# train_and_save_gradient_boosting_classifier(dataset_classification)

# dataset_regression = load_dataset_from_file(input_path="ellipticalmodelgener-gridsearch.dat", mode='regressor')
# train_and_save_MLP_regressor(dataset_regression)

# tmp = load_from('MLP_regressor_a_dataset').objects[0]
# model, dataset = tmp["model"], tmp["dataset"]
# threshold_curve(model, dataset, mode='regressor', save_path="MLP_regressor_graph")

dataset_classification = load_dataset_from_file(input_path="ellipticalmodelgener-gridsearch.dat", mode='classifier')
tmp = load_from('pretrained_models\MLP_a_dataset').objects[0]
model, dataset = tmp["model"], tmp["dataset"]
print(
    classifier_predict_datapoint(
        model=model,
        threshold=0.5,
        datapoint=dataset['data'][0][729:735],
        data_scaler=dataset['scalers'][0]
    ),
    dataset['data'][2][729:735]
)

dataset_regression = load_dataset_from_file(input_path="ellipticalmodelgener-gridsearch.dat", mode='regressor')
tmp = load_from('pretrained_models\MLP_regressor_a_dataset').objects[0]
model, dataset = tmp["model"], tmp["dataset"]
print(
    regressor_predict_datapoint(
        model=model,
        datapoint=dataset['data'][0][729:735],
        data_scaler=dataset['scalers'][0],
        target_scaler=dataset['scalers'][1]
    ),
    dataset['scalers'][1].inverse_transform(dataset['data'][2][729:735].reshape(-1,1)).reshape(-1),
    sep='\n'
)