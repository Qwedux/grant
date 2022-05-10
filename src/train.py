import models
import data_processor

def predict_one_datapoint():
    # example of prediction for one row
    datapoint = [[1000, 0.05, 0.01, 0.1, 0.0]]
    target    = [0.80780E+08]
    dataset_proc = data_processor.Dataset_processor(mode='classifier')
    dataset_proc.load(filename='data_proc_2022_05_10_03_03_55.obj')

    datapoint = dataset_proc.transform_data(data=datapoint)
    target    = dataset_proc.preprocess_targets(targets=target)

    print(datapoint[0], target)

    model = models.MLP_classifier_ensemble(num_models=5)
    model.load(filename='MLP_classifier_ensemble_2022_05_10_03_02_42.model')

    predicted = model.predict(datapoint)

    dataset_proc.compute_metrics(target, predicted=predicted)


if __name__ == "__main__":
    # data preparation
    dataset_proc = data_processor.Dataset_processor(mode='classifier')
    train_data, test_data, train_target, test_target = dataset_proc.preprocess_dataset(input_file="ellipticalmodelgener-gridsearch.dat")
    
    # example of saving / loading of Dataset_processor
    # dataset_proc.save()
    # dataset_proc.load(filename='data_proc_2022_05_10_02_19_13.obj')

    # # example usage of MLP_classifier_ensemble
    model = models.MLP_classifier_ensemble(num_models=5)

    # # it's possible to load pretrained model from file
    # model.load(filename="MLP_ensemble_2022_05_08_23_41_18.model")
    model.fit(train_data=train_data, train_target=train_target)

    # predict on test data and print metrics
    tipnute_prasknute = model.predict(test_data)
    dataset_proc.compute_metrics(test_target, tipnute_prasknute)
    print(model.best_params_)

    # save the model
    model.save()



    