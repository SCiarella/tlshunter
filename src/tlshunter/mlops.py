import os
import pandas as pd
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor


def train_model(training_set: pd.DataFrame, validation_set: pd.DataFrame,
                conf: dict) -> None:
    """Train the model using the training set.

    Parameters
    ----------
    training_set : pd.DataFrame
        DataFrame of training data.
    validation_set : pd.DataFrame
        DataFrame of validation data.
    conf : dict
        Configuration dictionary.
    """
    In_file = conf["In_file"]
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    model_name = f"{model_path}/classification-{In_label}"

    presets = ("good_quality_faster_inference_only_refit" if conf["Fast_class"]
               else "high_quality_fast_inference_only_refit")
    training_hours = conf["class_train_hours"]
    time_limit = training_hours * 60 * 60

    predictor = TabularPredictor(
        label=conf["class_name"], path=model_name, eval_metric="accuracy").fit(
            TabularDataset(training_set.drop(columns=["i", "j", "conf"])),
            time_limit=time_limit,
            presets=presets,
            excluded_model_types=["KNN"],
        )
