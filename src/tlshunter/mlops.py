import os
import time
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor


def train_classifier(training_set: pd.DataFrame, validation_set: pd.DataFrame,
                     conf: dict) -> None:
    """Train the model using the training set.

    This function trains a classification model using the provided training set and configuration.
    The model is saved to a specified path based on the input file name.

    Parameters
    ----------
    training_set : pd.DataFrame
        DataFrame of training data.
    validation_set : pd.DataFrame
        DataFrame of validation data.
    conf : dict
        Configuration dictionary containing parameters such as 'In_file', 'Fast_class', 'class_train_hours', and 'class_name'.
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


def classifier_filter(new_df: pd.DataFrame, classifier: TabularPredictor,
                      conf: dict) -> pd.DataFrame:
    """Filter the DataFrame using the classifier.

    This function classifies the rows of the DataFrame and filters out the rows that do not belong to class-1.
    It processes the DataFrame in chunks to handle large datasets efficiently.

    Parameters
    ----------
    new_df : pd.DataFrame
        DataFrame containing the data to be classified.
    classifier : TabularPredictor
        The trained classifier used to predict the class labels.
    conf : dict
        Configuration dictionary containing parameters such as 'class_name'.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing only the rows classified as class-1.
    """
    start = time.time()
    npairs = len(new_df)
    chunk_size = 1e6
    nchunks = int(npairs / chunk_size) + 1
    df_chunks = np.array_split(new_df, nchunks)
    del new_df
    print("Classification starting:", flush=True)
    filtered_df = pd.DataFrame()
    for chunk_id, chunk in enumerate(df_chunks):
        print(f"\n* Classifying part {chunk_id + 1}/{nchunks}", flush=True)
        df_chunks[chunk_id][conf["class_name"]] = classifier.predict(
            chunk.drop(columns=["conf", "i", "j"]))
        filtered_df = pd.concat(
            [filtered_df, chunk[chunk[conf["class_name"]] > 0]])
        print(
            f"done in {time.time() - start} sec (collected up to {len(filtered_df)} class-1) ",
        )
    timeclass = time.time() - start
    print(
        f"From the {npairs} pairs, only {len(filtered_df)} are in class-1 (in {timeclass} sec = {timeclass / npairs} sec per pair), so {npairs - len(filtered_df)} are class-0.",
    )
    return filtered_df
