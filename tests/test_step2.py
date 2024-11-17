import os
import sys
import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor
import tlshunter as th


@pytest.fixture(scope="session")
def config():
    return th.load_config("tests/test_data/test_config.yaml")


@pytest.fixture()
def prepared_data(config):
    return th.prepare_data(config)


@pytest.fixture(scope="session")
def classifier_path(config, save_classifier):
    return save_classifier


def test_apply_classifier(prepared_data, config, classifier_path):
    In_file = config["In_file"]
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    classifier_save_path = classifier_path

    output_name = f"{In_folder}/output_{In_label}/"
    th.create_directory(output_name)

    in_name = f"{model_path}/data-used-by-classifier-{In_label}.feather"
    new_df = th.load_data(in_name)
    new_df = new_df[new_df["Delta_E"] < 0.1]

    if not os.path.isdir(classifier_save_path):
        print(
            f"Error: I am looking for the classifier in {classifier_save_path}, but I can not find it. You probably have to run step1 before this",
        )
        sys.exit()
    else:
        classifier = TabularPredictor.load(classifier_save_path,
                                           require_version_match=False)

    filtered_df = th.classifier_filter(new_df, classifier, config)
    filtered_df_name = f"{output_name}/classified_{In_label}.feather"
    filtered_df.reset_index(drop=True).to_feather(filtered_df_name,
                                                  compression="zstd")

    assert isinstance(filtered_df, pd.DataFrame)
    assert os.path.exists(filtered_df_name)
