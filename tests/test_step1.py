import os
import pandas as pd
import pytest
import tlshunter as th


@pytest.fixture(scope="session")
def config():
    return th.load_config("tests/test_data/test_config.yaml")


@pytest.fixture(scope="session", autouse=True)
def remove_file(config):
    In_file = config["In_file"]
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    file_path = f"{model_path}/data-used-by-classifier-{In_label}.feather"
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture(scope="session")
def prepared_data(config):
    return th.prepare_data(config)


@pytest.fixture(scope="session")
def prepared_training_data(prepared_data, config):
    used_data, class_0_pairs, class_1_pairs, pretrain_df, use_new_calculations = prepared_data
    badpairs_df = pd.DataFrame()
    goodpairs_df = pd.DataFrame()
    new_training_df, training_set, validation_set = th.prepare_training_data(
        config,
        used_data,
        pretrain_df,
        badpairs_df,
        goodpairs_df,
    )
    return new_training_df, training_set, validation_set


@pytest.fixture(scope="session")
def save_classifier(prepared_training_data, config):
    new_training_df, training_set, validation_set = prepared_training_data
    th.train_classifier(training_set, validation_set, config)
    th.save_datasets(config, new_training_df, training_set, validation_set)
    In_file = config["In_file"]
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    classifier_save_path = f"{model_path}/classification-{In_label}"
    return classifier_save_path


def test_prepare_data(prepared_data):
    used_data, class_0_pairs, class_1_pairs, pretrain_df, use_new_calculations = prepared_data
    assert isinstance(used_data, pd.DataFrame)
    assert isinstance(class_0_pairs, pd.DataFrame)
    assert isinstance(class_1_pairs, pd.DataFrame)
    assert isinstance(pretrain_df, pd.DataFrame)
    assert isinstance(use_new_calculations, bool)


def test_load_data(config):
    all_pairs_df = th.load_data(config["In_file"])
    assert isinstance(all_pairs_df, pd.DataFrame)


def test_construct_pairs(prepared_data, config):
    used_data, class_0_pairs, class_1_pairs, pretrain_df, use_new_calculations = prepared_data
    if use_new_calculations:
        all_pairs_df = th.load_data(config["In_file"])
        ndecimals = config["ij_decimals"]
        rounding_error = 10**(-1 * (ndecimals + 1)) if ndecimals > 0 else 0
        badpairs_df, goodpairs_df = th.construct_pairs(
            class_0_pairs,
            class_1_pairs,
            all_pairs_df,
            rounding_error,
            ndecimals,
        )
        assert isinstance(badpairs_df, pd.DataFrame)
        assert isinstance(goodpairs_df, pd.DataFrame)


def test_prepare_training_data(prepared_training_data):
    new_training_df, training_set, validation_set = prepared_training_data
    assert isinstance(new_training_df, pd.DataFrame)
    assert isinstance(training_set, pd.DataFrame)
    assert isinstance(validation_set, pd.DataFrame)


def test_train_classifier(prepared_training_data, config, save_classifier):
    new_training_df, training_set, validation_set = prepared_training_data
    th.train_classifier(training_set, validation_set, config)
    th.save_datasets(config, new_training_df, training_set, validation_set)
    assert os.path.isdir(save_classifier)
