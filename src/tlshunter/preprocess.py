import os
import sys
from typing import Tuple
import multiprocess as mp
import pandas as pd
from .io import load_data
from .utils import create_directory


def process_chunk(
    chunk: pd.DataFrame, all_pairs_df: pd.DataFrame, rounding_error: float, ndecimals: int
) -> pd.DataFrame:
    """Process a chunk of data to find corresponding input pairs.

    Parameters
    ----------
    chunk : pd.DataFrame
        Chunk of data to process.
    all_pairs_df : pd.DataFrame
        DataFrame containing all pairs.
    rounding_error : float
        Rounding error for matching.
    ndecimals : int
        Number of decimals to round to.

    Returns:
    -------
    pd.DataFrame
        Processed data.
    """
    worker_df = pd.DataFrame()
    for index, row in chunk.iterrows():
        conf = row["conf"]
        i = row["i"]
        j = row["j"]
        if ndecimals > 0:
            a = all_pairs_df[
                (all_pairs_df["conf"] == conf)
                & (all_pairs_df["i"].between(i - rounding_error, i + rounding_error))
                & (all_pairs_df["j"].between(j - rounding_error, j + rounding_error))
            ]
        else:
            a = all_pairs_df[(all_pairs_df["conf"] == conf) & (all_pairs_df["i"] == i) & (all_pairs_df["j"] == j)]
        if len(a) > 1:
            print(f"Error! multiple correspondences for {row}")
            sys.exit()
        elif len(a) == 1:
            worker_df = pd.concat([worker_df, a])
    return worker_df


def prepare_data(
    conf: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Prepare data for training.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]
        DataFrames for used data, class 0 pairs, class 1 pairs, pretraining data, and a flag indicating if new calculations are used.
    """
    In_file = conf["In_file"]
    In_folder = os.path.dirname(In_file)
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    print(f"\n*** Requested to train the classifier from {In_file}")

    ndecimals = conf["ij_decimals"]
    rounding_error = 10 ** (-1 * (ndecimals + 1)) if ndecimals > 0 else 0
    model_path = f"{In_folder}/MLmodels"
    calc_dirname = f"{In_folder}/exact_calculations/{In_label}"

    create_directory(calc_dirname)
    create_directory(model_path)

    used_data = load_data(f"{model_path}/data-used-by-classifier-{In_label}.feather")
    if used_data.empty:
        print("First time training the classifier")

    if not os.path.isfile(f'{calc_dirname}/{conf["calculations_classifier"]}'):
        print("\n*(!)* Notice that there are no classification data\n")
        use_new_calculations = False
        class_0_pairs = pd.DataFrame()
        class_1_pairs = pd.DataFrame()
    else:
        use_new_calculations = True
        class_0_pairs = load_data(f"{calc_dir}/{conf['calculations_classifier']}", "csv")
        class_1_pairs = load_data(f"{calc_dir}/{conf['calculations_predictor']}", "csv")[["conf", "i", "j"]]
        print(
            f"From the calculation results we have {len(class_0_pairs)} class-0 and {len(class_1_pairs)} class-1",
        )

    pretrain_df = load_data(f'{model_path}/{conf["pretraining_classifier"]}')
    if pretrain_df.empty:
        print("Notice that no pretraining is available")

    if (len(pretrain_df) + len(class_0_pairs) + len(class_1_pairs)) > len(used_data):
        print(
            f"\n*****\nThe model was trained using {len(used_data)} data and now we could use:\n\t{len(pretrain_df)} from pretraining (both classes)\n\t{len(class_0_pairs)} calculated class-0\n\t{len(class_1_pairs)} calculated class-1",
        )
    else:
        print("All the data available have been already used to train the model")
        sys.exit()

    return used_data, class_0_pairs, class_1_pairs, pretrain_df, use_new_calculations


def construct_pairs(
    class_0_pairs: pd.DataFrame,
    class_1_pairs: pd.DataFrame,
    all_pairs_df: pd.DataFrame,
    rounding_error: float,
    ndecimals: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct pairs of data for training.

    Parameters
    ----------
    class_0_pairs : pd.DataFrame
        DataFrame of class 0 pairs.
    class_1_pairs : pd.DataFrame
        DataFrame of class 1 pairs.
    all_pairs_df : pd.DataFrame
        DataFrame containing all pairs.
    rounding_error : float
        Rounding error for matching.
    ndecimals : int
        Number of decimals to round to.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames of bad pairs and good pairs.
    """
    elements_per_worker = 20
    chunks = [
        class_0_pairs.iloc[i : i + elements_per_worker] for i in range(0, len(class_0_pairs), elements_per_worker)
    ]
    print(f"\nWe are going to submit {len(chunks)} chunks for the bad pairs")

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_chunk, [(chunk, all_pairs_df, rounding_error, ndecimals) for chunk in chunks])

    badpairs_df = pd.concat(results)
    badpairs_df[class_name] = 0
    print(
        f"Constructed the database of {len(badpairs_df)} bad pairs, from the new data",
    )

    chunks = [
        class_1_pairs.iloc[i : i + elements_per_worker] for i in range(0, len(class_1_pairs), elements_per_worker)
    ]
    print(f"\nWe are going to submit {len(chunks)} chunks for the good data")

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_chunk, [(chunk, all_pairs_df, rounding_error, ndecimals) for chunk in chunks])

    goodpairs_df = pd.concat(results)
    goodpairs_df[class_name] = 1
    print(
        f"Constructed the database of {len(goodpairs_df)} good pairs, from the new data",
    )

    return badpairs_df, goodpairs_df


def prepare_training_data(
    conf: dict,
    used_data: pd.DataFrame,
    pretrain_df: pd.DataFrame,
    badpairs_df: pd.DataFrame,
    goodpairs_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare training and validation data.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    used_data : pd.DataFrame
        DataFrame of used data.
    pretrain_df : pd.DataFrame
        DataFrame of pretraining data.
    badpairs_df : pd.DataFrame
        DataFrame of bad pairs.
    goodpairs_df : pd.DataFrame
        DataFrame of good pairs.

    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames of new training data, training set, and validation set.
    """
    In_file = conf["In_file"]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    class_name = conf["class_name"]

    if not pretrain_df.empty:
        if not goodpairs_df.empty:
            goodpairs_df = pd.concat([goodpairs_df, pretrain_df[pretrain_df[class_name] > 0]]).drop_duplicates()
        else:
            goodpairs_df = pretrain_df[pretrain_df[class_name] > 0]
        if not badpairs_df.empty:
            badpairs_df = pd.concat([badpairs_df, pretrain_df[pretrain_df[class_name] < 1]]).drop_duplicates
        else:
            badpairs_df = pretrain_df[pretrain_df[class_name] < 1]
    else:
        print("No pretraining data available")
        if goodpairs_df.empty or badpairs_df.empty:
            print("and no new data for training [ERROR]")
            sys.exit()

    qs_df = pd.concat([goodpairs_df, badpairs_df])
    qs_df[class_name] = qs_df[class_name].astype(bool)
    goodpairs_df[class_name] = goodpairs_df[class_name].astype(bool)
    badpairs_df[class_name] = badpairs_df[class_name].astype(bool)

    new_training_df = qs_df.copy()
    if len(new_training_df) <= len(used_data):
        print(
            "After removing the duplicates it appears that the number of data has not increased since the last time",
        )
        if os.path.isfile(f"{model_path}/predictor.pkl"):
            print("and since the model is already trained, I stop")
            sys.exit()
        else:
            print(f"but the model is not in {model_path}, so I train anyway")

    N = min(len(goodpairs_df), len(badpairs_df))
    print(
        f"Having {len(goodpairs_df)} good and {len(badpairs_df)} bad pairs, we select only {N} of each, to balance the classifier",
    )

    Nval = int(conf["validation_split"] * N)
    Ntrain = N - Nval

    goodpairs_df = goodpairs_df.sample(frac=1, random_state=20, ignore_index=True)
    badpairs_df = badpairs_df.sample(frac=1, random_state=20, ignore_index=True)

    training_set = pd.concat([goodpairs_df.iloc[:Ntrain], badpairs_df.iloc[:Ntrain]]).sample(
        frac=1, random_state=20, ignore_index=True
    )
    validation_set = pd.concat([goodpairs_df.iloc[Ntrain:N], badpairs_df.iloc[Ntrain:N]]).sample(
        frac=1, random_state=20, ignore_index=True
    )

    print(
        f"From the overall {len(new_training_df)} data we prepare:\n\t- training set of {len(training_set)}  (half good and half bad) \n\t- validation set of {len(validation_set)}  (half good and half bad)",
    )

    return new_training_df, training_set, validation_set
