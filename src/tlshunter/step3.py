import os
import sys
import multiprocess as mp
import myparams
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor

# This code takes all the available data and retrains the ML model according to the iterative training procedure

if __name__ == "__main__":
    In_label = myparams.In_file.split("/")[-1].split(".")[0]
    class_1_file = f"output_ML/{In_label}/classified_{In_label}.csv"
    print(f"\n*** Requested to train the predictor from {class_1_file}")
    ndecimals = myparams.ij_decimals
    if ndecimals > 0:
        rounding_error = 10 ** (-1 * (ndecimals + 1))
    model_path = f"MLmodel/prediction-{In_label}"

    # *************
    # First I load the data that the predictor has already used for its training
    try:
        used_data = pd.read_feather(f"MLmodel/data-used-by-predictor-{In_label}.feather")
    except:
        print("First time training the predictor")
        used_data = pd.DataFrame()

    # Then I check the calculations to see the what new data are available
    calculation_dir = f"./exact_calculations/{In_label}"
    if not os.path.isfile(f"{calculation_dir}/{myparams.calculations_predictor}"):
        print("\n*(!)* Notice that there are no prediction data\n")
        use_new_calculations = False
        calculated_pairs = pd.DataFrame()
    else:
        use_new_calculations = True
        calculated_pairs = pd.read_csv(f"{calculation_dir}/{myparams.calculations_predictor}", index_col=0)
        print(
            f"From the calculation results we have {len(calculated_pairs)} pairs for which we evaluated the target feature",
        )

    # then load the info about all the pairs
    pairs_df = pd.read_feather(f"In_data/{myparams.In_file}")
    # and format in the correct way
    if ndecimals > 0:
        pairs_df = pairs_df.round(decimals=ndecimals)

    # I also have to include the pre-training data, which I load now to see if overall we gained data
    try:
        pretrain_df = pd.read_feather(f"MLmodel/{myparams.pretraining_predictor}")
    except:
        print("\nNotice that no pretraining is available")
        pretrain_df = pd.DataFrame()

    # ************
    # * Check wether or not you should retrain the model
    if (len(pretrain_df) + len(calculated_pairs)) > len(used_data):
        print(
            f"\n*****\nThe model was trained using {len(used_data)} data and now we could use:\n\t{len(pretrain_df)} from pretraining \n\t{len(calculated_pairs)} from calculations",
        )
    else:
        print("All the data available have been already used to train the model")
        sys.exit()

    # If we are not exited, it means that we have more data, thus it makes sense to retrain the model
    if use_new_calculations:
        # split this task between parallel workers
        elements_per_worker = 100
        chunks = [
            calculated_pairs.iloc[i : i + elements_per_worker]
            for i in range(0, len(calculated_pairs), elements_per_worker)
        ]
        n_chunks = len(chunks)
        print(f"We are going to submit {n_chunks} chunks to get the data\n")

        def process_chunk(chunk):
            worker_df = pd.DataFrame()
            # I search for the given configuration
            for index, row in chunk.iterrows():
                conf = row["conf"]
                i = row["i"]
                j = row["j"]
                target = row["out"]
                if ndecimals > 0:
                    a = pairs_df[
                        (pairs_df["conf"] == conf)
                        & (pairs_df["i"].between(i - rounding_error, i + rounding_error))
                        & (pairs_df["j"].between(j - rounding_error, j + rounding_error))
                    ].copy()
                else:
                    a = pairs_df[(pairs_df["conf"] == conf) & (pairs_df["i"] == i) & (pairs_df["j"] == j)].copy()
                if len(a) > 1:
                    print("Error! multiple correspondences in dw")
                    sys.exit()
                elif len(a) == 1:
                    a["target_feature"] = target
                    worker_df = pd.concat([worker_df, a])
                else:
                    print(f"\nWarning: we do not have {row} in class-1")
            return worker_df

        print("collecting info for pairs")
        # Initialize the pool
        pool = mp.Pool(mp.cpu_count())
        # *** RUN THE PARALLEL FUNCTION
        results = pool.map(process_chunk, [chunk for chunk in chunks])
        pool.close()
        # and add all the new df to the final one
        out_df = pd.DataFrame()
        missed_dw = 0
        for df_chunk in results:
            out_df = pd.concat([out_df, df_chunk])
        print(
            f"\n\nFrom the calculations I constructed a database of {len(out_df)} pairs for which I have the input informations.",
        )
    else:
        print("(We are not using data from calculations, but only pretraining)")
        out_df = pd.DataFrame()

    # *******
    # add the pretrained data (if any)
    if len(pretrain_df) > 0:
        out_df = pd.concat([out_df, pretrain_df])
        out_df = out_df.drop_duplicates()
        out_df = out_df.reset_index(drop=True)
    print(out_df)

    # This is the new training df that will be stored at the end
    new_training_df = out_df
    if len(new_training_df) <= len(used_data):
        print(
            "\n(!) After removing the duplicates it appears that the number of data has not increased since the last time",
        )
        if os.path.isfile(f"{model_path}/predictor.pkl"):
            print("and since the model is already trained, I stop")
            sys.exit()
        else:
            print(f"but the model is not in {model_path}, so I train anyway", flush=True)

    # ************
    # Optional step to normalize small numbers
    # *** I do (-1) log of the data such that the values are closer and their weight is more balanced in the fitness
    new_training_df["10tominus_tg"] = new_training_df["target_feature"].apply(lambda x: -np.log10(x))

    # Split a part of this pairs for validation
    N = len(new_training_df)
    Nval = int(myparams.validation_split * N)
    Ntrain = N - Nval
    # shuffle
    new_training_df = new_training_df.sample(frac=1, random_state=20, ignore_index=True)
    # (!) To be compatible with feather format, I will replace the 'NotAvail' entries with 0
    new_training_df = new_training_df.replace("NotAvail", 0)
    # and slice
    training_set = new_training_df.iloc[:Ntrain]
    validation_set = new_training_df.iloc[Ntrain:N]

    print(
        "\nFrom the overall %d data we prepare:\n\t- training set of %d\n\t- validation set of %d\n\n"
        % (len(new_training_df), len(training_set), len(validation_set)),
        flush=True,
    )

    # **************   TRAIN
    #   Notice that autogluon offer different 'presets' option to maximize precision vs data-usage vs time
    #   if you are not satisfied with the results here, you can try different 'presets' option or build your own
    # check which one to use
    if myparams.Fast_pred == True:
        print("We are training in the fast way")
        presets = "good_quality_faster_inference_only_refit"
    else:
        presets = "high_quality_fast_inference_only_refit"

    # you can also change the training time
    training_hours = myparams.pred_train_hours
    time_limit = training_hours * 60 * 60

    # and define a weight=1/qs in order to increase the precision for the small qs
    training_set["weights"] = (training_set["target_feature"]) ** (-1)

    # train
    # * I am excluding KNN because it is problematic
    # * Convert to float to have optimal performances!
    predictor = TabularPredictor(
        label="10tominus_tg",
        path=model_path,
        eval_metric="mean_squared_error",
        sample_weight="weights",
        weight_evaluation=True,
    ).fit(
        TabularDataset(training_set.drop(columns=["i", "j", "conf", "target_feature"])),
        time_limit=time_limit,
        presets=presets,
        excluded_model_types=["KNN"],
    )

    # store
    new_training_df.reset_index().drop(columns="index").to_feather(
        f"MLmodel/data-used-by-predictor-{In_label}.feather", compression="zstd"
    )
    training_set.reset_index().drop(columns="index").to_feather(
        f"MLmodel/predictor-training-set-{In_label}.feather", compression="zstd"
    )
    validation_set.reset_index().drop(columns="index").to_feather(
        f"MLmodel/predictor-validation-set-{In_label}.feather", compression="zstd"
    )
