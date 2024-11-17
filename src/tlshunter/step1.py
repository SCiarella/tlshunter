import os
import sys
import pandas as pd
import tlshunter as th


def main() -> None:
    """Main function to execute the training process."""
    conf = th.load_config("tests/test_data/test_config.yaml")
    used_data, class_0_pairs, class_1_pairs, pretrain_df, use_new_calculations = th.prepare_data(conf)

    if use_new_calculations:
        all_pairs_df = th.load_data(conf["In_file"])
        if all_pairs_df.empty:
            print("Error: there are no data prepared")
            sys.exit()

        ndecimals = conf["ij_decimals"]
        rounding_error = 10 ** (-1 * (ndecimals + 1)) if ndecimals > 0 else 0

        badpairs_df, goodpairs_df = th.construct_pairs(
            class_0_pairs,
            class_1_pairs,
            all_pairs_df,
            rounding_error,
            ndecimals,
        )
    else:
        print(
            "We are not using data from calculations, but only the pretraining set",
        )
        goodpairs_df = pd.DataFrame()
        badpairs_df = pd.DataFrame()

    new_training_df, training_set, validation_set = th.prepare_training_data(
        conf,
        used_data,
        pretrain_df,
        badpairs_df,
        goodpairs_df,
    )
    th.train_model(training_set, validation_set, conf)

    In_file = conf["In_file"]
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    new_training_df.reset_index(drop=True).to_feather(
        f"{model_path}/data-used-by-classifier-{In_label}.feather",
        compression="zstd",
    )
    training_set.reset_index(drop=True).to_feather(
        f"{model_path}/classifier-training-set-{In_label}.feather",
        compression="zstd",
    )
    validation_set.reset_index(drop=True).to_feather(
        f"{model_path}/classifier-validation-set-{In_label}.feather",
        compression="zstd",
    )


if __name__ == "__main__":
    main()
