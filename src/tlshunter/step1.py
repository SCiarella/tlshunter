import sys
import pandas as pd
import tlshunter as th

if __name__ == "__main__":
    conf = th.load_config("tests/test_data/test_config.yaml")
    used_data, class_0_pairs, class_1_pairs, pretrain_df, use_new_calculations = th.prepare_data(
        conf)

    if use_new_calculations:
        all_pairs_df = th.load_data(conf["In_file"])
        if all_pairs_df.empty:
            print("Error: there are no data prepared")
            sys.exit()

        ndecimals = conf["ij_decimals"]
        rounding_error = 10**(-1 * (ndecimals + 1)) if ndecimals > 0 else 0

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
    th.train_classifier(training_set, validation_set, conf)

    th.save_datasets(conf, new_training_df, training_set, validation_set)
