import os
import sys
from autogluon.tabular import TabularPredictor
import tlshunter as th

if __name__ == "__main__":
    conf = th.load_config("tests/test_data/test_config.yaml")
    In_file = conf["In_file"]
    In_label = os.path.splitext(os.path.basename(In_file))[0]
    In_folder = os.path.dirname(In_file)
    model_path = f"{In_folder}/MLmodels"
    classifier_save_path = f"{model_path}/classification-{In_label}"
    in_name = f"{model_path}/data-used-by-classifier-{In_label}.feather"

    print(
        f"\n*** Requested to apply the classifier to all the pairs in {in_name}",
    )

    output_name = f"{In_folder}/output_{In_label}/"
    th.create_directory(output_name)

    # *************
    # (1) Load the preprocessed data
    #new_df = th.load_data(In_file)
    new_df = th.load_data(in_name)

    print(f"\n\t@@@@ Overall we have {len(new_df)} pairs")

    # *************
    # (2) (optionally) remove pairs with a classic energy splitting which is too large
    #    new_df = new_df[new_df[r"$\Delta E$"] < 0.1]
    new_df = new_df[new_df["Delta_E"] < 0.1]
    print(
        f"*-> We decide to keep only the ones with Delta_E<{0.1}, which are {len(new_df)}",
    )

    # *************
    # (3) apply the classifier
    if not os.path.isdir(classifier_save_path):
        print(
            f"Error: I am looking for the classifier in {classifier_save_path}, but I can not find it. You probably have to run step1 before this",
        )
        sys.exit()
    else:
        print(f"\nUsing the filter trained in {classifier_save_path}")

    print("\n* Classifier loading", flush=True)
    classifier = TabularPredictor.load(classifier_save_path,
                                       require_version_match=False)

    print(new_df.columns)
    filtered_df = th.classifier_filter(new_df, classifier, conf)

    filtered_df_name = f"{output_name}/classified_{In_label}.feather"
    filtered_df.reset_index(drop=True).to_feather(filtered_df_name,
                                                  compression="zstd")
