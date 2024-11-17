import os
import sys
import time
import myparams
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# This code runs the good vs bad pair classifier over all the pairs available


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


if __name__ == "__main__":
    In_file = myparams.In_file
    In_label = In_file.split("/")[-1].split(".")[0]
    print(
        f"\n*** Requested to apply the classifier to all the pairs in {In_file}",
    )

    ensure_dir(f"output_ML/{In_label}/")

    # *************
    # (1) Load the preprocessed data
    new_df = pd.read_feather(f"In_data/{myparams.In_file}")

    print(f"\n\t@@@@ Overall we have {len(new_df)} pairs")

    # *************
    # (2) (optionally) remove pairs with a classic energy splitting which is too large
    print(new_df)
    new_df = new_df[new_df[r"$\Delta E$"] < 0.1]
    print(
        f"*-> We decide to keep only the ones with Delta_E<{0.1}, which are {len(new_df)}",
    )

    # *************
    # (3) apply the classifier
    start = time.time()
    classifier_save_path = f"MLmodel/classification-{In_label}"
    # check if the model is there
    if not os.path.isdir(classifier_save_path):
        print(
            f"Error: I am looking for the classifier in {classifier_save_path}, but I can not find it. You probably have to run step1 before this",
        )
        sys.exit()
    else:
        print(f"\nUsing the filter trained in {classifier_save_path}")

    print("\n* Classifier loading", flush=True)
    classifier = TabularPredictor.load(classifier_save_path)

    npairs = len(new_df)
    chunk_size = 1e6
    nchunks = int(npairs / chunk_size) + 1
    processed_chunks = []
    print(f"Temporarily splitting the data ({npairs} pairs) in {nchunks} parts")
    df_chunks = np.array_split(new_df, nchunks)
    del new_df
    print("Classification starting:", flush=True)

    filtered_chunks = []
    filtered_df = pd.DataFrame()
    for chunk_id, chunk in enumerate(df_chunks):
        print(f"\n* Classifying part {chunk_id + 1}/{nchunks}", flush=True)
        df_chunks[chunk_id][class_name] = classifier.predict(chunk.drop(columns=["conf", "i", "j"]))
        # I only keep the predicted class-1
        filtered_df = pd.concat(
            [
                filtered_df,
                chunk[chunk[class_name] > 0],
            ],
        )
        print(
            f"done in {time.time() - start} sec (collected up to {len(filtered_df)} class-1) ",
        )
    timeclass = time.time() - start

    print(
        f"From the {npairs} pairs, only {len(filtered_df)} are in class-1 (in {timeclass} sec = {timeclass / npairs} sec per pair), so {npairs - len(filtered_df)} are class-0.",
    )

    filtered_df_name = f"output_ML/{In_label}/classified_{In_label}.feather"
    filtered_df.reset_index(drop=True).to_feather(filtered_df_name, compression="zstd")
