import os
import sys
import myparams
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor

# This code applied the predictor to all the pairs in class-1, and ranks them accordingly

In_label = myparams.In_file.split("/")[-1].split(".")[0]
print(f"\n*** Requested to apply the predictor to {In_label}")
model_path = f"MLmodel/prediction-{In_label}"

# check if the model is there
if not os.path.isdir(model_path):
    print(
        f"Error: I am looking for the ML model in {model_path}, but I can not find it. You probably have to run step3.py before this",
    )
    sys.exit()

# Load the model
predictor = TabularPredictor.load(model_path)
predictor.persist_models()

# Read the data
outdir = f"output_ML/{In_label}"
pairs_df = pd.read_feather(
    f"output_ML/{In_label}/classified_{In_label}.feather")
nconfs = len(pairs_df["conf"].drop_duplicates())
print(
    f"\n*** Found {nconfs} macro-configurations and a total of {len(pairs_df)} pairs\n\nStarting predictions",
)

# predict the qs
X = pairs_df.drop(columns=["i", "j", "conf", class_name])
X = TabularDataset(X)
y_pred_by_AI = predictor.predict(pairs_df)
y_pred_by_AI = np.power(10, -y_pred_by_AI)
print("The target has been predicted. Now storing results")

# store the predictions
pairs_df["target_feature"] = y_pred_by_AI
pairs_df = pairs_df.sort_values(by="target_feature")
pairs_df[[
    "conf",
    "i",
    "j",
    "target_feature",
]].to_csv(f"{outdir}/predicted_{In_label}_allpairs.csv", index=False)
all_qs_df = pairs_df.copy()

# then I exclude the pairs that are bad (according to the exact calculation)
calculation_dir = f"./exact_calculations/{In_label}"
if not os.path.isfile(f"{calculation_dir}/{myparams.calculations_classifier}"):
    print("\n*(!)* Notice that there are no classification data\n")
else:
    class_0_pairs = pd.read_csv(
        f"{calculation_dir}/{myparams.calculations_classifier}", index_col=0)
    class_1_pairs = pd.read_csv(
        f"{calculation_dir}/{myparams.calculations_predictor}",
        index_col=0)[["conf", "i", "j"]]

    temp_df = all_qs_df.reset_index(drop=True)
    temp_df["index"] = temp_df.index
    remove_df = temp_df.merge(class_0_pairs, how="inner", indicator=False)
    remove_df = remove_df.set_index("index")
    all_qs_df = all_qs_df.drop(remove_df.index).reset_index()
    print(
        f"\n*We know that {len(remove_df)} of the new pairs are class-0 (from calculations), so we do not need to predict them.\nWe then end up with {len(all_qs_df)} new pairs",
    )

# then exclude the pairs for which I already calculated the exact target property
if not os.path.isfile(f"{calculation_dir}/{myparams.calculations_predictor}"):
    print("\n*(!)* Notice that there are no prediction data\n")
else:
    calculated_pairs = pd.read_csv(
        f"{calculation_dir}/{myparams.calculations_predictor}", index_col=0)

    temp_df = all_qs_df.reset_index(drop=True)
    temp_df["index"] = temp_df.index
    remove_df = temp_df.merge(calculated_pairs, how="inner", indicator=False)
    remove_df = remove_df.set_index("index")
    all_qs_df = all_qs_df.drop(remove_df.index)
    all_qs_df = all_qs_df.reset_index(drop=True)
    print(
        f"\n*For {len(remove_df)} of the new pairs we already run the calculations, so we do not need to predict them.\nWe then finish with {len(all_qs_df)} new pairs",
    )

# Storing
all_qs_df[[
    "conf",
    "i",
    "j",
    "target_feature",
]].to_csv(f"{outdir}/predicted_{In_label}_newpairs.csv", index=False)
