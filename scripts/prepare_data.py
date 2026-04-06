from __future__ import annotations
from pathlib import Path

import yaml

from src.data import (
    keep_required_columns,
    load_dreaddit_dataset,
    make_train_calibration_split,
    save_splits,
    summarize_split,
)
from src.utils import load_yaml_config, add_text_length_features


def main() -> None:
    config = load_yaml_config()

    seed = config["seed"]
    calibration_size = config["data"]["calibration_size"]
    output_dir = config["data"]["output_dir"]

    raw_splits = load_dreaddit_dataset(config["data"]["dataset_name"])

    train_df = add_text_length_features(keep_required_columns(raw_splits["train"]))
    validation_df = add_text_length_features(keep_required_columns(raw_splits["validation"]))
    test_df = add_text_length_features(keep_required_columns(raw_splits["test"]))

    proper_train_df, calibration_df = make_train_calibration_split(
        train_df=train_df,
        calibration_size=calibration_size,
        seed=seed,
    )

    processed_splits = {
        "train": proper_train_df,
        "calibration": calibration_df,
        "validation": validation_df,
        "test": test_df,
    }

    save_splits(processed_splits, output_dir=output_dir)

    print ("Processed splits saved to... ", Path(output_dir).resolve())
    for split_name, df in processed_splits.items():
        print (summarize_split(df, split_name))



if __name__ == "__main__":
    main()

