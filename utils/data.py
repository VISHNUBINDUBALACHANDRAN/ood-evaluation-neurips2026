import os 
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict
from datasets import load_dataset, dataset_dict, arrow_dataset
from transformers import LayoutLMv3Processor
from datasets import Features, Sequence, Value, Array2D, Array3D
import datasets
import argparse
from torchvision import transforms
from multiprocessing import Pool, cpu_count
import functools
from PIL import Image
import json
import torch
from torch.utils.data import DataLoader



FEATURES_PROCESSOR = {
    "v2": Features(
        {
            "image": Array3D(dtype="int64", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
        }
    ),
    "v3": Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "pixel_values": Array3D(dtype="float64", shape=(3, 224, 224)),
        }
    ),
}


def parse_data_paths(path_input):

    """
    Parses directories to gather all image file paths.

    Arguments:
        path_input: (str) : The root directory path containing subdirectories 
                            with image files.

    Returns:
        all_paths_imagess: (list) : A list of file paths for all `.png` images 
                                    found in the specified directory and its 
                                    subdirectories.
    """
    
    all_paths_dirs = [os.path.join(path_input,i) for i in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, i))]
    all_paths_imagess = []

    for i in all_paths_dirs:
        all_paths_imagess += [os.path.join(i,ii) for ii in os.listdir(i) if os.path.isfile(os.path.join(i, ii))]
        
    all_paths_imagess = [i for i in all_paths_imagess if i.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'))]
    
    return all_paths_imagess 



def paralelize_df(data, col_func, sub_func, col, part, cor, **kwargs):

    """
    Performs multi processing on data

    Arguments:
        data    :  (pd.DataFrame)  : datset to process
        col_func:  (function)      : function to operate on column of dataframe
        col     :  (str)           : column name
        part    :  (int)           : number od parts to split data
        cor     :  (int)           : number od cores to use
        kwargs  :  (dict)          : additional keyword arguments required for functions
    Returns:
        data    :  (pd.DataFrame)  : processed
    """

    # Split DataFrame into chunks using pandas-native approach (numpy splits can
    # convert DataFrames to arrays in newer numpy versions)
    chunk_size = max(1, len(data) // part)
    data_split = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    pool = Pool(cor)

    data = pd.concat(
        pool.map(
            functools.partial(col_func, sub_func=sub_func, col=col, **kwargs),
            data_split,
        )
    )

    pool.close()
    pool.join()

    return data


def col_func(data, sub_func, col, **kwargs):

    """
    Applies a function on a the column col of the dataframe

    Arguments:
        data    :  (pd.DataFrame)  : datset to process
        sub_func:  (function)      : function to operate on column of dataframe
        col     :  (str)           : column name
        kwargs  :  (dict)          : additional keyword arguments required for functions
    Returns:
        data    :  (pd.DataFrame)  : processed
    """

    data[col] = data[col].apply(functools.partial(sub_func, **kwargs))

    return data



def apply_tesser(x, **kwargs):

    """
    processing function

    Arguments:
        x       :  (any)     : item to apply processing on
        kwargs  :  (dict)    : additional keyword arguments required for functions
    Returns:
        encoded_inputs    :  processed output
    """

    image = Image.open(x).convert("RGB")
    
    

    dl_processor = kwargs.get("processor")
    transformation = kwargs.get("transformation")
    
    image = transformation(image)

    encoded_inputs = dl_processor(
        image, padding="max_length", truncation=True, return_token_type_ids=True
    )

    return encoded_inputs


def write_df_file(df, path) -> None:

    """
    Writes a DataFrame to a CSV file in a specified directory.

    Arguments:
        df  : (pd.DataFrame) : The DataFrame to be saved as a CSV file.
        path: (str)          : The path to the output directory where the 
                               temporary directory will be created and the CSV 
                               file will be saved.

    Returns:
        None
    """

    path_temp_to_save = os.path.join(path, "temp")

    if not os.path.exists(path_temp_to_save):
        os.makedirs(path_temp_to_save)

    path_temp_to_file = os.path.join(path_temp_to_save, "train.csv")
    df.to_csv(path_temp_to_file, index=False)


def fdir(path,specific_end='.zip',specific_remove=None,specific_include=None):


    """
    Filters and retrieves file paths in a directory based on specific criteria.


    Arguments:
        path            : (str)  : The directory path to search for files.
        specific_end    : (str)  : Optional. The file extension that files must 
                                   end with to be included in the output list. 
                                   Defaults to '.zip'.
        specific_remove : (str)  : Optional. A substring that, if found in a 
                                   filename, will exclude that file from the 
                                   output list.
        specific_include: (str)  : Optional. A substring that must be present 
                                   in a filename for it to be included in the 
                                   output list.

    Returns:
        out: (list) : A list of file paths that meet the specified criteria.
    """
    
    out = [os.path.join(path,i) for i in os.listdir(path)]

    if specific_end is not None:
        out = [i for i in out if i.endswith(specific_end)]
        
    if specific_remove is not None:
        out = [i for i in out if specific_remove not in i]
        
    if specific_include is not None:
        out = [i for i in out if specific_include in i]
    
    return out


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_serialize(dicto):

    """
    Recursively serializes a dictionary to ensure all values are JSON-compatible.


    Arguments:
        dicto: (dict) : The input dictionary that may contain nested structures 
                        (e.g., lists, NumPy arrays, or other dictionaries).

    Returns:
        new_dict: (dict) : A new dictionary with all values converted to 
                           JSON-serializable formats.
    """

    new_dict = {}

    for i,j in dicto.items():
        # print(type(i),type(j))
        if isinstance(j,dict):
            new_dict[i] = json_serialize(j)

        elif isinstance(j, list) or isinstance(j, np.ndarray):

            new_dict[i] = [int(o) for o in j]

        else:
            if isinstance(j, int) or isinstance(j, float):
                new_dict[i] = int(j)

            else:
                new_dict[i] = j

    return new_dict
    


def write_json(file_name_path,data):
    with open(file_name_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============================================================
# Centralized Data Utilities (shared across run scripts)
# ============================================================

def create_dataloader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True):
    """
    Create a DataLoader with custom collation for LayoutLMv3.
    
    Handles input_ids, attention_mask, bbox, pixel_values,
    token_type_ids, and labels fields.
    
    Args:
        dataset: HuggingFace dataset or compatible
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    def collate_fn(features):
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["bbox"] = torch.stack([f["bbox"] for f in features])
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])
        
        if "labels" in features[0]:
            batch["labels"] = torch.tensor(
                [f["labels"] for f in features], dtype=torch.long
            )
        
        return batch
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def encode_labels(dataset, label2id):
    """
    Convert string labels to integer IDs in a HuggingFace dataset.
    
    Args:
        dataset: HuggingFace dataset with 'labels' column
        label2id: Dict mapping label strings to integer IDs
        
    Returns:
        Dataset with integer labels
    """
    def _encode(example):
        label = example["labels"]
        if isinstance(label, str):
            example["labels"] = label2id[label]
        return example
    return dataset.map(_encode)


def filter_dataset_by_classes(dataset, class_list, label_column="labels"):
    """
    Keep only samples belonging to specified classes.
    
    Also handles alias matching for classes that may have alternate names
    (e.g., 'ADVE' vs 'Advertisement' in Tobacco3482).
    
    Args:
        dataset: HuggingFace dataset
        class_list: List of class names to keep
        label_column: Name of the label column
        
    Returns:
        Filtered dataset
    """
    def _filter(example):
        label = example[label_column]
        if isinstance(label, str):
            return label in class_list
        return True
    return dataset.filter(_filter)


def data_collator(features):
    """
    Custom data collator for LayoutLMv3 features (for HuggingFace Trainer).
    
    Args:
        features: List of dicts with input_ids, attention_mask, bbox, etc.
        
    Returns:
        Batched dict of tensors
    """
    batch = {}
    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    batch["bbox"] = torch.stack([f["bbox"] for f in features])
    batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    
    if "token_type_ids" in features[0]:
        batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])
    
    if "labels" in features[0]:
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    
    return batch