import os
import datasets
from parse import parse
from typing import Union

"""
Data tools: Utilities related to handling data
"""


def dyck2dataset(
    files: Union[str, list], use_splits: bool = False
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    """
    Returns a dataset constructed from the data contained in files of the format used in this repo:
    https://github.com/princeton-nlp/dyck-transformer/tree/master/data
    Example file name: k128_m10_tr2000000.train
    :param files: A directory, file or list of files
    :param use_splits: If true, returns an instance of DatasetDict with splits for every file extension;
    If false, returnts an instance of Dataset
    :return: A datasets.Dataset or a datasets.DatasetDict with splits for every file extension

    Example:
        ```python
        >>> from training_dynamics.utils.data_tools import dyck2dataset
        >>> data_dir = "dyck-transformer/data"
        >>> dataset = dyck2dataset(data_dir)
        >>> dataset_splits = dyck2dataset(data_dir, use_splits = True)
        ```
    """

    if isinstance(files, str):
        if os.path.isdir(files):
            files = [
                os.path.join(files, f)
                for f in os.listdir(files)
                if os.path.isfile(os.path.join(files, f))
            ]
        else:
            files = [files]

    dataset_dict = {}
    col_names = ["k", "m", "tr"]
    if not use_splits:
        col_names = col_names + ["subset"]
    for file_path in files:
        file_name = os.path.basename(file_path)
        parsed = parse("k{k:d}_m{m:d}_tr{tr}.{subset}", file_name)
        current_split = dataset_dict
        subset = parsed["subset"]
        if use_splits:
            if subset not in dataset_dict.keys():
                dataset_dict[subset] = {}
            current_split = dataset_dict[subset]

        for key in col_names + ["text"]:
            if key not in current_split.keys():
                current_split[key] = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            # Remove "END" string and line break
            lines = [
                line.replace("END", "").replace("\n", "").strip() for line in lines
            ]
            current_split["text"].extend(lines)
            for key in col_names:
                current_split[key].extend([parsed[key]] * len(lines))

    if use_splits:
        return datasets.DatasetDict(
            {
                split: datasets.Dataset.from_dict(dataset_dict[split])
                for split in dataset_dict.keys()
            }
        )
    else:
        return datasets.Dataset.from_dict(dataset_dict)


def is_balanced(txt, symbols={")": "(", "]": "[", "}": "{"}):
    """
    Checks if a string contains only well balanced opening and closing symbols
    (e.g. brackets, parentheses)
    :param txt: A string to test for non balanced symbols
    :param symbols: A dictionary of opening and closing symbols
    :return: True if symbols are well balance, False otherwise
    Example:
    ```python
    >>> import os
    >>> from training_dynamics.utils.data_tools import dyck2dataset, is_balanced
    >>> data_dir = "dyck-transformer/data"
    >>> file_path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)][0]
    >>> dataset = dyck2dataset(file_path)
    >>> all([is_balanced(example) for example in dataset["text"]])
    >>> is_balanced("({abc}) 123 []")
    >>> is_balanced("({abc}) 123 [] (}")
    ```
    """
    stack = []

    char_list = list(txt)
    for ch in char_list:
        if ch in symbols.values():
            stack.append(ch)
        if ch in symbols.keys():
            opening_symbol = ""
            if len(stack) > 0:
                opening_symbol = stack.pop()
            if opening_symbol != symbols[ch]:
                return False

    return len(stack) == 0
