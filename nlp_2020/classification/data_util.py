import os
import csv
import logging
from random import choice, shuffle
from itertools import chain

from typing import List
from collections import defaultdict

from tqdm import tqdm

logger = logging.getLogger(__name__)


# TODO: read data in stream for bigger file.
def split_dataset(
    fpath,
    ratios,
    num_out_categories=5,
    max_train_examples=20000,
    min_train_examples=1000,
    delimiter='_!_',
    sep='\t',
):
    """ Split original dataset to train/dev/test/... dataset
    
    Args:
        `fpath`: path to your data file.
        `ratios` (List[float]): split dataset according to the ratios,
                note that the sum of ratios should smaller than 1.
        `num_out_categories`: how many categories do you want to output.
        `max_train_examples`: ensure the up bound of your ***train dataset***
        `min_train_examples`: ensure the lower bound of your ***train dataset***
    """
    assert num_out_categories >= 1
    assert sum(ratios) <= 1 and len(ratios) > 0

    if len(ratios) < 3:
        ratios += [(1 - sum(ratios)) / (3 - len(ratios))] * (3 - len(ratios))
    elif len(ratios) > 3:
        ratios += [1 - sum(ratios)]
    ratios.sort(reverse=True)

    # `mapping`: category to it's examples
    f_obj = open(fpath, 'r')
    mapping = defaultdict(list)
    # HACK: just for not so big file.
    for line in tqdm(f_obj, desc='Count statistic'):
        # Ignore keyword for that news
        id_, code, category, example, *_ = line.split(delimiter)
        example = convert_full_to_half(example)
        rebuild_row = f'{sep}'.join([id_, code, category, example])
        mapping[category].append(rebuild_row)

    # Shuffle to chose out_categories.
    out_categories = list(
        filter(lambda x: len(mapping[x]) >= min_train_examples,
               mapping.keys()))
    shuffle(out_categories)
    out_categories = out_categories[:num_out_categories]

    # Of course should use `info` rather than `warning`.
    msg = '\n'.join(
        f'category : {category}, num of examples: {len(mapping[category])}'
        for category in out_categories)
    logging.warning('%s', msg)

    min_num_cls_examples = min(
        len(examples) for category, examples in mapping.items()
        if category in out_categories)
    min_num_cls_examples = min(
        min_num_cls_examples,
        int(max_train_examples / max(ratios) / num_out_categories))
    logging.warning('Number of each category for train: %d',
                    min_num_cls_examples)

    end = 0
    dirname = os.path.dirname(fpath)
    for i, ratio in enumerate(tqdm(ratios, desc='File writing')):
        start, end = end, end + int(ratio * min_num_cls_examples)
        with open(os.path.join(dirname, f'{ratio}_{i}.csv'), 'w') as f_obj:
            for category in out_categories:
                for idx in range(start, end):
                    f_obj.write(mapping[category][idx] + '\n')


def convert_full_to_half(ustring):
    """Convert full width char to half width char"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                # Convert whitespace directly
                inside_code = 32
            elif inside_code >= 65281 and inside_code <= 65374:
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)