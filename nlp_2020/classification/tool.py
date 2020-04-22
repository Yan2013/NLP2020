import os
import logging
import jieba

import torch
import torch.nn as nn
from torchtext.data import Field, LabelField, TabularDataset

logger = logging.getLogger(__name__)


def preprocessing(label2id):
    pass


def build_and_cache_dataset(args, mode='train'):

    ID = Field(sequential=False, use_vocab=False)
    CATEGORY = LabelField(sequential=False, use_vocab=True, is_target=True)
    NEWS = Field(
        sequential=True,
        tokenize=jieba.lcut,
        include_lengths=True,
    )

    fields = [
        ('id', ID),
        (None, None),
        ('category', CATEGORY),
        ('news', NEWS),
    ]

    logger.info("Creating features from dataset file at %s", args.data_dir)

    # Since dataset is split by `\t`.
    dataset = TabularDataset(
        os.path.join(args.data_dir, f'{mode}.csv'),
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '\t'},
    )

    features = ((ID, CATEGORY, NEWS), dataset)
    return features


def save_model(args, model, tokenizer, optimizer, scheduler, global_step):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "ckpt-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Take care of distributed/parallel training
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir,
                                                    "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir,
                                                    "scheduler.pt"))
