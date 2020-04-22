import os
import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support

from nlp_2020.classification.args import get_args
from nlp_2020.classification.model import TextClassifier
from nlp_2020.classification.tool import build_and_cache_dataset, save_model

logger = logging.getLogger(__name__)


def train(args, writer):

    # Build train dataset
    fields, train_dataset = build_and_cache_dataset(args, mode='train')

    # Build vocab
    ID, CATEGORY, NEWS = fields
    vectors = Vectors(name=args.embed_path, cache=args.data_dir)
    # NOTE: use train_dataset to build vocab!
    NEWS.build_vocab(
        train_dataset,
        max_size=args.vocab_size,
        vectors=vectors,
        unk_init=torch.nn.init.xavier_normal_,
    )
    CATEGORY.build_vocab(train_dataset)

    model = TextClassifier(
        vocab_size=len(NEWS.vocab),
        output_dim=args.num_labels,
        pad_idx=NEWS.vocab.stoi[NEWS.pad_token],
        dropout=args.dropout,
    )
    # Init embeddings for model
    model.embedding.from_pretrained(NEWS.vocab.vectors)

    bucket_iterator = BucketIterator(
        train_dataset,
        batch_size=args.train_batch_size,
        sort_within_batch=True,
        shuffle=True,
        sort_key=lambda x: len(x.news),
        device=args.device,
    )

    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     eps=args.adam_epsilon)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.learning_rate * 10,
                           epochs=args.num_train_epochs,
                           steps_per_epoch=len(bucket_iterator))

    global_step = 0
    model.zero_grad()
    train_trange = trange(0, args.num_train_epochs, desc="Train epoch")
    for _ in train_trange:
        epoch_iterator = tqdm(bucket_iterator, desc='Training')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            news, news_lengths = batch.news
            category = batch.category
            preds = model(news, news_lengths)

            loss = criterion(preds, category)
            loss.backward()
            # NOTE: optimizer should update before scheduler

            # Logging
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/lr',
                              scheduler.get_last_lr()[0], global_step)

            # Update model
            optimizer.step()
            scheduler.step()
            global_step += 1

            # NOTE:Evaluate
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                results = evaluate(args, model, CATEGORY.vocab, NEWS.vocab)
                for key, value in results.items():
                    writer.add_scalar("Eval/{}".format(key), value,
                                      global_step)

            # NOTE: save model
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, model, optimizer, scheduler, global_step)

    writer.close()


def evaluate(args, model, category_vocab, example_vocab, mode='dev'):
    fields, eval_dataset = build_and_cache_dataset(args, mode=mode)
    bucket_iterator = BucketIterator(
        eval_dataset,
        train=False,
        batch_size=args.eval_batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.news),
        device=args.device,
    )
    ID, CATEGORY, NEWS = fields
    CATEGORY.vocab = category_vocab
    NEWS.vocab = example_vocab
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # NOTE: Eval!
    model.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss, eval_steps = 0.0, 0
    labels_list, preds_list = [], []
    for batch in tqdm(bucket_iterator, desc='Evaluation'):
        news, news_lengths = batch.news
        labels = batch.category
        with torch.no_grad():
            logits = model(news, news_lengths)
            loss = criterion(logits, labels)
            eval_loss += loss.item()

        eval_steps += 1
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds)
        labels_list.append(labels)

    y_true = torch.cat(labels_list).detach().cpu().numpy()
    y_pred = torch.cat(preds_list).detach().cpu().numpy()
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')

    # Write into tensorboard
    # TODO: recore false-pos and false-neg samples.
    results = {
        'loss': eval_loss / eval_steps,
        'f1': f1_score,
        'precision': precision,
        'recall': recall
    }
    msg = f'*** Eval: loss {loss}, f1 {f1_score}, precision {precision}, recall {recall}'
    logger.info(msg)
    return results


def main():
    args = get_args()
    writer = SummaryWriter(log_dir=args.output_dir)

    # Check output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if os.path.exists(args.output_dir) \
            and os.listdir(args.output_dir) \
            and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.")

    # Set device
    device = "cuda" if torch.cuda.is_available() \
            and not args.no_cuda else "cpu"
    args.device = torch.device(device)
    logger.warning("Process device: %s", device)

    train(args, writer)


if __name__ == "__main__":
    main()