# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import numpy as np
from joblib import dump
from torch.utils import data
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
import argparse
from hashlib import md5
from datetime import datetime
from pathlib import Path
import json
import sys

import datasets
from models import ModelForSequenceClassification
import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument("--resume_from", type=str, required=False, default=None)
    parser.add_argument(
        "--task", type=str, required=True,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tqdm", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_head", default=False, action="store_true")

    hparams = {
        "head": "multilayer",  # 'linear' or 'multilayer'
        "batch_size": 4,
        "grad_acc_steps": 8,
        "max_seq_length": 256,
        "epochs": 10,
        "hidden_dim": 1024,
        "learning_rate": 1e-5,
    }

    for key, value in hparams.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))

    args = parser.parse_args()
    args.device = torch.device(args.device)

    if args.seed is None:
        args.seed = np.random.randint(0, 100_000)

    param_str = ""
    for key, value in args.__dict__.items():
        param_str += f"{key}={value}"


    if args.resume_from:
        args.run_hash = args.resume_from
    else:
        args.run_hash = (
            args.task
            + "_"
            + datetime.now().strftime("%d%m%H%M")
            + "_"
            + md5(param_str.encode("utf-8")).hexdigest()[:6]
        )
    args.log_dir = (Path(__file__) / "..").resolve() / "logs" / args.run_hash
    if not args.resume_from:
        args.log_dir.mkdir(parents=True)

    sys.stdout = open(args.log_dir / "out.log", "a")
    sys.stderr = sys.stdout

    if args.resume_from:
        print(f"Resuming from {args.resume_from}", flush=True)

    print("\nParameters:")
    for key, value in args.__dict__.items():
        print(f"{key}={value}", flush=True)

    return args


def evaluate(model, val_loader, unshuffled_train_loader):
    print("Get predictions for val_loader", flush=True)
    result = get_predictions(model, val_loader)
    print("Get predictions for unshuffled_train_loader", flush=True)
    extraction_result = get_predictions(model, unshuffled_train_loader)

    if n_labels == 1:
        preds = result[0][:, 0] > 0.5
    else:
        preds = np.argmax(result[0], 1)

    print("valid accuracy:", (preds == val_data["label"].values).mean(), flush=True)

    epoch_val_features = np.concatenate([result[1], result[0]], 1)
    epoch_extracted_features = np.concatenate(
        [extraction_result[1], extraction_result[0]], 1
    )

    return epoch_val_features, epoch_extracted_features


if __name__ == "__main__":
    args = get_args()
    utils.seed_everything(args.seed)

    resume_state = None
    if args.resume_from:
        resume_state = torch.load(f"logs/{args.run_hash}/state.pt")

    if not args.resume_from:
        json.dump(
            {key: str(value) for key, value in args.__dict__.items()},
            open(args.log_dir / "params.json", "w"),
            indent=4,
        )

    dataset = datasets.DATASETS[args.task]

    n_labels = dataset.get_n_labels()
    train_data, val_data, test_data = dataset.load()



    print("Loading Roberta Large MNLI Model ...", flush=True)
    model = ModelForSequenceClassification(
        AutoModel.from_pretrained(str(args.model_path)),
        hidden_dim=args.hidden_dim,
        n_labels=n_labels,
        head=args.head,
    ).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))

    if args.load_head:
        assert args.head == "multilayer"

        x = AutoModelForSequenceClassification.from_pretrained(str(args.model_path))
        model.classifier.load_state_dict(x.classifier.state_dict())

    if args.resume_from:
        print(f"Loading model state dict from {args.resume_from}")
        model.load_state_dict(resume_state["model_state_dict"])

    print("Loading train dataset", flush=True)
    train_dataset = dataset(tokenizer, train_data, max_length=args.max_seq_length)

    print("Loading val dataset", flush=True)
    val_dataset = dataset(tokenizer, val_data, max_length=args.max_seq_length)

    print("Loading test dataset", flush=True)
    test_dataset = dataset(tokenizer, test_data, max_length=args.max_seq_length)

    train_loader = data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    unshuffled_train_loader = data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
    )
    print("Setting up optimizer", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume_from:
        print(f"Loading optimizer state dict from {args.resume_from}")
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])


    print("Setting up scheduler", flush=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=int(len(train_loader) / args.grad_acc_steps * args.epochs),
    )
    if args.resume_from:
        print(f"Loading scheduler state dict from {args.resume_from}")
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])


    def get_predictions(model, loader):
        predictions = []
        labels = []
        features = []

        for batch in tqdm(loader, disable=not args.tqdm):
            input_ids = batch[0][0].to(args.device)
            attention_mask = batch[0][1].to(args.device)

            try:
                labels.append(batch[1][0].numpy())
            except IndexError:
                pass

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs[0]
            features.append(outputs[1].detach().cpu().numpy())
            if n_labels == 1:
                batch_predictions = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                batch_predictions = torch.softmax(logits, 1).detach().cpu().numpy()

            predictions.append(batch_predictions)

        out = [np.concatenate(predictions, 0), np.concatenate(features, 0)]

        if len(labels) > 0:
            out.append(np.concatenate(labels, 0))

        return out

    mode = "w+" if not args.resume_from else "r+"

    print("Creating numpy arrays 1", flush=True)
    all_train_features = np.memmap(f"logs/{args.run_hash}/__all_train_features.numpy", dtype='float64', mode=mode, shape=(args.epochs, len(train_dataset), args.hidden_dim + n_labels))

    print("Creating numpy arrays 2", flush=True)
    all_train_labels = np.memmap(f"logs/{args.run_hash}/__all_train_labels.numpy", dtype="float64", mode=mode, shape=(args.epochs, len(train_dataset)))

    print("Creating numpy arrays 3", flush=True)
    all_extracted_train_features = np.memmap(f"logs/{args.run_hash}/__all_extracted_train_features.numpy", dtype="float64", mode=mode, shape=(args.epochs + 1, len(train_dataset), args.hidden_dim + n_labels))

    print("Creating numpy arrays 4", flush=True)
    all_val_features = np.memmap(f"logs/{args.run_hash}/__all_val_features.numpy", dtype="float64", mode=mode, shape=(args.epochs + 1, len(val_dataset), args.hidden_dim + n_labels))


    if not args.resume_from:
        print("Putting model in eval mode", flush=True)
        model.eval()
        print("Running first evaluation", flush=True)
        epoch_val_features, epoch_extracted_features = evaluate(
            model, val_loader, unshuffled_train_loader
        )
        print("Finished first evaluation, starting epochs", flush=True)

        all_extracted_train_features[0] = epoch_extracted_features
        all_val_features[0] = epoch_val_features

    epoch_range = range(resume_state["epoch"] + 1, args.epochs) if args.resume_from else range(args.epochs)

    for epoch in tqdm(epoch_range, desc="Epoch", disable=not args.tqdm):
        print(f"Epoch {epoch}", flush=True)

        epoch_train_loss = 0
        model.train()
        model.zero_grad()

        epoch_features = []
        epoch_labels = []

        for step, batch in enumerate(tqdm(train_loader, disable=not args.tqdm)):
            input_ids = batch[0][0].to(args.device)
            attention_mask = batch[0][1].to(args.device)
            labels = batch[1][0].to(args.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss / args.grad_acc_steps

            epoch_features.append(
                np.concatenate(
                    [
                        outputs[2].detach().cpu().numpy(),
                        outputs[1].detach().cpu().numpy(),
                    ],
                    1,
                )
            )
            epoch_labels.append(labels.detach().cpu().numpy())

            epoch_train_loss += loss.item()
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                model.zero_grad()

        model.eval()

        epoch_val_features, epoch_extracted_features = evaluate(
            model, val_loader, unshuffled_train_loader
        )
        epoch_features = np.concatenate(epoch_features, 0)
        epoch_labels = np.concatenate(epoch_labels, 0)

        all_extracted_train_features[epoch + 1] = epoch_extracted_features
        all_train_features[epoch] = epoch_features
        all_train_labels[epoch] = epoch_labels
        all_val_features[epoch + 1] = epoch_val_features




        all_train_features.flush()
        all_extracted_train_features.flush()
        all_train_labels.flush()
        all_val_features.flush()

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }

        torch.save(state, f"logs/{args.run_hash}/state.pt")



    test_result = get_predictions(model, test_loader)
    test_features = np.concatenate([test_result[1], test_result[0]], 1)

    dump(
        {
            "train_features": all_train_features,
            "extracted_train_features": all_extracted_train_features,
            "train_labels": all_train_labels,
            "val_features": all_val_features,
            "test_features": test_features,
        },
        args.log_dir / "features.joblib",
    )
