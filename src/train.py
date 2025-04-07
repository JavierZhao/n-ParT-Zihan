#!/bin/env python3.7

# load standard python modules
import sys

sys.path.insert(0, "../src")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import glob
import argparse
import copy
import tqdm
import gc
from pathlib import Path
import math
from tqdm import tqdm
import filelock  # Add this import at the top

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn import metrics


from src.model import Classifier, GPTConfig, ModelUtils
from src.dataset.ParticleDataset import ParticleDataset

import matplotlib.pyplot as plt
import copy


# set the number of threads that pytorch will use
torch.set_num_threads(2)

torch.autograd.set_detect_anomaly(True)


def load_data(args, dataset_path):
    num_jets = 100 * 1000 if args.small else None
    dataset = ParticleDataset(dataset_path, num_jets=num_jets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return dataloader


def init_model(logfile, config, device="cpu"):
    model = Classifier(
        config, proj_dims="auto" if not args.finetune_mlp else args.finetune_mlp
    ).to(device)
    if not args.from_checkpoint:
        print(model, file=logfile, flush=True)
    return model


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_perf_stats(labels, measures):
    measures = np.nan_to_num(measures)  # Replace NaNs with 0
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, _ = metrics.roc_curve(labels, measures)

    # Only keep fpr/tpr where tpr >= 0.5
    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i] >= 0.5]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i] >= 0.5]

    epsilon = 1e-8  # Small value to avoid division by zero or very small numbers

    # Calculate IMTAFE, handle edge cases
    try:
        if len(tpr2) > 0 and len(fpr2) > 0:
            nearest_tpr_idx = list(tpr2).index(find_nearest(list(tpr2), 0.5))
            imtafe = np.nan_to_num(1 / (fpr2[nearest_tpr_idx] + epsilon))
            if imtafe > 1e4:  # something went wrong
                imtafe = 1
        else:
            imtafe = 1  # Default value if tpr2 or fpr2 are empty
    except (ValueError, IndexError):  # Handle cases where index is not found
        imtafe = 1

    return auc, imtafe


def get_unique_dir(base_dir):
    """Get a unique directory using file locking to prevent race conditions"""
    lock_file = os.path.join(os.path.dirname(base_dir), ".dir_lock")
    lock = filelock.FileLock(lock_file, timeout=60)  # 60 second timeout

    with lock:
        trial_num = 0
        while True:
            trial_dir = os.path.join(base_dir, f"trial-{trial_num}")
            if not os.path.exists(trial_dir):
                os.makedirs(trial_dir)
                return trial_dir
            trial_num += 1


def normalize_matrices(model):
    """Normalize model matrices with proper error handling and type management."""
    try:
        # Normalize encoder input projection
        model.encoder.input_proj.weight.data.copy_(
            ModelUtils.justnorm(model.encoder.input_proj.weight.data, 1)
        )
        # Normalize transformer blocks in encoder
        for block in model.encoder.blocks:
            # QKV projection
            block.qkv.weight.data.copy_(ModelUtils.justnorm(block.qkv.weight.data, 1))

            # Attention output projection
            block.att_c_proj.weight.data.copy_(
                ModelUtils.justnorm(block.att_c_proj.weight.data, 1)
            )

            # MLP layers
            block.c_fc.weight.data.copy_(ModelUtils.justnorm(block.c_fc.weight.data, 1))

            block.mlp_c_proj.weight.data.copy_(
                ModelUtils.justnorm(block.mlp_c_proj.weight.data, 1)
            )

        # Normalize projector layers
        for layer in model.projector.layers:
            # UV gating linear layer
            layer["linear"].weight.data.copy_(ModelUtils.justnorm(layer["linear"].weight.data, 1))

            # Projection layer
            layer["proj"].weight.data.copy_(ModelUtils.justnorm(layer["proj"].weight.data, 1))

    except Exception as e:
        print(f"Error during matrix normalization: {e}")
        raise


def compute_weighted_norms(model):

    norms = {}

    # Compute norms for input embeddings
    input_embeds = model.encoder.input_proj.weight.data
    input_norms = torch.norm(input_embeds.float(), p=2, dim=1).detach().cpu().numpy()
    norms["input_embeddings"] = np.sort(input_norms)[::-1]

    # Compute norms for transformer encoder blocks
    block_norms = []

    for block in model.encoder.blocks:
        # Compute norms along the last dimension (-1), as per justnorm
        qkv_norm = torch.norm(block.qkv.weight.float(), p=2, dim=1).detach().cpu().numpy()
        att_c_proj_norm = (
            torch.norm(block.att_c_proj.weight.float(), p=2, dim=1).detach().cpu().numpy()
        )
        c_fc_norm = torch.norm(block.c_fc.weight.float(), p=2, dim=1).detach().cpu().numpy()
        mlp_c_proj_norm = (
            torch.norm(block.mlp_c_proj.weight.float(), p=2, dim=1).detach().cpu().numpy()
        )

        # Store the mean norm per layer
        block_norms.append(
            np.mean(
                [
                    np.mean(qkv_norm),
                    np.mean(att_c_proj_norm),
                    np.mean(c_fc_norm),
                    np.mean(mlp_c_proj_norm),
                ]
            )
        )

    norms["encoder_blocks"] = np.array(block_norms)

    # Compute norms for projector layers
    projector_norms = []
    for layer in model.projector.layers:
        linear_norm = torch.norm(layer["linear"].weight.float(), p=2, dim=1).detach().cpu().numpy()
        proj_norm = torch.norm(layer["proj"].weight.float(), p=2, dim=1).detach().cpu().numpy()
        projector_norms.append(np.mean([np.mean(linear_norm), np.mean(proj_norm)]))

    norms["projector_layers"] = np.array(projector_norms)

    return norms


def check_normalization(model, tolerance=1e-2):

    norms = compute_weighted_norms(model)

    encoder_mean_norm = np.mean(norms["encoder_blocks"])
    projector_mean_norm = np.mean(norms["projector_layers"])
    input_mean_norm = np.mean(norms["input_embeddings"])

    print(f"Mean Norm (Encoder Blocks): {encoder_mean_norm:.5f}")
    print(f"Mean Norm (Projector Layers): {projector_mean_norm:.5f}")
    print(f"Mean Norm (Input Embeddings): {input_mean_norm:.5f}")

    issues = []

    if abs(encoder_mean_norm - 1.0) > tolerance:
        issues.append("Encoder Blocks")
    if abs(projector_mean_norm - 1.0) > tolerance:
        issues.append("Projector Layers")
    if abs(input_mean_norm - 1.0) > tolerance:
        issues.append("Input Embeddings")

    if issues:
        print(
            f"WARNING: The following components are NOT properly normalized: {', '.join(issues)}"
        )
    else:
        print("All checked model weights are properly normalized.")


def plot_embedding_norms(model, save_path, fig_name="embedding_norms.png"):

    plt.figure(figsize=(8, 6))

    norms = compute_weighted_norms(model)

    input_norms = np.sort(norms["input_embeddings"])[::-1]  # Sorted in descending order
    encoder_norms = np.sort(norms["encoder_blocks"])[::-1]  # Sorted
    projector_norms = np.sort(norms["projector_layers"])[::-1]  # Sorted

    # Normalized rank
    input_rank = np.linspace(0, 1, len(input_norms))
    encoder_rank = np.linspace(0, 1, len(encoder_norms))
    projector_rank = np.linspace(0, 1, len(projector_norms))

    # Plot norms
    plt.plot(
        input_rank, input_norms, label=f"Input Embedding", linestyle="-", marker="o", alpha=0.7
    )
    plt.plot(encoder_rank, encoder_norms, label=f"Encoder ", linestyle="--", marker="s", alpha=0.7)
    plt.plot(
        projector_rank, projector_norms, label=f"Projector ", linestyle="-.", marker="D", alpha=0.7
    )

    # Add reference line for expected norm (1.0)
    plt.axhline(y=1.0, color="black", linestyle="-", linewidth=1, label="Expected Norm")

    # Labels and title
    plt.xlabel("Normalized Rank")
    plt.ylabel("L2 Norm")
    plt.title("Distribution of Norms for Model Components")
    plt.legend()
    plt.grid(True)

    plt.ylim(0.5, 1.5)

    fig_path = os.path.join(save_path, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    # print(f"Figure saved at: {save_path}")
    plt.close()


def main(args):
    t0 = time.time()
    # set up results directory
    config = GPTConfig()
    config.use_nGPT = args.use_nGPT
    out_dir = args.out_dir
    args.learning_rate = 1e-6 * args.batch_size / 128
    args.output_dim = config.output_dim

    # check if experiment already exists and is not empty
    if not args.from_checkpoint:
        out_dir = get_unique_dir(out_dir)
    else:
        # Ensure directory exists for checkpoint loading
        os.makedirs(out_dir, exist_ok=True)
    # initialise logfile
    args.logfile = f"{out_dir}/logfile.txt"
    logfile = open(args.logfile, "a")

    # define the global base device
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        device_type = "cuda"
        for i in range(world_size):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}", file=logfile, flush=True)
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print("Device: CPU", file=logfile, flush=True)
    args.device = device

    checkpoint = {}
    checkpoint_path = os.path.join(out_dir, "last_checkpoint.pt")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        print(f"Previous checkpoint found. Restarting from epoch {checkpoint['epoch'] + 1}")
        args.from_checkpoint = 1

    if not args.from_checkpoint:
        print("logfile initialised", file=logfile, flush=True)
        print("output dimension: " + str(args.output_dim), file=logfile, flush=True)
    else:
        print("loading from checkpoint", file=logfile, flush=True)
    print(f"batch size: {args.batch_size}", file=logfile, flush=True)
    print(f"learning rate: {args.learning_rate}", file=logfile, flush=True)
    print(f"max grad norm: {args.max_grad_norm}", file=logfile, flush=True)
    print(f"use nGPT: {args.use_nGPT}", file=logfile, flush=True)
    print(f"decay lr: {args.decay_lr}", file=logfile, flush=True)

    print("loading data")
    train_dataloader = load_data(args, args.train_dataset_path)
    val_dataloader = load_data(args, args.val_dataset_path)
    if args.small:
        print("using small dataset for training", file=logfile, flush=True)
        print(
            f"number of jets for training: {len(train_dataloader.dataset):e}",
            file=logfile,
            flush=True,
        )
        print(
            f"number of jets for validation: {len(val_dataloader.dataset):e}",
            file=logfile,
            flush=True,
        )
    else:
        print("using full dataset for training", file=logfile, flush=True)
        print(
            f"number of jets for training: {len(train_dataloader.dataset):e}",
            file=logfile,
            flush=True,
        )
        print(
            f"number of jets for validation: {len(val_dataloader.dataset):e}",
            file=logfile,
            flush=True,
        )

    t1 = time.time()

    print(
        "time taken to load and preprocess data: " + str(np.round(t1 - t0, 2)) + " seconds",
        flush=True,
        file=logfile,
    )

    # initialise the network
    model = init_model(logfile, config, args.device)

    if config.use_nGPT == 0:
        min_lr = 0.0
        weight_decay = 0.1
        warmup_iters = 2000
    if config.use_nGPT == 1:
        min_lr = 0.0
        weight_decay = 0.0
        warmup_iters = 0
    beta1 = 0.9
    beta2 = 0.95

    # learning rate decay settings
    decay_lr = args.decay_lr  # whether to decay the learning rate
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(iter_num, epoch):
        # 1) linear warmup for warmup_iters steps
        it = iter_num + epoch * len(train_dataloader)
        if it < warmup_iters:
            return args.learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (args.learning_rate - min_lr)

    optimizer = model.configure_optimizers(
        weight_decay, args.learning_rate, (beta1, beta2), device_type
    )

    loss = nn.CrossEntropyLoss(reduction="mean")
    epoch_start = 0
    l_val_best = 99999
    acc_val_best = 0
    rej_val_best = 0
    # Load the checkpoint
    if args.from_checkpoint:
        # Load state dictionaries
        model.load_state_dict(checkpoint["classifier"])
        optimizer.load_state_dict(checkpoint["opt"])

        # Restore additional variables
        epoch_start = checkpoint["epoch"] + 1
        l_val_best = checkpoint["val loss"]
        acc_val_best = checkpoint["val acc"]
        rej_val_best = checkpoint["val rej"]

    softmax = torch.nn.Softmax(dim=1)
    loss_train_all = []
    loss_val_all = []
    acc_val_all = []

    if config.use_nGPT == 1:
        normalize_matrices(model)

    for epoch in range(epoch_start, args.n_epochs):
        # !!!!!!!
        check_normalization(model)
        print("Model loaded and checked. Ready for training!")

        plot_embedding_norms(model, save_path=out_dir)

        # initialise timing stats
        te_start = time.time()
        te0 = time.time()

        # initialise lists to store batch stats
        losses_e = []
        losses_e_val = []
        predicted_e = []  # store the predicted labels by batch
        correct_e = []  # store the true labels by batch

        # Create a single iterator and wrap it with tqdm
        data_iter = iter(train_dataloader)
        # Prefetch the first batch using the same iterator
        features, labels = next(data_iter)
        # normalize the features (batch_size, 7, 50)
        epsilon = 1e-8
        norm_features = torch.norm(features, p=2, dim=2, keepdim=True)
        features = features / (norm_features + epsilon)
        features = features.to(dtype=torch.float32).pin_memory().to(args.device, non_blocking=True)
        labels = labels.pin_memory().to(args.device, non_blocking=True)

        pbar = tqdm(data_iter, total=len(train_dataloader) - 1, desc="Training")

        for i, (next_features, next_labels) in enumerate(pbar):
            # normalize the features (batch_size, 7, 50)
            epsilon = 1e-8
            norm_features = torch.norm(next_features, p=2, dim=2, keepdim=True)
            next_features = next_features / (norm_features + epsilon)

            if i % 50 == 0:
                check_normalization(model)
                print(f"Model loaded and checked for {epoch}th iter")
                plot_embedding_norms(model, save_path=out_dir, fig_name=f"embedding_norms.png")

            optimizer.zero_grad()
            lr = get_lr(i, epoch) if decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Process current batch (e.g., B1 then B2, etc.)
            out = model(features.transpose(1, 2))
            batch_loss = loss(out, labels.long()).to(args.device)

            # Prefetch next batch asynchronously while processing the current one
            next_features = (
                next_features.to(dtype=torch.float32)
                .pin_memory()
                .to(args.device, non_blocking=True)
            )
            next_labels = next_labels.pin_memory().to(args.device, non_blocking=True)

            # Backward pass and optimization
            batch_loss.backward()
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_grad_norm
                )
                print(f"Gradient norm: {grad_norm}", flush=True, file=logfile)
            optimizer.step()

            if config.use_nGPT == 1:
                normalize_matrices(model)

            # Swap prefetched data into the current batch for the next iteration
            features, labels = next_features, next_labels

            # Log and update progress bar description
            batch_loss_train = batch_loss.detach().cpu().item()
            losses_e.append(batch_loss_train)
            pbar.set_description(f"loss: {batch_loss_train}")

        # Process the final prefetched batch that was not handled in the loop
        if features is not None:
            optimizer.zero_grad()
            out = model(features.transpose(1, 2))
            batch_loss = loss(out, labels.long()).to(args.device)
            batch_loss.backward()
            optimizer.step()

            # Add normalization after optimizer step
            if config.use_nGPT == 1:
                normalize_matrices(model)

            batch_loss_train = batch_loss.detach().cpu().item()
            losses_e.append(batch_loss_train)

        # Compute average loss for the epoch
        loss_e = np.mean(losses_e)
        loss_train_all.append(loss_e)

        te1 = time.time()
        print(f"Training done in {te1-te0} seconds", flush=True, file=logfile)
        te0 = time.time()

        # validation
        with torch.no_grad():
            model.eval()
            # Create iterator and wrap with tqdm
            data_iter = iter(val_dataloader)
            pbar = tqdm(data_iter, total=len(val_dataloader))

            # Prefetch first batch
            try:
                features, labels = next(pbar)
            except StopIteration:
                features, labels = None, None

            if features is not None:
                # Ensure consistent device usage
                features = (
                    features.to(dtype=torch.float32).pin_memory().to(device, non_blocking=True)
                )
                labels = labels.pin_memory().to(device, non_blocking=True)

            for i, (next_features, next_labels) in enumerate(pbar):
                # Process current batch
                out = model(features.transpose(1, 2))
                batch_loss = loss(out, labels.long()).detach().cpu().item()
                losses_e_val.append(batch_loss)
                predicted_e.append(softmax(out).cpu().numpy())
                correct_e.append(labels.cpu())

                # Prefetch next batch
                next_features = (
                    next_features.to(dtype=torch.float32)
                    .pin_memory()
                    .to(device, non_blocking=True)
                )
                next_labels = next_labels.pin_memory().to(device, non_blocking=True)

                # Update progress bar
                pbar.set_description(f"batch val loss: {batch_loss}")

                # Swap prefetched data into the current batch variables
                features, labels = next_features, next_labels

            # Process the final prefetched batch if it exists
            if features is not None:
                out = model(features.transpose(1, 2))
                batch_loss = loss(out, labels.long()).detach().cpu().item()
                losses_e_val.append(batch_loss)
                predicted_e.append(softmax(out).cpu().numpy())
                correct_e.append(labels.cpu())

            loss_e_val = np.mean(np.array(losses_e_val))
            loss_val_all.append(loss_e_val)

        te1 = time.time()
        print(f"validation done in {round(te1-te0, 1)} seconds", flush=True, file=logfile)

        print(
            "epoch: "
            + str(epoch)
            + ", loss: "
            + str(round(loss_train_all[-1], 5))
            + ", val loss: "
            + str(round(loss_val_all[-1], 5)),
            flush=True,
            file=logfile,
        )

        # get the predicted labels and true labels
        predicted = np.concatenate(predicted_e)
        target = np.concatenate(correct_e)

        # get the accuracy
        accuracy = accuracy_score(target, predicted[:, 1] > 0.5)
        print(
            "epoch: " + str(epoch) + ", accuracy: " + str(round(accuracy, 5)),
            flush=True,
            file=logfile,
        )
        acc_val_all.append(accuracy)

        # save the latest model
        torch.save(model.state_dict(), f"{out_dir}/model_last.pt")
        # save the model if lowest val loss is achieved
        if loss_val_all[-1] < l_val_best:
            # print("new lowest val loss", flush=True, file=logfile)
            l_val_best = loss_val_all[-1]
            torch.save(model.state_dict(), f"{out_dir}/model_best_loss.pt")
            np.save(
                f"{out_dir}/validation_target_vals_loss.npy",
                target,
            )
            np.save(
                f"{out_dir}/validation_predicted_vals_loss.npy",
                predicted,
            )
        # also save the model if highest val accuracy is achieved
        if acc_val_all[-1] > acc_val_best:
            print("new highest val accuracy", flush=True, file=logfile)
            acc_val_best = acc_val_all[-1]
            torch.save(model.state_dict(), f"{out_dir}/model_best_acc.pt")
            np.save(
                f"{out_dir}/validation_target_vals_acc.npy",
                target,
            )
            np.save(
                f"{out_dir}/validation_predicted_vals_acc.npy",
                predicted,
            )
        # calculate the AUC and imtafe and output to the logfile
        auc, imtafe = get_perf_stats(target, predicted[:, 1])

        print(
            f"epoch: {epoch}, AUC: {auc}, IMTAFE: {imtafe}",
            flush=True,
            file=logfile,
        )
        if imtafe > rej_val_best:
            print("new highest val rejection", flush=True, file=logfile)

            rej_val_best = imtafe
            torch.save(model.state_dict(), f"{out_dir}/model_best_rej.pt")
            np.save(
                f"{out_dir}/validation_target_vals_rej.npy",
                target,
            )
            np.save(
                f"{out_dir}/validation_predicted_vals_rej.npy",
                predicted,
            )

        # save all losses and accuracies
        np.save(
            f"{out_dir}/loss_train.npy",
            np.array(loss_train_all),
        )
        np.save(
            f"{out_dir}/loss_val.npy",
            np.array(loss_val_all),
        )
        np.save(
            f"{out_dir}/acc_val.npy",
            np.array(acc_val_all),
        )
        te_end = time.time()
        print(
            f"epoch {epoch} done in {round(te_end - te_start, 1)} seconds",
            flush=True,
            file=logfile,
        )
        # save checkpoint, including optimizer state, model state, epoch, and loss
        save_dict = {
            "classifier": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "val loss": loss_val_all[-1],
            "val acc": acc_val_all[-1],
            "val rej": imtafe,
        }
        torch.save(save_dict, f"{out_dir}/last_checkpoint.pt")

    # Training done
    print("Training done", flush=True, file=logfile)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="",
        action="store",
        dest="out_dir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--finetune-mlp",
        default="",
        action="store",
        dest="finetune_mlp",
        type=str,
        help="Size and number of layers of the MLP finetuning head following output_dim of model, e.g. 512-256-128",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        action="store",
        default="/j-jepa-vol/n-ParT-Zihan/data/top/train/",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--val-dataset-path",
        type=str,
        action="store",
        default="/j-jepa-vol/n-ParT-Zihan/data/top/val/",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--n-epoch",
        type=int,
        action="store",
        dest="n_epochs",
        default=300,
        help="Epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=256,
        help="batch_size",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=int,
        action="store",
        dest="from_checkpoint",
        default=0,
        help="whether to start from a checkpoint",
    )
    parser.add_argument(
        "--small",
        type=int,
        action="store",
        dest="small",
        default=0,
        help="whether to use a small dataset (10%) for finetuning",
    )
    parser.add_argument(
        "--decay-lr",
        type=int,
        action="store",
        default=1,
        help="whether to decay the learning rate",
    )
    parser.add_argument(
        "--use-nGPT",
        type=int,
        action="store",
        default=1,
        help="whether to use normalized transformer",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        action="store",
        default=1e-2,
        help="maximum gradient norm",
    )
    args = parser.parse_args()
    main(args)
