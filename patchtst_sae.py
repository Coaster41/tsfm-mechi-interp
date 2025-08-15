from transformer_lens import (HookedTransformer, utils)
from transformer_lens.hook_points import HookPoint
import functools
import torch
import matplotlib.pyplot as plt
from torch import Tensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import PatchTSTForPrediction
from transformers.models.patchtst.modeling_patchtst import (
    PatchTSTForPredictionOutput
)
from data_loader import *
import pandas as pd

import torch
import os
import argparse


from sae_lens import (
    SAE,
    upload_saes_to_huggingface,
    LanguageModelSAERunnerConfig,
    TimeSeriesModelSAERunnerConfig,
    TimeSeriesModelSAETrainingRunner,
    SAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
    HookedSAETransformer,
)

parser = argparse.ArgumentParser()
parser.add_argument("--expansion_factor", type=int, default=16, help="expansion_factor")
parser.add_argument("--l1_coef", type=float, default=5, help="l1_coef")
parser.add_argument("--block_num", type=int, default=0, help="block_num")

args = parser.parse_args()


model = HookedTransformer.from_pretrained("patchtst_relu", center_unembed=False)

# Loading tsmixup dataset
train_dataset, val_dataset = create_cached_tsmixup_datasets(
        max_samples=None,
        context_length=512,
        prediction_length=1, # 1 or 96
        num_workers=8,
        cache_dir="/extra/datalab_scratch0/ctadler/time_series_models/mechanistic_interpretability/data/tsmixup_cache/",
        processed_cache_path="/extra/datalab_scratch0/ctadler/time_series_models/mechanistic_interpretability/data/tsmixup_cache/tsmixup_processed_None_512_1.pkl",
        batch_size=4000
    )


total_training_steps = 100_000  # probably we should do more
batch_size = 4096*2
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 10  # 10% of training
d_in = 256
expansion_factor = args.expansion_factor
l1_coef = args.l1_coef
block_num = args.block_num
hook_name = f"blocks.{block_num}.hook_mlp_out"

cfg = TimeSeriesModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="patchtst_relu",  # my model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name=hook_name,  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    dataset_path="autogluon/chronos_datasets",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    is_dataset_tokenized=True,
    dataset_dtype="torch.float32",
    streaming=False,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    sae=StandardTrainingSAEConfig(
        d_in=d_in,  # the width of the mlp output.
        d_sae=d_in * expansion_factor,  # the width of the SAE. Larger will result in better stats but slower training.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_activations="none",
        l1_coefficient=l1_coef,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    ),
    # Training Parameters
    lr=1e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    train_batch_size_tokens=batch_size,
    context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    num_patches=32,
    # Activation Store Parameters
    n_batches_in_buffer=64*16,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="patchtst_sae_lens_tests",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        run_name=f"patchtst_relu_{expansion_factor}_{l1_coef}_{block_num}"
    ),
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)
# look at the next cell to see some instruction for what to do while this is running.
sparse_autoencoder = TimeSeriesModelSAETrainingRunner(cfg, override_dataset=train_dataset).run()

# Test loading/uploading
hf_repo_id = f"Coaster41/patchtst-sae-{expansion_factor}-{l1_coef}-{block_num}"
sae_id = hook_name

upload_saes_to_huggingface({sae_id: sparse_autoencoder}, hf_repo_id=hf_repo_id)

sae_test = SAE.from_pretrained(
    release=hf_repo_id, sae_id=sae_id, device=str(device)
)