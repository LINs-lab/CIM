"""Smoke test: full CIM pipeline on CIFAR-10 IPC=10 ResNet-18_modified.

Expected (paper Tab. 1 / Tab. 6):  ~66.2-66.6 Top-1
"""
import sys
import time
sys.argv = [
    "main.py",
    "-d", "cifar10",
    "-m", "resnet18_modified",
    "--depth", "-1",
    "--ipc", "10",
    "--epochs", "1000",
]

t0 = time.time()
from argument import args
print("ARGS:", args.__dict__, flush=True)

from condense.condense import main as synt_main
synt_main(args)
print(f"[CONDENSE done in {time.time()-t0:.1f}s]", flush=True)

from data.utils.evaluator import eval_data
t1 = time.time()
eval_data(
    save_dir=args.save_dir,
    model_ls=[args.model],
    tar_model_ls=[args.model],
    factor=args.factor,
    epochs=1000,
    batch_size=None,
    crop_method="factor",
    mix_type="vanilla",
    dsa_strategy="color_crop_cutout_flip_scale_rotate",
    store_log=True,
    eval_times=3,
    num_val=4,
    zca=False,
    logger_name="evaluation_log",
)
print(f"[EVAL done in {time.time()-t1:.1f}s]", flush=True)
print(f"[TOTAL {time.time()-t0:.1f}s]")
