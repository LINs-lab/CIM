import argparse

parser = argparse.ArgumentParser(description="Efficient Dataset Distillation")
# Dataset
parser.add_argument(
    "-d",
    "--dataset",
    default="cifar100",
    type=str,
    help=
    "dataset (options: mnist, fashion, svhn, cifar10, cifar100, and imagenet)",
)
parser.add_argument("-j",
                    "--workers",
                    default=4,
                    type=int,
                    help="number of data loading workers")
parser.add_argument("--ipc",
                    type=int,
                    default=10,
                    help="number of condensed data per class")
parser.add_argument("-f",
                    "--factor",
                    type=int,
                    default=2,
                    help="multi-formation factor. (1 for IDC-I)")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="conv3",
    help="model used for distillation",
)
parser.add_argument(
    "-a",
    "--dsa_strategy",
    type=str,
    # default="color_crop_cutout_flip_scale_rotate",
    default="crop_cutout_flip",
)
parser.add_argument("-i",
                    "--niter",
                    type=int,
                    default=200,
                    help="number of outer iteration")
parser.add_argument(
    "--mipc",
    type=int,
    default=300,
    help="number of samples in pre-selected subset per clas",
)
parser.add_argument("--depth",
                    type=int,
                    default=-1,
                    help="number of outer iteration")
# evaluation
parser.add_argument(
    "--epochs",
    type=int,
    default=4000,
    help="number of samples in pre-selected subset per clas",
)
parser.add_argument(
    "--mix_type",
    type=str,
    default="vanilla",
)
# parser.add_argument(
#     "--crop_method",
#     type=str,
#     default="factor",
# )
args = parser.parse_args()

# define depth of features getting
# if args.model == "resnet18":
#     args.depth = -4
# elif args.model == "resnet18_modified":
#     args.depth = -1
# elif "conv" in args.model:
#     args.depth = int(args.model[-1]) - 1

args.save_dir = f"./results/{args.dataset}_{args.model}/ipc{args.ipc}"
