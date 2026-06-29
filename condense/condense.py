import os
import json
import math
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from condense.utils import matchloss, selector

from data.utils.tools import get_time, Logger, save_img
from data.utils.transforms import DiffAug, decode_zoom, decode_fn, mix_images

from data.utils.load_dataset import load_dataset, load_normalize
from data.utils.load_model import load_model, get_features


def condense(args, logger, device="cuda"):
    """Optimize condensed data"""
    # Define real dataset and loader
    ripc = args.ipc * args.factor**2

    if "imagenet" in args.dataset and args.dataset != "tinyimagenet":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
        if "conv" in args.model:
            if args.dataset == "imagenet-1k":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(73),
                    transforms.CenterCrop(64)
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(146),
                    transforms.CenterCrop(128)
                ])
        trainset = load_dataset(shuffle=False,
                                dataset=args.dataset,
                                train=True,
                                ipc=args.mipc,
                                transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = load_dataset(dataset=args.dataset,
                                train=True,
                                ipc=args.mipc,
                                transform=transform,
                                shuffle=False)

    loader_real = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    # Define augmentation function
    normalize = load_normalize(args.dataset)
    aug = transforms.Compose(
        [normalize, DiffAug(strategy=args.dsa_strategy, batch=True)])

    logger(f"\nStart condensing with feat matching for {args.niter} iteration")

    model = (load_model(
        model_name=args.model,
        dataset=args.dataset,
        pretrained=True,
    ).eval().to(device))
    for p in model.parameters():
        p.requires_grad = False

    imgs, init_imgs = [], []
    for c, (img, lab) in enumerate(loader_real):
        logger(f"Start syntheizing class {c}")
        # img = img[:ripc].cuda()

        img = selector(
            n=ripc,
            model=lambda x: model(normalize(x)),
            images=img,
            labels=lab,
        )
        img_syn_com = mix_images(
            img,
            args.factor,
            args.ipc,
        )
        img_syn_com.requires_grad = False

        init_imgs.append(img_syn_com.detach().cpu())

        delta = torch.zeros_like(img_syn_com, requires_grad=True)

        optim_img = optim.AdamW(
            [delta],
            0.01,
            # weight_decay=1e-4,
        )
        loss_total = 0

        for it in range(args.niter):
            img_syn, lab_syn = decode_zoom(img_syn_com + delta, lab,
                                           args.factor)

            n = img.shape[0]
            img_aug = aug(torch.cat([img, img_syn], dim=0))

            feat = get_features(model, img_aug, [args.depth])[0][0]

            # feat1 = model.get_feature(img_aug, args.depth, args.depth + 1)[0].flatten(
            #     start_dim=1
            # )

            # print(feat1)
            # print(feat)
            # print(feat.equal(feat1))
            # exit()

            loss = matchloss(feat[:n], feat[n:])

            loss_total = 0.1 * loss.item() + loss_total * 0.9

            optim_img.zero_grad()
            loss.backward()
            optim_img.step()
            if it % (args.niter // 10) == (args.niter // 10) - 1:
                logger(f"{get_time()} (Iter {it:3d}) loss: {loss_total:.10f}")

        imgs.append((img_syn_com + delta).detach().cpu())

    imgs, init_imgs = torch.cat(imgs, dim=0), torch.cat(init_imgs, dim=0)

    targets = torch.tensor(
        [np.ones(args.ipc) * i for i in range(trainset.nclass)],
        dtype=torch.long,
        requires_grad=False,
    ).view(-1)

    torch.save(
        {
            "dataset": args.dataset,
            "data": imgs.detach(),
            "label": targets
        },
        os.path.join(args.save_dir, f"data.pt"),
    )
    print("img and data saved!")

    save_img(
        os.path.join(args.save_dir, f"img{args.niter}.png"),
        imgs.data,
    )

    save_img(
        os.path.join(args.save_dir, f"init_img.png"),
        init_imgs.data,
    )

    init_imgs, _ = decode_fn(init_imgs, targets, args.factor)

    save_img(
        os.path.join(args.save_dir, f"dec_init_img.png"),
        init_imgs.data,
    )

    imgs, _ = decode_fn(imgs, targets, args.factor)

    save_img(
        os.path.join(args.save_dir, f"dec_img{args.niter}.png"),
        imgs.data,
    )


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir, "condensation_log")
    logger(json.dumps(args.__dict__, indent=2))

    condense(args, logger)
