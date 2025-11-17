import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
import os
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted as model_adapter
import Datasets.visa as visa
import Datasets.mvtec as mvtec
import time
import argparse
from tools.loss import FocalLoss, BinaryDiceLoss
from tools.utils import get_anomaly_map
from torch.optim.lr_scheduler import LambdaLR
import math
from Datasets import DATASET_REGISTRY, DATASET_CLASSES

visa_ALL = {"candle", "capsules", "cashew", "chewinggum",
            "fryum", "macaroni1", "macaroni2", "pcb1",
            "pcb2", "pcb3", "pcb4", "pipe_fryum"}

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

def _downsample_mask_to_tokens(mask_bchw: torch.Tensor, S: int) -> torch.Tensor:
    if mask_bchw.dim() == 3:
        mask_bchw = mask_bchw.unsqueeze(1)
    m_small = F.interpolate(mask_bchw.float(), size=(S, S), mode="nearest")
    return (m_small > 0.5).squeeze(1)

def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"❌ Unsupported dataset: {dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    if getattr(args, "dataset_root", None):
        root_path = args.dataset_root

    test_dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=512,
        imagesize=512,
    )

    print(f"✅ Loaded [{dataset_name}] ({category}) train set, size: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    return test_loader

def train_epoch(optimizer, loss_focal, loss_dice, epoch, anomaly_awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data):
    for idx, image_info in enumerate(train_data):
        anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score = get_anomaly_map(clip_model, image_info, device, model, Dino_model)
        anomaly_awareness_loss = loss_focal(anomaly_map, mask) + loss_dice(anomaly_map[:, 1, :, :], mask)
        seg_loss = loss_focal(anomaly_map_cross_modal, mask) + loss_dice(anomaly_map_cross_modal[:, 1, :, :], mask)
        global_anomaly_loss = F.cross_entropy(global_anomaly_score.squeeze(1), image_info["is_anomaly"].to(device).long())
        loss = 0.25 * anomaly_awareness_loss + 0.5 * seg_loss + 0.25 * global_anomaly_loss
        print(
                f"Epoch {epoch+1}/{10} | Batch {idx+1}/{len(train_data)} "
                f"| loss: {loss.item():.4f} | anomaly_awareness_loss: {anomaly_awareness_loss.item():.4f} | seg_loss: {seg_loss.item():.4f} | global_anomaly_loss: {global_anomaly_loss.item():.4f} | Time: {time.time()-start_time:.2f}s",
                end="\r",
                flush=True
        )
        anomaly_awareness_loss_list.append(anomaly_awareness_loss.item())
        seg_loss_list.append(seg_loss.item())
        global_anomaly_loss_list.append(global_anomaly_loss.item())
        loss_list.append(loss.item())
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return 

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result", help="path to result")
    parser.add_argument("--device", type=str, default="cuda:6", help="device")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--dataset", type=str, default="visa", help="dataset")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--lr", type=float, default=0.00001, help="lr")
    parser.add_argument("--backbone_name", type=str, default="dinov3_vitl16", help="dinov3 backbone name")
    parser.add_argument("--dinov3_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", help="dinov3 weights path")
    parser.add_argument("--dataset_root", type=str, default=None, help="override dataset root path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.result_path}", exist_ok=True)
    
    repo_dir = './dinov3'
    Dino_model = torch.hub.load(repo_dir, args.backbone_name, source='local', weights=args.dinov3_weights)
    Dino_model.to(device)
    Dino_model.eval()

    # loading clip
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device=device, pretrained='openai', require_pretrained=True)
    clip_model.to(device)
    clip_model.eval()

    anchor = getattr(Dino_model, "norm", None) or getattr(Dino_model, "fc_norm", None)
    c_in = anchor.normalized_shape[0] if anchor is not None else getattr(Dino_model, "embed_dim", 1024)
    model = model_adapter(c_in=c_in, device=device)
    model.to(device)
    model.train()

    update_params = ['patch_token_adapter', 'cls_token_adapter', 'prompt_adapter']
    params_to_update = []
    for name, param in model.named_parameters():
        for update_name in update_params:
            if update_name in name:
                print(f"Learnable parameter: {name}")
                params_to_update.append(param)

    train_data = prepare_data(args.dataset, 'ALL', args, **kwargs)

    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    total_steps = args.epoch * len(train_data)
    warmup_steps = int(0.03 * total_steps)

    # scheduler = LambdaLR(optimizer, lr_lambda)

    # If you want to speed up the convergence, please use the following line of code.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch/10 + 1))

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    
    for epoch in range(args.epoch):
        start_time = time.time()
        awareness_loss_list, seg_loss_list, loss_list, global_anomaly_loss_list = [], [], [], []

        train_epoch(optimizer, loss_focal, loss_dice, epoch, awareness_loss_list, seg_loss_list, global_anomaly_loss_list, loss_list, clip_model, start_time, train_data)
        print()
        # scheduler.step()

        os.makedirs(f"{args.result_path}/ckpt", exist_ok=True)
        torch.save({'cls_token_adapter': model.cls_token_adapter.state_dict(), 'patch_token_adapter': model.patch_token_adapter.state_dict(), 'prompt_adapter': model.prompt_adapter.state_dict()}, f"{args.result_path}/ckpt/{epoch}.pth")
        
        with open(f"{args.result_path}/loss.txt", "a") as f:
            f.write(
                f"epoch_{epoch}: "
                f"awareness_loss={np.mean(awareness_loss_list):.6f}\t"
                f"seg_loss={np.mean(seg_loss_list):.6f}\t"
                f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}\t"
                f"total_loss={np.mean(loss_list):.6f}\n"
            )
        print(
            f"epoch_{epoch}: "
            f"awareness_loss={np.mean(awareness_loss_list):.6f}, "
            f"seg_loss={np.mean(seg_loss_list):.6f}, "
            f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}, "
            f"total_loss={np.mean(loss_list):.6f}"
        )
