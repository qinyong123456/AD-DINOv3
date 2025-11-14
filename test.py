from utils import encode_text_with_prompt_ensemble
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
import os
from CLIP.clip import create_model
import torch.nn.functional as F
from CLIP.adapter import CLIP_Inplanted as model_adapter
import argparse
from tools.utils import get_anomaly_map
from tools.visualization import visualization
from Datasets import DATASET_REGISTRY, DATASET_CLASSES

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"❌ Unsupported dataset: {dataset_name}. "
                         f"Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]

    test_dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=512,
        imagesize=512,
    )

    print(f"✅ Loaded [{dataset_name}] ({category}) test set, size: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    return test_loader

def test(clip_model, result_path, epoch):
    AUROC = []
    F1 = []
    print(f"--------------------------------------Testing epoch {epoch}--------------------------------------")
    for category in sorted(DATASET_CLASSES[args.dataset]):
        os.makedirs(result_path + f"/{category}", exist_ok=True)
        pixel_pred = []
        pixel_gt = []
        image_pred = []
        image_gt = []
        img_list = []
        test_data = prepare_data(args.dataset, category, args, **kwargs)

        for image_info in tqdm(test_data):
            _, mask, anomaly_map_cross_modal, _ = get_anomaly_map(clip_model, image_info, device, model, Dino_model)
            pixel_gt.extend(mask.squeeze(1).cpu().detach().numpy())
            img_list.extend(image_info["image_path"])
            pixel_pred.extend(anomaly_map_cross_modal[:, 1, :, :].cpu().detach().numpy())

        gt_mask_list = np.array(pixel_gt)
        pred_mask_list = np.array(pixel_pred)
        pred_mask_list = (pred_mask_list - pred_mask_list.min()) / (pred_mask_list.max() - pred_mask_list.min())
        visualization(img_list, pred_mask_list, gt_mask_list, category, result_path)

        auroc = roc_auc_score(gt_mask_list.flatten(), pred_mask_list.flatten())
        AUROC.append(auroc)

        precisions, recalls, _ = precision_recall_curve(gt_mask_list.flatten(), pred_mask_list.flatten())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        f1 = np.max(f1_scores[np.isfinite(f1_scores)])
        F1.append(f1)

        print(f"{category}: AUC_Pixel={auroc:.5f}\t F1_Pixel={f1:.5f}")
    
    AUROC_all = np.array(AUROC)
    print(f"mean_AUC_Pixel: ", np.mean(AUROC_all))

    F1_all = np.array(F1)
    print("mean_F1_Pixel: ", np.mean(F1_all))

    metric_file = f"{result_path}/metric.txt"

    with open(metric_file, "a") as f:
        f.write(f"----------Dataset: {args.dataset}----------\n")
        f.write(f"Classname    AUC_Pixel/F1_Pixel_epoch_{epoch}\n")
        for i, cname in enumerate(sorted(DATASET_CLASSES[args.dataset])):
            f.write(f"{cname:<14s}{AUROC[i]:.5f}/{F1[i]:.5f}\n")
        f.write(f"{'mean':<14s}{np.mean(AUROC_all):.5f}/{np.mean(F1_all):.5f}\n\n")
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./Result", help="path to result")
    parser.add_argument("--device", type=str, default="cuda:1", help="device")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--dataset", type=str, default="mvtec", help="dataset")
    parser.add_argument("--backbone_name", type=str, default="dinov3_vitl16", help="dinov3 backbone name")
    parser.add_argument("--dinov3_weights", type=str, default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", help="dinov3 weights path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.result_path}/{args.dataset}", exist_ok=True)
    
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
    for i in range(100):
        ckpt = f'./Result/ckpt/{i}.pth'
        model.patch_token_adapter.load_state_dict(torch.load(ckpt, map_location=device)['patch_token_adapter'])
        model.cls_token_adapter.load_state_dict(torch.load(ckpt, map_location=device)['cls_token_adapter'])
        model.prompt_adapter.load_state_dict(torch.load(ckpt, map_location=device)['prompt_adapter'])
        model.to(device)
        model.eval()
        test(clip_model, f"{args.result_path}/{args.dataset}", i)
