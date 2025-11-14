import torch
import torch.nn.functional as F
from utils import encode_text_with_prompt_ensemble

def get_feature_dinov3(image_path, batch_img, device, Dino_model):
    with torch.no_grad():
        depth = len(Dino_model.blocks)
        idxs = [int(round((depth - 1) * k / (4 - 1))) for k in range(4)]
        layers = sorted(set([max(0, min(depth - 1, i)) for i in idxs]))
        patch_tokens_dict = {i: [] for i in layers}
        cls_tokens_dict = {i: [] for i in layers}

        for j in range(len(image_path)):
            patch_dict, tokens_dict, cls_dict = {}, {}, {}
            handles = []

            image = batch_img[j].unsqueeze(0).to(device)

            anchor = getattr(Dino_model, "norm", None) or getattr(Dino_model, "fc_norm", None)
            assert anchor is not None, "There is no norm/fc_norm module, please print(Dino_model) to confirm the name"

            for i in layers:
                def _mk_hook(idx):
                    def _hook(module, inp, out):
                        tokens_dict[idx] = anchor(out[0]).detach().cpu()
                    return _hook
                handles.append(Dino_model.blocks[i].register_forward_hook(_mk_hook(i)))

            with torch.inference_mode():
                _ = Dino_model(image)

            for h in handles:
                h.remove()

            for i, toks in tokens_dict.items():
                tokens = toks[:, 5:, :]
                tokens = (tokens - tokens.mean(dim=1, keepdim=True)) / (
                    tokens.std(dim=1, keepdim=True) + 1e-6
                )
                patch_dict[i] = tokens
                cls_dict[i] = toks[:, 0, :].unsqueeze(1)

            for i in layers:
                patch_tokens_dict[i].append(patch_dict[i])
                cls_tokens_dict[i].append(cls_dict[i])

        patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0).to(device) for i in layers]
        cls_token = [torch.cat(cls_tokens_dict[i], dim=0).to(device) for i in layers]

        return cls_token, patch_tokens

def get_anomaly_map(clip_model, image_info, device, model, Dino_model):
    image = image_info["image"].to(device)
    image_path = image_info["image_path"]
    mask = image_info["mask"].to(device)
    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
    y = image_info["is_anomaly"]

    # textual branch
    text_feature = torch.zeros(len(image_path), 768, 2).to(device)
    with torch.no_grad():
        for i in range(len(image_path)):
            text_feature[i] = encode_text_with_prompt_ensemble(clip_model, image_path[i].split('/')[-4], device, '', y)
    adjusted_feats_0 = []
    adjusted_feats_1 = []
    for i in range(len(image_path)):
        f0 = model.prompt_adapter[0](text_feature[i, :, 0])[0]
        f1 = model.prompt_adapter[1](text_feature[i, :, 1])[0]
        adjusted_feats_0.append(f0)
        adjusted_feats_1.append(f1)
    adjusted_feats_0 = torch.stack(adjusted_feats_0, dim=0)
    adjusted_feats_1 = torch.stack(adjusted_feats_1, dim=0)
    adjusted_text_feature = torch.cat([adjusted_feats_0, adjusted_feats_1], dim=1).view(len(image_path), 768, 2) 

    # visual branch
    cls_token, patch_tokens = get_feature_dinov3(image_path, image, device, Dino_model)
    anomaly_map = []
    anomaly_maps_cross_modal = []
    global_anomaly_scores = []
    for i in range(4):
        cls_features = model.cls_token_adapter[i](cls_token[i])[0]
        cls_features = cls_features / cls_features.norm(dim=-1, keepdim=True)
        patch_features = model.patch_token_adapter[i](patch_tokens[i])[0]
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        # cross-modal contrastive learning
        anomaly_map_cross_modal = 100 * patch_features @ adjusted_text_feature
        S = int((patch_tokens[i].shape[1]) ** 0.5)
        anomaly_map_cross_modal = F.interpolate(anomaly_map_cross_modal.permute(0, 2, 1).view(-1, 2, S, S),
                                    size=512, mode='bilinear', align_corners=True)
        anomaly_map_cross_modal = torch.softmax(anomaly_map_cross_modal, dim=1)
        anomaly_maps_cross_modal.append(anomaly_map_cross_modal)

        # anomaly-aware calibration
        anomaly_awareness_cls_patch = 10 * patch_features @ cls_features.squeeze().unsqueeze(-1)
        anomaly_awareness_cls_patch = F.interpolate(anomaly_awareness_cls_patch.permute(0, 2, 1).view(-1, 1, S, S),
                                    size=512, mode='bilinear', align_corners=True)
        anomaly_awareness_cls_patch = torch.sigmoid(anomaly_awareness_cls_patch)
        anomaly_map.append(torch.cat([1 - anomaly_awareness_cls_patch, anomaly_awareness_cls_patch], dim=1))

        # global anomaly score  
        anomaly_score = 100 * cls_features @ adjusted_text_feature
        global_anomaly_scores.append(anomaly_score)
    
    anomaly_map_cross_modal = torch.mean(torch.stack(anomaly_maps_cross_modal, dim=0), dim=0)
    anomaly_awareness = torch.mean(torch.stack(anomaly_map, dim=0), dim=0)
    global_anomaly_score = torch.mean(torch.stack(global_anomaly_scores, dim=0), dim=0)

    return anomaly_awareness, mask, anomaly_map_cross_modal, global_anomaly_score
