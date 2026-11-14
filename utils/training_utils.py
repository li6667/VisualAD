"""
Training utility functions for VisualAD
"""
import torch
import torch.nn as nn
from .feature_transform import create_feature_transform
from .ats_module import create_ats_transform


def print_training_parameters(args, logger):
    """Print all training parameters before starting training"""
    logger.info(f"Training: {args.train_dataset} | Backbone: {args.backbone} | "
                f"Epochs: {args.epoch} | BS: {args.batch_size} | LR: {args.learning_rate} | "
                f"Image: {args.image_size} | Layers: {args.features_list}")


def validate_training_setup(args, model, device, logger):
    """Validate training setup and requirements"""
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cuda':
        dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
        with torch.no_grad():
            _ = model.encode_image(dummy_input, args.features_list)
        del dummy_input
        torch.cuda.empty_cache()

    import os
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data path does not exist: {args.train_data_path}")


def setup_model_training(model):
    """Configure model parameters for training"""
    for param in model.parameters():
        param.requires_grad = False

    model.visual.anomaly_token.requires_grad = True
    model.visual.normal_token.requires_grad = True

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        ln_post.weight.requires_grad = True
        ln_post.bias.requires_grad = True


def create_optimizer(model, layer_transforms, args, cross_attn=None):
    """Create optimizer with different learning rates for different components"""
    optimizer_params = [
        {'params': [model.visual.anomaly_token], 'lr': args.learning_rate, 'weight_decay': 0.01},
        {'params': [model.visual.normal_token], 'lr': args.learning_rate, 'weight_decay': 0.01},
    ]

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        optimizer_params.append({
            'params': [ln_post.weight, ln_post.bias],
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    for transform in layer_transforms.values():
        alpha_param = getattr(getattr(transform, "ats", None), "alpha", None)
        if alpha_param is not None:
            rest = [p for p in transform.parameters() if p is not alpha_param]
            optimizer_params.append({
                'params': rest,
                'lr': args.learning_rate * 0.1,
                'weight_decay': 0.01
            })
            # 残差标量 alpha 单独给 0.01x 学习率（梯度量级远大于 token），
            # 并在 validate_gradients 里裁剪，防止首步就把 alpha 从 0.05 推飞
            optimizer_params.append({
                'params': [alpha_param],
                'lr': args.learning_rate * 0.01,
                'weight_decay': 0.0
            })
        else:
            optimizer_params.append({
                'params': transform.parameters(),
                'lr': args.learning_rate * 0.1,
                'weight_decay': 0.01
            })

    if cross_attn is not None:
        optimizer_params.append({
            'params': cross_attn.parameters(),
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    return torch.optim.AdamW(optimizer_params, betas=(0.9, 0.999))


def setup_feature_transforms(features_list, device, feature_dim,
                             use_ats: bool = False, use_spectral: bool = True,
                             alpha_init: float = 0.05):
    """Setup feature transformation modules.

    use_ats=False 时与原版 VisualAD 完全一致（单隐层 MLP）；
    use_ats=True  时换成 ATSFeatureTransform（原 MLP + 频域高频残差分支）。
    """
    layer_transforms = nn.ModuleDict()
    for layer_idx in features_list:
        hidden_dim = int(feature_dim * 1.0)
        if use_ats:
            layer_transforms[f'layer_{layer_idx}'] = create_ats_transform(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
                alpha_init=alpha_init,
                use_spectral=use_spectral,
            ).to(device)
        else:
            layer_transforms[f'layer_{layer_idx}'] = create_feature_transform(
                transform_type="mlp", input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim, dropout=0.1
            ).to(device)
    return layer_transforms


def check_for_nan(tensor, name, logger, epoch=None):
    """Check tensor for NaN values and log if found"""
    if torch.isnan(tensor).any():
        msg = f"NaN detected in {name}"
        if epoch is not None:
            msg += f" at epoch {epoch+1}"
        logger.error(msg)
        return True
    return False


def compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice):
    """Compute segmentation loss from similarity maps"""
    seg_losses = []
    for similarity_map in similarity_map_list:
        # Hardcoded: beta_focal=1.0, beta_dice=1.0
        seg_losses.append(loss_focal(similarity_map, gt))
        # Only use anomaly channel to avoid double-counting (since channel 0 = 1 - channel 1)
        seg_losses.append(loss_dice(similarity_map[:, 1, :, :], gt))
    return sum(seg_losses) if seg_losses else torch.tensor(0.0, device=gt.device, requires_grad=False)


def validate_gradients(model, logger, epoch):
    """Validate and clip gradients"""
    if model.visual.anomaly_token.grad is not None:
        if check_for_nan(model.visual.anomaly_token.grad, "anomaly_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.anomaly_token], max_norm=1.0)
    
    if model.visual.normal_token.grad is not None:
        if check_for_nan(model.visual.normal_token.grad, "normal_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.normal_token], max_norm=1.0)

    # ATS 的 alpha 梯度裁剪（量级可达 1e2，不裁会一步跳飞）
    alphas = [m.ats.alpha for m in getattr(model, "_ats_modules", []) if m.ats.alpha.grad is not None]
    if alphas:
        torch.nn.utils.clip_grad_norm_(alphas, max_norm=1.0)

    return True


def save_checkpoint(model, layer_transforms, args, epoch, checkpoint_path, cross_attn=None,
                    uses_ats: bool = False, ats_config=None):
    """Save model checkpoint"""
    transform_state_dict = {name: t.state_dict() for name, t in layer_transforms.items()}

    ln_post = getattr(model.visual, "ln_post", None)
    checkpoint_data = {
        "anomaly_token": model.visual.anomaly_token.data.clone(),
        "normal_token": model.visual.normal_token.data.clone(),
        "ln_post_weight": ln_post.weight.data.clone() if ln_post else None,
        "ln_post_bias": ln_post.bias.data.clone() if ln_post else None,
        "features_list": args.features_list,
        "image_size": args.image_size,
        "epoch": epoch,
        "backbone": args.backbone,
        "layer_transforms": transform_state_dict,
        "transform_type": "ats_mlp" if uses_ats else "mlp",
        "uses_ats": bool(uses_ats),
    }

    if uses_ats:
        checkpoint_data["ats_config"] = ats_config or {
            "use_spectral": True,
            "alpha_init": 0.05,
            "reduction": 16,
        }

    if cross_attn is not None:
        checkpoint_data["cross_attn"] = cross_attn.state_dict()
        checkpoint_data["cross_attn_config"] = {
            "num_anchors": 4,
            "dropout": 0.1,
            "res_scale_init": 0.01,
        }

    torch.save(checkpoint_data, checkpoint_path)
