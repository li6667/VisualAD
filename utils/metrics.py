from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from tabulate import tabulate
import numpy as np
import logging
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    """Per-Region Overlap (PRO) 曲线下面积。

    与原实现的数值完全等价，但把 measure.label/regionprops 提到阈值循环之外：
    掩码在整个阈值扫描过程中是不变的，原实现在 200 个阈值上重复做了 200*N 次
    regionprops，是这里的主要瓶颈（DAGM/VisA 单类要跑十几分钟）。
    """
    masks = np.asarray(masks, dtype=np.float32)
    amaps = np.asarray(amaps, dtype=np.float32)
    n = len(amaps)
    if n == 0:
        return 0.0

    regions_per_img = []
    for i in range(n):
        mask = masks[i]
        if mask.sum() == 0:
            regions_per_img.append(None)
            continue
        labeled = measure.label(mask)
        regions_per_img.append([
            (np.asarray(r.coords[:, 0]), np.asarray(r.coords[:, 1]), float(r.area))
            for r in measure.regionprops(labeled)
        ])

    total_fp_pixels = float(np.sum(1.0 - masks))
    min_th, max_th = float(np.min(amaps)), float(np.max(amaps))
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        pro = []
        for i in range(n):
            regs = regions_per_img[i]
            if not regs:
                continue
            amap = amaps[i]
            for ys, xs, area in regs:
                tp = int(np.sum(amap[ys, xs] > th))
                pro.append(tp / area)
        pros.append(float(np.mean(pro)) if pro else 0.0)
        fp = float(np.sum((amaps > th) * (1.0 - masks)))
        fprs.append(fp / total_fp_pixels if total_fp_pixels != 0 else 0.0)

    pros, fprs = np.asarray(pros), np.asarray(fprs)
    valid = fprs <= expect_fpr
    if not np.any(valid):
        return 0.0
    fpr_valid, pro_valid = fprs[valid], pros[valid]
    if len(fpr_valid) < 2:
        return 0.0
    fpr_norm = (fpr_valid - fpr_valid.min()) / (fpr_valid.max() - fpr_valid.min())
    return auc(fpr_norm, pro_valid)


def compute_metrics(results, obj_list, logger):
    table_ls = []
    for obj in obj_list:
        obj_data = results[obj]
        # Pixel-level data
        # 注意：原实现用 `list.extend(numpy.flatten())` 累积逐像素值，单个类别
        # （如 DAGM Class1：575 张 518x518）就会产生 1.5 亿个 Python float，
        # 约 5GB+ 内存并触发 MemoryError。这里改为 numpy 拼接 + float32。
        if obj_data['imgs_masks']:
            masks = np.stack([
                m.squeeze().cpu().numpy().astype(np.float32, copy=False)
                for m in obj_data['imgs_masks']
            ])
            amaps = np.stack([
                a.squeeze().cpu().numpy().astype(np.float32, copy=False)
                for a in obj_data['anomaly_maps']
            ])
        else:
            masks = np.zeros((0, 0), np.float32)
            amaps = np.zeros((0, 0), np.float32)
        gt_px = masks.ravel()
        pr_px = amaps.ravel()
        # Sample-level data
        gt_sp = np.array(obj_data['gt_sp'])
        pr_sp = np.array(obj_data['pr_sp'])
        
        # Calculate metrics
        pixel_auroc = roc_auc_score(gt_px, pr_px) if gt_px.size else 0
        pixel_ap = average_precision_score(gt_px, pr_px) if gt_px.size else 0
        # 论文报了 PRO（VisA 91.0 / MVTec 87.5），但官方代码把这一行注释掉了，导致表格里 PRO 恒为 0
        pixel_aupro = cal_pro_score(masks, amaps) if gt_px.size else 0
        sample_auroc = roc_auc_score(gt_sp, pr_sp) if gt_sp.size else 0
        sample_ap = average_precision_score(gt_sp, pr_sp) if gt_sp.size else 0
        
        # F1 scores
        precisions, recalls, _ = precision_recall_curve(gt_px, pr_px)
        pixel_f1 = np.max(2 * (precisions * recalls) / (precisions + recalls + 1e-8)) if gt_px.size else 0
        precisions_sp, recalls_sp, _ = precision_recall_curve(gt_sp, pr_sp)
        sample_f1 = np.max(2 * (precisions_sp * recalls_sp) / (precisions_sp + recalls_sp + 1e-8)) if gt_sp.size else 0
        
        # Format table
        table = [
            obj,
            f"{pixel_auroc * 100:.1f}",
            f"{pixel_f1 * 100:.1f}",
            f"{pixel_ap * 100:.1f}",
            f"{pixel_aupro * 100:.1f}",
            f"{sample_auroc * 100:.1f}",
            f"{sample_f1 * 100:.1f}",
            f"{sample_ap * 100:.1f}"
        ]
        table_ls.append(table)
    
    # === New: Calculate and add mean row ===
    if len(table_ls) == 0:
        return

    # Extract numeric part (skip first column class name)
    numeric_data = []
    for row in table_ls:
        numeric_values = [float(x.strip('%')) for x in row[1:]]  # Remove possible % and convert to float
        numeric_data.append(numeric_values)

    # Calculate mean for each column
    mean_values = np.array(numeric_data).mean(axis=0)
    mean_values = [f"{v:.1f}" for v in mean_values]

    # Add mean row
    mean_row = ['Mean'] + mean_values
    table_ls.append(mean_row)

    # === Generate table ===
    headers = ['Class', 'Pixel-AUROC', 'Pixel-F1', 'Pixel-AP', 'Pixel-AUPRO', 
              'Sample-AUROC', 'Sample-F1', 'Sample-AP']
    results_table = tabulate(table_ls, headers=headers, tablefmt='pipe')
    logger.info("\n%s", results_table)