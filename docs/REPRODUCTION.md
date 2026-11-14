# 复现记录（Evaluation Reproduction）

本文件记录在本机（Windows 11 + RTX 2070 SUPER 8GB + torch 2.0.0+cu118）上
复现 CVPR 2026 VisualAD 论文数字的完整过程、数据准备方式与结果对照。

---

## 1. 环境

| 项 | 值 |
|---|---|
| OS | Windows |
| GPU | NVIDIA GeForce RTX 2070 SUPER（8GB） |
| 驱动 | 536.23 |
| Python | 3.9.5（venv 位于仓库外的 `.venv/`） |
| torch / torchvision | 2.0.0+cu118 / 0.15.1+cu118 |
| numpy | 1.24.4（**必须 <1.25**，否则 scipy 1.9.1 与 torch 2.0 都会崩） |
| opencv-python | 4.8.0.76（**5.x 强制 numpy>=2，不可用**） |

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
.venv/Scripts/python -m pip install -r VisualAD/requirements.txt
```

> 国内网络提示：`download.pytorch.org` 官方源在部分网络下只有 ~150KB/s，
> 可改用镜像 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/` 直接下 wheel
> 再本地安装（实测 6.5MB/s）。

## 2. 预训练权重

`VisualAD_lib/model_load.py` 会在首次运行时自动下载 OpenAI CLIP 主干：

```
~/.cache/clip/ViT-L-14-336px.pt          934,088,680 B
sha256 = 3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02
```

校验后与 `_MODELS["ViT-L/14@336px"]` 完全一致；加载后确认：

```
visual.conv1.weight          (1024, 3, 14, 14)
visual.positional_embedding  (577, 1024)        # 24x24+1 -> 336px/14
```

训练好的 adapter（126MB ×2）来自仓库 `weight/`（Git LFS）。

## 3. 数据集

论文用的 13 个数据集不在本仓库内。作者提供的是 **Google Drive 目录（散装图片）**，
不是压缩包，且 `gdown` 对 >50 文件的目录会直接报 `FolderContentsMaximumLimitError`。
本机使用 `datasets/gdrive_sync.py`：`embeddedfolderview` 递归列目录 + 逐文件下载 +
manifest 断点续传 + 多线程。

| 数据集 | 来源 | 规模 | meta.json |
|---|---|---|---|
| VisA | 官方 `VisA_20220922.tar`（AWS S3，1.93GB） | 12039 文件 | `generate_dataset_json/visa.py`（需 `split_csv/1cls.csv`） |
| MVTec-AD | 作者 Drive | 6645 文件 / 5.0GB | `generate_dataset_json/mvtec.py` |
| BTAD | 作者 Drive（`BTech_Dataset_transformed/`） | 2833 文件 / 5.6GB | `generate_dataset_json/btad.py` |
| KSDD2 | 作者 Drive（`Metal/`，仅 test） | 1070 文件 / 830MB | 自定义（见下） |
| DTD-Synthetic | 作者 Drive | 3453 文件 / 356MB | `generate_dataset_json/DTD.py` |
| DAGM | 作者 Drive（`pic1/`） | 18234 文件 / 2.8GB | 自定义（见下） |

各脚本内 `root` 是硬编码的，需要显式传参：

```python
import sys; sys.path.insert(0, "generate_dataset_json")
from mvtec import MVTecSolver; MVTecSolver(root=".../datasets/mvtec").run()
```

### 两个需要自己写 meta 的数据集

**KSDD2**：作者镜像只有 `Metal/test/{good,anomaly}` + `Metal/ground_truth/anomaly`，
没有 train；且 `SDD.py` 里的类名是 `electrical commutators`，与目录名 `Metal` 不一致。
把 `cls_name` 写成 `electrical commutators`、`img_path` 写成 `Metal/test/...` 即可。

**DAGM**：`generate_dataset_json/DAGM.py` 有两个问题
1. 输出的键是 `Train` / `Test`（大写），而 `dataset.py` 读的是 `meta['train']` / `meta['test']` → KeyError；
2. 默认指向 `/remote-home/.../DAGM_KaggleUpload`。

按它的逻辑重写并输出小写键即可（掩码命名 `0002_label.PNG` 对应图像 `0002.PNG`）。

## 4. 评测命令

> ⚠️ `test.py` 会把所有图片/掩码常驻内存，全量跑 VisA(2162 张) 需要 21GB+ 内存，
> 实测会被 OOM killer 直接杀掉。本仓库的 `test.py` 已修复为仅在 `--enable_analysis`
> 时缓存原图，但更稳的做法是按类别分批评测（指标本身就是逐类统计的，Mean 行
> 是各类算术平均，分批与全量结果等价）。

```bash
# 单类/单数据集直接跑
python test.py --test_data_path ../datasets/mvtec --test_dataset mvtec \
    --checkpoint_path weight/train_on_visa/CLIP.pth \
    --save_path ../results/mvtec --device cuda:0

# 按类别分批评测（自动改写 meta.json 再还原）
python datasets/eval_per_class.py ../datasets/mvtec mvtec \
    weight/train_on_visa/CLIP.pth ../results/mvtec --python .venv/Scripts/python.exe
```

速度参考（bs=1，518×518）：推理 ~4.2 img/s，峰值显存 ~2GB；训练 bs=1 峰值 ~4.2GB
（论文的 bs=8 在 8GB 卡上会 OOM）。

## 5. 结果对照（官方权重）

### VisA（`weight/train_on_mvtec/CLIP.pth`）

| | Pixel (AUROC / F1 / AP / PRO) | Image (AUROC / F1 / AP) |
|---|---|---|
本仓库实测 | **95.8 / 34.6 / 28.4 / 90.9** | **84.7 / 82.4 / 87.6** |
论文 | 95.8 / 34.6 / 28.4 / 91.0 | 84.7 / 82.5 / 87.6 |

### MVTec-AD（`weight/train_on_visa/CLIP.pth`）

| | Pixel (AUROC / F1 / AP / PRO) | Image (AUROC / F1 / AP) |
|---|---|---|
本仓库实测 | **90.8 / 43.6 / 41.2 / 87.6** | **92.2 / 93.2 / 96.7** |
论文 | 90.8 / 43.9 / 41.2 / 87.5 | 92.2 / 93.2 / 96.7 |

### BTAD（`weight/train_on_visa/CLIP.pth`）

| | Pixel (AUROC / F1 / AP / PRO) | Image (AUROC / F1 / AP) |
|---|---|---|
本仓库实测 | **91.1 / 49.8 / 43.1 / 80.8** | **94.9 / 93.9 / 97.0** |
论文 | 91.1 / 49.8 / 43.1 / 80.4 | 94.9 / 93.9 / 97.0 |

### KSDD2（`weight/train_on_visa/CLIP.pth`）

| | Pixel (AUROC / F1 / AP / PRO) | Image (AUROC / F1 / AP) |
|---|---|---|
本仓库实测 | **98.5 / 60.9 / 62.1 / 98.5** | **98.0 / 93.9 / 98.3** |
论文 | 98.5 / 60.9 / 62.1 / 98.5 | 98.0 / 93.9 / 98.3 |

### DTD-Synthetic（`weight/train_on_visa/CLIP.pth`）

| | Pixel (AUROC / F1 / AP / PRO) | Image (AUROC / F1 / AP) |
|---|---|---|
本仓库实测 | **98.1 / 64.3 / 65.5 / 94.8** | **97.5 / 96.7 / 99.1** |
论文 | 98.1 / 64.3 / 65.5 / 94.8 | 97.5 / 96.6 / 99.1 |

### DAGM（`weight/train_on_visa/CLIP.pth`，10 个类别）

| | Pixel (AUROC / F1 / AP / PRO) | Image (AUROC / F1 / AP) |
|---|---|---|
本仓库实测 | **92.2 / 58.0 / 56.4 / 89.2** | **99.5 / 95.0 / 97.8** |
论文 | 92.2 / 57.9 / 56.4 / 89.3 | 99.5 / 95.0 / 97.8 |

> DAGM 的 `Class1/Class7/Class8` 在全部类别一起排队跑时偶发 `MemoryError`
> （`roc_auc_score` 内部 `label_binarize` 会额外申请 float64 副本，类别样本
> 数 1150 时约 191M 像素），单独重跑即可通过。最终均值即上表。

## 6. 复现时踩到的坑（均已修复，见 git log）

1. `train.py` 的 `torch.use_deterministic_algorithms(True, warn_only=False)` 会让
   `F.interpolate(mode='bilinear')` 的反向传播抛
   `RuntimeError: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation`
   —— 发布版代码在 CUDA 上**无法训练**，改为 `warn_only=True`。
2. `utils/analysis.py` 等 3 个文件共 22 处 `print` 含 emoji，Windows 控制台 GBK
   编码下抛 `UnicodeEncodeError`，且发生在推理跑完之后 → 入口脚本启动时切 UTF-8。
3. `test.py` 无条件把每张图 `image.detach().cpu()` 存进列表，VisA 需要 ~7GB 额外内存。
4. `test.py` 的 `obj_list` 来自硬编码类别表，与 meta.json 不一致时空类会导致
   `np.stack([])` 崩溃。
5. `utils/metrics.py` 把 `pixel_aupro = cal_pro_score(...)` **注释掉并硬写 0**，
   导致 PRO 恒为 0；恢复后 VisA 90.9 / MVTec 87.6 / BTAD 80.8 / KSDD2 98.5 /
   DTD 94.8，全部与论文一致。
6. `utils/metrics.py` 用 `list.extend(numpy.flatten())` 累积逐像素值，单类
   （DAGM 575 张）即 1.5 亿 Python float，触发 `MemoryError`；
   `cal_pro_score` 在 200 个阈值上重复 `regionprops`，慢 5~10 倍。
8. 大数据集单类（DAGM 1150 张）跑 `roc_auc_score` 仍可能 `MemoryError`：
   sklearn 内部会为 0/1 标签额外申请 float64 副本。按类别分批评测最稳。
7. `requirements.txt` 漏了 `opencv-python` / `matplotlib` / `pandas` / `ftfy` /
   `regex` / `tabulate` / `Pillow` / `setuptools`，且未约束 numpy 版本。
