# ATS-ViT · Phase 1 侦察记录与 ATS-SAF 设计说明

> 本文记录：把 `AnisotropicTextureSuppressionGate`（ATSGate）接入 VisualAD 的**精确定位**、
> 数学实现、以及"不影响原版行为"的验证证据。所有行号对应仓库当前 `ats-vit` 分支。

## 1. Phase 1：三个核心接口的位置（已核实）

| 环节 | 位置 | 说明 |
|---|---|---|
**冻结 ViT 的多层 Patch 特征** | `VisualAD_lib/VisualAD.py:59` `Transformer.forward(x, out_layers)` → `:66` `if idx in out_layers: out_tokens.append(x.clone())`；`:71` `forward_dispatch`；`:164` `'patch_start_idx': 3` | 在层 `{6,12,18,24}` 各缓存一份 **完整 token 序列**（前 3 个是 anomaly/normal/class 特殊 token） |
**SCA 更新原型 token** | `utils/spatial_cross_attention.py:148` `class LayerAdaptiveCrossAttention`，`:177` `forward(anomaly_token, normal_token, patch_tokens_list, layer_ids)`，内部 `:95` `SpatialBottleneckAttention.forward` | 逐层独立 SCA；`attn = softmax(Q_a·Kᵀ·scale)`（`:121-123`），已缓存 `last_attn` 供可视化 |
**SAF 校准 patch 特征** | `utils/feature_transform.py:36` `FeatureTransformMLP`（`Linear(C,C) → GELU → Dropout → Linear(C,C) → Dropout`）；应用点：`train.py:174` `setup_feature_transforms(...)`，在 `train.py`/`test.py` 里对每层 `patch_feature` 做前向 | 官方权重里对应 `layer_transforms.layer_{6,12,18,24}.mlp.{0,3}` |
**异常图与打分** | `utils/anomaly_detection.py:9` `generate_anomaly_map_from_tokens`，`:32` `anomaly_score = anomaly_sim - normal_sim`；`utils/scoring.py` top-k 1% | 4 层图求和 → 上采样到 518 → 高斯 σ=4 → top-1% 均值作图像级分数 |

关键结论：**ATS 的插入点是 SAF，而不是 SCA**。SCA 已经在更新 `t_a/t_n`，而 SAF 负责 patch 侧的
非线性重标定 —— 频域高频证据正好属于 patch 侧，因此把 ATSGate 作为 SAF 的**并联高频残差分支**。

## 2. Phase 2：ATS-SAF 实现

### 2.1 数学结构

对第 $l$ 层的 patch token 特征 $P_l \in \mathbb{R}^{B\times N\times C}$（$N=37\times37=1369$，$C=1024$）：

1. **2D 正交频域去基底 + 各向异性抑制**
   $$\tilde P = \mathrm{reshape}(P_l)\in\mathbb{R}^{B\times C\times H_p\times W_p},\quad
   \mathcal{F}=\mathrm{rfft2}(\tilde P,\ \text{norm=ortho})$$
   $$\mathcal{F}'(0,0)\leftarrow 0 \quad(\text{去金属镜面高光直流分量}),\qquad
   \mathcal{F}'' = \mathcal{F}'\odot \mathbf{S}_l$$
   其中 $\mathbf{S}_l\in\mathbb{R}^{H_p\times (W_p/2+1)}$ 是**可学习的各向异性频域权重**（初值为径向高通倾向
   $0.5+0.5\,r/r_{\max}$，因此起点保留部分低频、抑制最平的基底）。逆变换得到高频边缘特征
   $$P_{hf}=\mathrm{irfft2}(\mathcal{F}'',\ H_p\times W_p)$$
   > 若 patch 数不是完全平方数，自动回退 **1D rfft**（沿 token 序列方向），保证不破坏空间假设。
2. **通道级缺陷边缘激励门控**
   $$w_l=\sigma\!\Big(W_2\,\mathrm{GELU}(W_1\,[\mathrm{GAP}(P_{hf});\mathrm{GMP}(P_{hf})])\Big)\in\mathbb{R}^{B\times C}$$
   $W_1\in\mathbb{R}^{C/16\times 2C},\ W_2\in\mathbb{R}^{C\times C/16}$，$W_2$ 零初始化 ⇒ 起点 $w_l\equiv0.5$。
3. **残差自对齐融合**
   $$\hat P=\mathrm{normalize}\Big(\mathrm{normalize}(\mathrm{SAF}(P_l))+\alpha\cdot\mathrm{normalize}(P_{hf}\odot w_l)\Big),\qquad \alpha_{\text{init}}=0.05$$
   双分支各自 L2 归一化后再融合，使 $\alpha$ 直接对应"余弦空间中的旋转量"，量纲稳定。

### 2.2 "起点必须继承原版 SOTA" 的实现与验证

| 措施 | 效果 |
|---|---|
门控末层权重零初始化 + 偏置零 | $w_l = 0.5$，高频分支幅度受控 |
$\alpha$ 初始化为 0.05，且只有它单独一组学习率（`1e-2 × lr`）并在 `validate_gradients` 里裁剪 | 不会首步跳飞（实测未裁剪时 $\partial L/\partial\alpha\sim10^2$） |
双分支 L2 归一化后融合 | 起点与纯 SAF 的方向偏差极小 |
`--use_ats` 默认关闭；关闭时 `patch_feature` 仍走原 `view(-1,C)` 路径 | **与原版逐位一致** |

实测证据（脚本 `scripts/tests/test_ats_invariants.py`）：

```
2D 拍平路径 == 纯 MLP                : True        # 完全一致
3D 输出 L2 范数                      : 1.0
alpha / gate 末层                    : 0.05 / 全零初始化
与纯 SAF 输出平均余弦（起点偏差）      : 0.998753   # 约 2.9° 旋转
非平方 patch 数                      : 断言拦截，不会静默出错
alpha & spectrum 梯度                 : 均可回传
```

端到端回归（`_smoke` 合成集，6 张图）：

```
baseline  : 43.4 / 11.7 / 5.1 / 8.2  |  33.3 / 75.0 / 47.8
+ATS 初始化: 43.2 / 11.7 / 5.1 / 8.1  |  33.3 / 75.0 / 47.8   ← 起点等价
```

### 2.3 开关与兼容性

```bash
# 原版（完全不变）
python train.py --train_data_path ... --device cuda:0
# ATS-ViT
python train.py ... --use_ats --pretrained_path weight/train_on_visa/CLIP.pth
# 消融 (e)：去掉频域抑制，只留通道门控
python train.py ... --use_ats --ats_no_spectral
# 推理：test.py 会从 ckpt 的 uses_ats/ats_config 自动恢复，无需手动指定
python test.py ... --checkpoint_path results_ats_vit/ckpt/finetune_ats.pth
```

ckpt 新增字段：`uses_ats`、`ats_config{use_spectral, alpha_init, reduction}`、
`transform_type="ats_mlp"`；`layer_transforms` 内新增 `ats.gate.*`、`ats.spectrum`、`ats.alpha`，
加载官方权重时统一 `strict=False`（已验证 4 层全部命中、unexpected keys = 0）。

## 3. Phase 3/4 实验协议（如实记录）

| 项 | 设置 |
|---|---|
源域训练 | 官方权重 `weight/train_on_visa/CLIP.pth` 热启动，**baseline 与 ATS 完全相同的预算**：VisA 6 类 × 1 epoch（1010 张，bs=1，lr=1e-3） |
公平性 | 两侧唯一差异是 `--use_ats`；数据、步数、随机种子、超参完全相同 |
评测协议 | 论文零样本跨数据集：源域 VisA → 测 MVTec-AD / KSDD2 / DAGM / BTAD / DTD-Synthetic；另用官方 `train_on_mvtec` 权重给出 VisA 参考 |
指标 | Image: AUROC / F1-max / AP；Pixel: AUROC / F1-max / AP / PRO（PRO 由 `cal_pro_score` 恢复计算） |
筛选轮 | `SKIP_PRO=1`（约 5× 加速），随后对 KSDD2 / BTAD / MVTec 补算 PRO |

**已知局限（不修饰）**：
1. 论文原配方是 VisA 全 12 类 1 epoch；本次为控制单夜 GPU 预算改用 6 类子集，且两侧同设置，
   因此**相对比较有效，但与论文绝对值不完全可比**。
2. 消融 (a)(b)(c)(e) 需各自重训一个变体，本轮预算内未跑，表格中标记 `pending`，未做任何数值填充。
3. 单一随机种子；无多次重复的均值/方差。

## 4. 与论文的差异清单（已核对原文）

| 论文 | 本实现 |
|---|---|
只训练 `t_a, t_n, SCA, {F}` | 额外训练 `ln_post.weight/bias`（官方发布权重即如此，`test.py` 强制要求） |
Adam | AdamW（官方代码如此），分组 lr：tokens 1e-3 / 变换与 SCA 1e-4 / ATS-alpha 1e-5 |
未提高斯后处理 | `test.py` 有 `gaussian_filter(sigma=4)` |
未提 DINOv2 实现细节 | 官方代码无 DINOv2 入口，本工作同样只做 CLIP ViT-L/14@336px |
