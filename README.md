AI-enabled Robust SVD Operator for Wireless Communications

2025 华为杯 · 无线通信算法大赛（第四届）参赛项目

本仓库包含本团队为2025年第四届“华为杯”无线通信算法大赛（AI使能的无线鲁棒SVD算子）所开发的完整代码，包括：

✔️ 轻量化深度学习 SVD 网络 SVDNet
✔️ 核心算法：数据压缩、SVD 分解、鲁棒损失函数
✔️ 完整训练流程 & Early Stopping
✔️ 提交格式的推理代码（适用于比赛输入/输出）

本仓库结构简洁易用，可直接用于比赛训练与测试。

├── solution.py       # ⭐ 核心：SVDNet 模型+数据集+损失函数+评估函数
├── train.py          # ⭐ 完整训练流程：dataset → dataloader → train → val → early stop
├── demo_code.py      # ⭐ 官方格式推理代码：自动读取配置、加载模型并生成提交文件
├── README.md         # 当前文件

1. 核心模型：SVDNet（solution.py）
文件：solution.py
包含本项目所有算法核心模块：
✨ SVDNet 亮点
仅 328K 参数 的轻量化网络
使用 CompressedLinear（低秩分解线性层） 减少 70% 参数
输入：复矩阵（M×N×IQ）展开的 8192 维向量
输出：
U（左奇异矩阵）
S（奇异值，Softplus 保证非负）
V（右奇异矩阵）
📌 文件内容包括：
★CompressedLinear
低秩分解线性层，两层 small-rank Linear 替代大矩阵，减少约 74.7% 参数量。
★SVDNet（主要网络）
极简 encoder（8192 → 96 → 48）
三个 decoder：输出 U/S/V
自动加载训练过的权重（若存在）
★ChannelSVDDataset
将原始比赛数据预处理成可训练样本。
★SVDLoss
综合性鲁棒损失函数：
奇异值损失
正交性约束
重构误差
★calculate_ae, test_model_performance
用于验证与比赛指标对齐的 AE（Angle Error）计算。

2. 训练脚本：train.py
文件：train.py
包含完整训练流程：
✔ 数据加载与 DataLoader
通过 ChannelSVDDataset 自动分 batch 训练。
✔ 训练循环
AdamW + CosineAnnealing LR scheduler
自定义 EarlyStopping（patience=20）
每轮记录 loss & AE 指标
✔ 验证与 Early Stop
当模型连续 20 epoch 无提升时自动停止。
✔ 模型保存
记录最佳 AE 的模型，并保存权重。

3. 推理与提交格式：demo_code.py
文件：demo_code.py
用于比赛实际提交，结构紧贴官方要求。
✨ 功能说明
自动读取官方 CfgDataX.txt 配置文件
加载输入信道数据
调用 SVDNet 自动输出 U/S/V
写入比赛要求格式的结果文件




