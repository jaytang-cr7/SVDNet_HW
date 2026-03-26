import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import json
from datetime import datetime


# ================ 核心模型定义 ================
class CompressedLinear(nn.Module):
    """压缩线性层 - 低秩分解减少参数"""

    def __init__(self, in_features, out_features, rank_ratio=0.5):
        super().__init__()
        rank = max(1, int(min(in_features, out_features) * rank_ratio))
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features)

    def forward(self, x):
        return self.linear2(self.linear1(x))


# 原始参数量：8192 × 96 = 786,432
# 替代参数量：8192 × 24 + 24 × 96 = 196,608 + 2,304 = 198,912
# 节省比例：约 74.7%

class SVDNet(nn.Module):
    """轻量化SVD网络 - 328K参数"""

    def __init__(self, dim=64, rank=64, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.input_dim = dim * dim * 2  # 8192

        # 极简编码器
        self.encoder = nn.Sequential(
            CompressedLinear(self.input_dim, hidden_dim * 2, rank_ratio=0.25),  # 8192->96
            nn.ReLU(),
            nn.Dropout(0.1),
            CompressedLinear(hidden_dim * 2, hidden_dim, rank_ratio=0.5),  # 96->48
            nn.ReLU()
        )

        # 直接输出
        self.u_out = CompressedLinear(hidden_dim, dim * rank * 2, rank_ratio=0.3)
        self.v_out = CompressedLinear(hidden_dim, dim * rank * 2, rank_ratio=0.3)
        self.s_out = nn.Sequential(
            nn.Linear(hidden_dim, rank),
            nn.Softplus()
        )

        # 尝试加载训练好的权重
        self._try_load_weights()

    def _try_load_weights(self):
        """尝试加载已训练的模型权重"""
        try:
            if os.path.exists('svd_model.pth'):
                checkpoint = torch.load('svd_model.pth', map_location='cpu')
                self.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 已加载训练好的模型权重")
        except Exception as e:
            print(f"⚠️ 使用随机初始化权重: {e}")


    # 训练用的forward函数
    # def forward(self, x):
    #     batch_size = 1
    #     x_flat = x.view(batch_size, -1).float()  # 确保float32类型
    #
    #     features = self.encoder(x_flat)  # [1, 48]
    #
    #     U_pred = self.u_out(features).view(self.dim, self.rank, 2)
    #     V_pred = self.v_out(features).view(self.dim, self.rank, 2)
    #     S_pred = self.s_out(features).view(self.rank)
    #
    #     # 归一化
    #     U_norm = self._normalize(U_pred)
    #     V_norm = self._normalize(V_pred)
    #
    #     return U_norm, S_pred, V_norm

    # 测试用的forward函数
    # 测试用的forward函数
    def forward(self, x):
        # 兼容 numpy 和 torch.Tensor 输入
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # 如果输入是 3D，则视为单样本 [H, W, 2]，增加 batch 维度
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1, H, W, 2]

        batch_size = x.shape[0]

        # ---------------- 归一化处理开始 ----------------
        real = x[..., 0]
        imag = x[..., 1]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)

        # 仅在空间维度 (H, W) 上做均值和标准差
        mean_mag = magnitude.mean(dim=(1, 2), keepdim=True)
        std_mag = magnitude.std(dim=(1, 2), keepdim=True) + 1e-20

        real_norm = (real - mean_mag) / std_mag
        imag_norm = (imag - mean_mag) / std_mag

        x = torch.stack([real_norm, imag_norm], dim=-1)  # [B, H, W, 2]
        # ---------------- 归一化处理结束 ----------------

        x_flat = x.view(batch_size, -1).float()  # 展平为 [B, 8192]
        features = self.encoder(x_flat)  # [B, 48]

        U_pred = self.u_out(features).view(batch_size, self.dim, self.rank, 2)
        V_pred = self.v_out(features).view(batch_size, self.dim, self.rank, 2)
        S_pred = self.s_out(features).view(batch_size, self.rank)

        # 归一化输出向量
        U_norm = self._normalize(U_pred[0])  # 去除 batch 维度 [64, 32, 2]
        V_norm = self._normalize(V_pred[0])

        # # 还原：
        # 1. 实部和虚部拼成复数矩阵
        U_c = torch.complex(U_norm[..., 0], U_norm[..., 1])  # [64, 32]
        V_c = torch.complex(V_norm[..., 0], V_norm[..., 1])  # [64, 32]
        # 2. V^H
        V_H = V_c.conj().T  # [32, 64]
        # 3. 构造对角矩阵 S
        S = S_pred[0]  # [32]
        S_diag = torch.diag(S).to(dtype=U_c.dtype)  # [32, 32]
        # 4. 计算归一化下的 H_pred
        H_pred_norm = U_c @ S_diag @ V_H  # [64, 64]
        # 5. 反归一化
        H_pred_real = H_pred_norm.real * std_mag + mean_mag
        H_pred_imag = H_pred_norm.imag * std_mag + mean_mag
        H_pred = torch.complex(H_pred_real, H_pred_imag)  # [64, 64]
        # 假设 H_pred 是 [1, 64, 64]，复数类型
        H_pred = H_pred.squeeze(0)  # -> [64, 64]
        u_inv = torch.linalg.inv(U_c)
        v_conj_inv = torch.linalg.inv(V_H)
        S_duijiao = u_inv @ H_pred @ v_conj_inv
        S_real = torch.diagonal(S_duijiao).real  # [32]
        # S_real, _ = torch.topk(S_real, k=32, largest=True, sorted=True)  # shape: [32]
        S_real = S_real[:32]  # shape: [32]
        U = U_c[:, :32]  # [64, 32]
        V = V_c[:, :32]  # [64, 32]
        U = torch.stack((U.real, U.imag), dim=-1)  # [64, 32, 2]
        V = torch.stack((V.real, V.imag), dim=-1)  # [64, 32, 2]

        # 对输出USV做列正交（非理想还原）
        # # 构造 64×64 的单位矩阵，并扩展为复数类型
        # I = torch.eye(64, dtype=H_pred.dtype)  # [64, 64]
        # # 截取前 32 列构成 U
        # U = I[:, :32]  # [64, 32]
        # V = U
        # print(U.dtype, H_pred.dtype, V.dtype)
        #
        # S_duijiao = U.conj().T @ H_pred @ V  # [32, 32] 近似为 diag(S)
        # S_real = torch.diagonal(S_duijiao).real  # [32]
        # print(S_real.shape)
        # U = torch.stack((U.real, U.imag), dim=-1)  # [64, 32, 2]
        # V = torch.stack((V.real, V.imag), dim=-1)  # [64, 32, 2]

        # ！！！！！！！
        # I = torch.eye(64, dtype=H_pred.dtype)  # shape: [64, 64]
        # S_duijiao = I.conj().T @ H_pred @ I  # [32, 32] diag(S)
        # S_real = torch.diagonal(S_duijiao).real  # [64]
        # S_real, _ = torch.topk(S_real, k=32, largest=True, sorted=True)  # shape: [32]
        # # S_real = S_real[:32]  # shape: [32]
        # U = I[:, :32]  # [64, 32]
        # V = U
        # U = torch.stack((U.real, U.imag), dim=-1)  # [64, 32, 2]
        # V = torch.stack((V.real, V.imag), dim=-1)  # [64, 32, 2]

        # print(S_real.shape)

        ####  对输出USV再做SVD分解得到最优
        # # H_pred: torch.complex64, [64, 64]
        # U_orth, S_new, Vh_orth = torch.linalg.svd(H_pred, full_matrices=False)  # all are complex64
        # # 只保留前 k=32 项（与之前的U_norm匹配）
        # k = 32
        # U_orth = U_orth[:, :k]  # [64, 32]
        # S_new = S_new[:k]  # [32]
        # V_orth = Vh_orth.conj().T[:, :k]  # [64, 32]
        # # 拆为实虚 [64, 32, 2]
        # U_real = torch.stack([U_orth.real, U_orth.imag], dim=-1)  # [64, 32, 2]
        # V_real = torch.stack([V_orth.real, V_orth.imag], dim=-1)  # [64, 32, 2]
        # S_real = S_new  # [32]
        # print(f'U_real: {U.shape}')
        # print(f'V_real: {V.shape}')

        return U, S_real, V
        # return U_real, S_real, V_real

    def _normalize(self, mat):
        real, imag = mat[..., 0], mat[..., 1]
        complex_mat = real + 1j * imag
        norms = torch.norm(complex_mat, dim=0, keepdim=True) + 1e-20
        normalized = complex_mat / norms
        return torch.stack([normalized.real, normalized.imag], dim=-1)


# ================ 数据处理 ================
class DataPreprocessor:
    def __init__(self, noise_level=0.1, enable_augmentation=True):
        # self.noise_level = noise_level
        self.enable_augmentation = enable_augmentation

    def __call__(self, data):
        # data = self.denoise_complex(data)
        if self.enable_augmentation:
            # data = self.augment_data(data)
            data = self.normalize_complex(data)
        return data



    def normalize_complex(self, data):
        """对复数数据进行归一化处理"""
        # 提取实部和虚部
        real = data[..., 0]
        imag = data[..., 1]

        # 计算幅度 (Magnitude)
        magnitude = np.sqrt(real ** 2 + imag ** 2)

        # 计算幅度的均值和标准差
        mean_mag = np.mean(magnitude, axis=(1, 2), keepdims=True)  #
        std_mag = np.std(magnitude, axis=(1, 2), keepdims=True) + 1e-20  # 防止除以零

        # 归一化实部和虚部
        real_normalized = (real - mean_mag) / std_mag
        imag_normalized = (imag - mean_mag) / std_mag

        # 返回归一化后的复数数据
        return np.stack([real_normalized, imag_normalized], axis=-1)



class ChannelSVDDataset(Dataset):
    def __init__(self, data_files, label_files, cfg_files, transform=None):
        self.data = []
        self.labels = []
        self.cfgs = []

        for data_file, label_file, cfg_file in zip(data_files, label_files, cfg_files):
            with open(cfg_file, 'r') as f:
                lines = f.readlines()
                cfg = {
                    'samp_num': int(lines[0].strip()),
                    'M': int(lines[1].strip()),
                    'N': int(lines[2].strip()),
                    'IQ': int(lines[3].strip()),
                    'R': int(lines[4].strip())
                }

            data = np.load(data_file).astype(np.float32)
            labels = np.load(label_file).astype(np.float32)

            if transform:
                data = transform(data)
                labels = transform(labels)

            for i in range(data.shape[0]):
                self.data.append(data[i])
                self.labels.append(labels[i])
                self.cfgs.append(cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx]), self.cfgs[idx]


# ================ 损失函数 ================
class SVDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, U_pred, S_pred, V_pred, H_label):
        U_complex = U_pred[..., 0] + 1j * U_pred[..., 1]  # [64, 32]
        V_complex = V_pred[..., 0] + 1j * V_pred[..., 1]  # [64, 32]
        H_complex = H_label[..., 0] + 1j * H_label[..., 1]  # [64, 64]

        S_diag = torch.diag(S_pred).to(U_complex.dtype)  # [32, 32]
        H_recon = torch.matmul(torch.matmul(U_complex, S_diag), V_complex.conj().T)  # [64, 64]

        # 重构损失
        recon_loss = torch.norm(H_complex - H_recon, p='fro') / torch.norm(H_complex, p='fro')

        # 正交性损失
        U_orth_loss = torch.norm(
            torch.matmul(U_complex.conj().T, U_complex) - torch.eye(64, device=U_complex.device, dtype=U_complex.dtype),
            p='fro')
        V_orth_loss = torch.norm(
            torch.matmul(V_complex.conj().T, V_complex) - torch.eye(64, device=V_complex.device, dtype=V_complex.dtype),
            p='fro')

        total_loss = self.alpha * recon_loss + self.beta * U_orth_loss + self.gamma * V_orth_loss

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'u_orth_loss': U_orth_loss.item(),
            'v_orth_loss': V_orth_loss.item(),
            'total_loss': total_loss.item()
        }


def calculate_ae(U_pred, S_pred, V_pred, H_label):
    """计算近似误差AE"""
    U_complex = U_pred[..., 0] + 1j * U_pred[..., 1]
    V_complex = V_pred[..., 0] + 1j * V_pred[..., 1]
    H_complex = H_label[..., 0] + 1j * H_label[..., 1]

    S_diag = torch.diag(S_pred).to(U_complex.dtype)
    H_recon = torch.matmul(torch.matmul(U_complex, S_diag), V_complex.conj().T)

    recon_error = torch.norm(H_complex - H_recon, p='fro') / torch.norm(H_complex, p='fro')

    I = torch.eye(64, device=U_complex.device, dtype=U_complex.dtype)
    u_orth_error = torch.norm(torch.matmul(U_complex.conj().T, U_complex) - I, p='fro')
    v_orth_error = torch.norm(torch.matmul(V_complex.conj().T, V_complex) - I, p='fro')

    ae = recon_error + u_orth_error + v_orth_error
    return ae.item()


# ================ 性能测试工具 ================
def test_model_performance(model, device='cuda'):
    """测试模型推理性能"""
    model.eval()
    test_input = torch.randn(64, 64, 2).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)

    # 测速
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(test_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

    avg_inference_time = sum(times) / len(times)

    print(f"模型性能指标:")
    print(f"  参数量: {total_params:,}")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    print(f"  推理时间: {avg_inference_time:.2f} ms")
    print(f"  满足约束: 参数<10M({total_params < 10_000_000}), 推理<5ms({avg_inference_time < 5.0})")

    return {
        'parameters': total_params,
        'size_mb': model_size_mb,
        'inference_time_ms': avg_inference_time,
        'satisfies_constraints': total_params < 10_000_000 and avg_inference_time < 5.0
    }