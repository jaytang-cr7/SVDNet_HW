import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从solution.py导入所有需要的类
from solution import (
    SVDNet,
    DataPreprocessor,
    ChannelSVDDataset,
    SVDLoss,
    calculate_ae,
    test_model_performance
)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-5, device='cuda'):
    """训练SVD模型"""
    criterion = SVDLoss(alpha=2.0, beta=0.1, gamma=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    val_losses = []
    val_aes = []

    best_ae = float('inf')
    best_model_state = None

    print(f"开始训练，总轮数: {num_epochs}")
    print("-" * 60)

    early_stopping = EarlyStopping(patience=20)
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_count = 0

        for data, labels, cfg in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            data, labels = data.to(device).float(), labels.to(device).float()

            optimizer.zero_grad()

            # 前向传播
            U_pred, S_pred, V_pred = model(data.squeeze(0))

            # 计算损失
            loss, metrics = criterion(U_pred, S_pred, V_pred, labels.squeeze(0))

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch += loss.item()
            train_count += 1

        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_ae_epoch = 0
        val_count = 0

        with torch.no_grad():
            for data, labels, cfg in val_loader:
                data, labels = data.to(device).float(), labels.to(device).float()

                U_pred, S_pred, V_pred = model(data.squeeze(0))
                loss, _ = criterion(U_pred, S_pred, V_pred, labels.squeeze(0))
                ae = calculate_ae(U_pred, S_pred, V_pred, labels.squeeze(0))

                val_loss_epoch += loss.item()
                val_ae_epoch += ae
                val_count += 1

        # 更新学习率
        scheduler.step()

        # 记录指标
        avg_train_loss = train_loss_epoch / train_count
        avg_val_loss = val_loss_epoch / val_count
        avg_val_ae = val_ae_epoch / val_count

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_aes.append(avg_val_ae)

        # 保存最佳模型
        if avg_val_ae < best_ae:
            best_ae = avg_val_ae
            best_model_state = model.state_dict().copy()

        # 打印进度
        # if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
        print(f'Epoch {epoch + 1:3d}/{num_epochs}: '
              f'Loss={avg_train_loss:.4f}, '
              f'AE={avg_val_ae:.6f}, '
              f'Best={best_ae:.6f}, '
              f'LR={scheduler.get_last_lr()[0]:.6f}')

        if early_stopping(avg_train_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break


    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("-" * 60)
    print(f"训练完成! 最佳AE: {best_ae:.6f}")

    return train_losses, val_losses, val_aes, best_ae


def main():
    parser = argparse.ArgumentParser(description='训练轻量化SVD神经网络')
    parser.add_argument('--data_dir', type=str, default='./CompetitionData1', help='数据目录')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save_path', type=str, default='svd_model.pth', help='模型保存路径')

    args = parser.parse_args()

    # 设备检查
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据路径
    data_files = [
        os.path.join(args.data_dir, 'Round1TrainData1.npy'),
        os.path.join(args.data_dir, 'Round1TrainData2.npy'),
        os.path.join(args.data_dir, 'Round1TrainData3.npy')
    ]

    label_files = [
        os.path.join(args.data_dir, 'Round1TrainLabel1.npy'),
        os.path.join(args.data_dir, 'Round1TrainLabel2.npy'),
        os.path.join(args.data_dir, 'Round1TrainLabel3.npy')
    ]

    cfg_files = [
        os.path.join(args.data_dir, 'Round1CfgData1.txt'),
        os.path.join(args.data_dir, 'Round1CfgData2.txt'),
        os.path.join(args.data_dir, 'Round1CfgData3.txt')
    ]

    # 检查文件是否存在
    all_files = data_files + label_files + cfg_files
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        print("缺少以下文件:")
        for f in missing_files:
            print(f"  {f}")
        return

    # 数据预处理
    transform = DataPreprocessor(noise_level=0.1, enable_augmentation=True)

    # 创建数据集
    print("加载数据...")
    dataset = ChannelSVDDataset(data_files, label_files, cfg_files, transform=transform)

    # 划分训练和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 创建模型
    print("创建模型...")
    model = SVDNet(dim=64, rank=64, hidden_dim=64).to(device)

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    print(f'训练样本: {len(train_dataset)}')
    print(f'验证样本: {len(val_dataset)}')
    print(f'满足参数约束: {total_params < 10_000_000}')

    # 训练模型
    train_losses, val_losses, val_aes, best_ae = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )

    # 测试最终模型性能
    print("\n测试模型性能...")
    performance = test_model_performance(model, device)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_ae': best_ae,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aes': val_aes,
        'performance': performance,
        'config': {
            'dim': 64,
            'rank': 32,
            'hidden_dim': 48,
            'epochs': args.epochs,
            'learning_rate': args.lr
        }
    }, args.save_path)

    print(f'\n模型已保存到: {args.save_path}')
    print(f'最佳AE: {best_ae:.6f}')
    print(f'约束满足情况:')
    print(f'  参数量<10M: {performance["parameters"] < 10_000_000} ({performance["parameters"]:,})')
    print(f'  推理时间<5ms: {performance["inference_time_ms"] < 5.0} ({performance["inference_time_ms"]:.2f}ms)')
    print(f'  AE<0.05: {best_ae < 0.05} ({best_ae:.6f})')

    # 绘制训练曲线
    try:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='训练损失', alpha=0.7)
        plt.plot(val_losses, label='验证损失', alpha=0.7)
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练和验证损失')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(val_aes, color='red', alpha=0.7)
        plt.xlabel('轮数')
        plt.ylabel('验证AE')
        plt.title('验证近似误差')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.semilogy(train_losses, label='训练损失', alpha=0.7)
        plt.semilogy(val_losses, label='验证损失', alpha=0.7)
        plt.xlabel('轮数')
        plt.ylabel('损失 (对数)')
        plt.legend()
        plt.title('训练和验证损失 (对数尺度)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print(f'训练曲线已保存到: training_curves.png')
    except Exception as e:
        print(f'绘制训练曲线时出错: {e}')


if __name__ == '__main__':
    main()