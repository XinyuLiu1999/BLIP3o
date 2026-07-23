

import re
import matplotlib.pyplot as plt
import pandas as pd  # 建议使用 pandas 处理滑动平均

# 1. 提取数据
pattern = r"Diffusion loss\s+([\d\.]+)"
with open('/backup/user/liuxinyu/BLIP3o-Pretrain-results/Pretrain-04021034/logs/train_rank0.log', 'r', encoding='utf-8') as f:
    losses = [float(x) for x in re.findall(pattern, f.read())]

# 2. 计算滑动平均 (窗口大小设为 10，可根据数据量调整)
window_size = 100
smooth_losses = pd.Series(losses).rolling(window=window_size).mean()

# 3. 绘图
plt.figure(figsize=(10, 5))

# 绘制原始曲线（浅色）
plt.plot(losses, alpha=0.3, color='blue', label='Original Loss')
# 绘制滑动平均线（深色/加粗）
plt.plot(smooth_losses, color='red', linewidth=2, label=f'Moving Average (w={window_size})')

plt.title('Diffusion Loss with Moving Average')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 4. 保存
plt.savefig('diffusion_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()