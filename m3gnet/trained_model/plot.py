import pandas as pd
import matplotlib.pyplot as plt

# read the metrics.csv file
df = pd.read_csv('./logs/M3GNet_training/version_5/metrics.csv')

# Create separate dataframes for training and validation metrics
train_df = df[df['train_Total_Loss'].notna()].copy()
val_df = df[df['val_Total_Loss'].notna()].copy()

# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
fig.tight_layout(pad=4.0)  # adjust the space between subplots

# 1. total loss figure
ax1.plot(train_df['epoch'], train_df['train_Total_Loss'], 'b-', label='Training Loss')
ax1.plot(val_df['epoch'], val_df['val_Total_Loss'], 'r-', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Total Loss vs. Epoch')
ax1.grid(True)
ax1.legend()

# 2. energy MAE figure
ax2.plot(train_df['epoch'], train_df['train_Energy_MAE'], 'b-', label='Train Energy MAE')
ax2.plot(val_df['epoch'], val_df['val_Energy_MAE'], 'r-', label='Val Energy MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Energy MAE')
ax2.set_title('Energy MAE vs. Epoch')
ax2.grid(True)
ax2.legend()

# 3. force MAE figure
ax3.plot(train_df['epoch'], train_df['train_Force_MAE'], 'b-', label='Train Force MAE')
ax3.plot(val_df['epoch'], val_df['val_Force_MAE'], 'r-', label='Val Force MAE')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Force MAE')
ax3.set_title('Force MAE vs. Epoch')
ax3.grid(True)
ax3.legend()

# save the figure
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# if you want to save each plot separately, you can do the following:

# 1. total loss figure
plt.figure(figsize=(10, 6))
plt.plot(train_df['epoch'], train_df['train_Total_Loss'], 'b-', label='Training Loss')
plt.plot(val_df['epoch'], val_df['val_Total_Loss'], 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss vs. Epoch')
plt.grid(True)
plt.legend()
plt.savefig('total_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. energy MAE figure
plt.figure(figsize=(10, 6))
plt.plot(train_df['epoch'], train_df['train_Energy_MAE'], 'b-', label='Train Energy MAE')
plt.plot(val_df['epoch'], val_df['val_Energy_MAE'], 'r-', label='Val Energy MAE')
plt.xlabel('Epoch')
plt.ylabel('Energy MAE')
plt.title('Energy MAE vs. Epoch')
plt.grid(True)
plt.legend()
plt.savefig('energy_mae.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. force MAE figure
plt.figure(figsize=(10, 6))
plt.plot(train_df['epoch'], train_df['train_Force_MAE'], 'b-', label='Train Force MAE')
plt.plot(val_df['epoch'], val_df['val_Force_MAE'], 'r-', label='Val Force MAE')
plt.xlabel('Epoch')
plt.ylabel('Force MAE')
plt.title('Force MAE vs. Epoch')
plt.grid(True)
plt.legend()
plt.savefig('force_mae.png', dpi=300, bbox_inches='tight')
plt.close()
