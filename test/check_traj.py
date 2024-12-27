from ase.io.trajectory import Trajectory

# 检查pretraining的轨迹
traj_pre = Trajectory('pre300K.traj')
print(f"Pretraining frames: {len(traj_pre)}")

# 检查finetuning的轨迹
traj_fine = Trajectory('fine300K.traj')
print(f"Finetuning frames: {len(traj_fine)}")