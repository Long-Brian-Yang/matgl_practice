from ase.io.trajectory import Trajectory

traj_pre = Trajectory('pre300K.traj')
print(f"Pretraining frames: {len(traj_pre)}")

traj_fine = Trajectory('fine300K.traj')
print(f"Finetuning frames: {len(traj_fine)}")