from ase.io.trajectory import Trajectory

traj = Trajectory('300K.traj')
n_frames = len(traj)
timestep = 0.5  # fs
total_time_fs = n_frames * timestep
total_time_ps = total_time_fs / 1000

print(f"Total simulation time: {total_time_ps:.2f} ps")