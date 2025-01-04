# from ase.io.trajectory import Trajectory

# traj = Trajectory("1000K.traj", "r")
# for i, frame_atoms in enumerate(traj):
#     print("Frame:", i, "Time:", frame_atoms.info["time"])
# traj.close()

from ase.io import read, write

# Read the .traj file and convert it to .xyz
atoms = read('1000K.traj', index=':')
write('1000K.xyz', atoms)



