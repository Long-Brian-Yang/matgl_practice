# analysis_script.py
from diffusion_analysis import run_all_analysis
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Run analysis
run_all_analysis(
    trajectories=[
        './pretraining_md_results/T_1000K/md_Y_BaZrO3_MC_1000K.traj'
        # 'results/T_600K/md_600K.traj',
        # 'results/T_900K/md_900K.traj'
    ],
    temperatures=[1000],
    proton_index=1,  # Index of proton in the structure
    timestep=0.5,
    output_dir=Path('analysis_results'),
    logger=logger
)
