from __future__ import annotations
import os
import shutil
import warnings
import numpy as np
import lightning as pl
from functools import partial
from dgl.data.utils import split_dataset
from mp_api.client import MPRester
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS

# Clear DGL cache
os.system('rm -r ~/.dgl')

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# Setup Materials Project API key
mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
entries = mpr.get_entries_in_chemsys(["Ba", "Zr", "O"])
structures = [e.structure for e in entries]
energies = [e.energy for e in entries]

# Modify the preparation of forces and stresses
forces = [np.zeros((len(e.structure), 3), dtype=np.float32) for e in entries]
stresses = [np.zeros((3, 3), dtype=np.float32) for _ in structures]

labels = {
    "energies": np.array(energies, dtype=np.float32),
    "forces": forces,
    "stresses": stresses
}

print(f"{len(structures)} downloaded from MP.")

element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

# Modify the way the dataset is created
dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels=labels,
    include_line_graph=True,
    save_cache=False  # Key modification: disable caching
)

train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)

my_collate_fn = partial(collate_fn_pes, include_line_graph=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=16,  # Increase batch size
    num_workers=0,
)

# Use the direct PES model
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")  # Note the use of DIRECT-PES here
model_pretrained = m3gnet_nnp.model

# Get element reference energies and data normalization parameters from the pretrained model
element_refs_m3gnet = m3gnet_nnp.element_refs.property_offset
data_std = m3gnet_nnp.data_std

lit_module = PotentialLightningModule(
    model=model_pretrained,
    lr=1e-3,
    include_line_graph=True,
    force_weight=1.0,
    stress_weight=0.1,
    element_refs=element_refs_m3gnet,
    data_std=data_std,
    decay_steps=100,
    decay_alpha=0.01
)

# Training setup
logger = CSVLogger("logs", name="M3GNet_training")
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the model
model_export_path = "./trained_model/"
lit_module.model.save(model_export_path)

m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model

# Get element energy offsets
property_offset = m3gnet_nnp.element_refs.property_offset

# Create a LightningModule for fine-tuning
# Note the use of the pretrained model and element reference energies
lit_module_finetune = PotentialLightningModule(
    model=model_pretrained,          # Use the pretrained model
    element_refs=property_offset,    # Use the original energy offsets
    lr=1e-4,                         # Smaller learning rate for fine-tuning
    include_line_graph=True,         # Include line graph
    stress_weight=0.01               # Stress weight can be adjusted as needed
)

# Setup the trainer for fine-tuning
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = pl.Trainer(
    max_epochs=1, 
    accelerator="cpu", 
    logger=logger, 
    inference_mode=False
)

# Perform fine-tuning
trainer.fit(
    model=lit_module_finetune, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader
)

# Save the fine-tuned model
model_save_path = "./finetuned_model/"
lit_module_finetune.model.save(model_save_path)

# Load the fine-tuned model
trained_model = matgl.load_model(path=model_save_path)

# Clean up temporary files
for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

# shutil.rmtree("logs")
# shutil.rmtree("trained_model")
# shutil.rmtree("finetuned_model")