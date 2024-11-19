from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
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

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# Obtain your API key here: https://next-gen.materialsproject.org/api
mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
entries = mpr.get_entries_in_chemsys(["Si", "O"])
structures = [e.structure for e in entries]
energies = [e.energy for e in entries]
forces = [np.zeros((len(s), 3)).tolist() for s in structures]
stresses = [np.zeros((3, 3)).tolist() for s in structures]
labels = {
    "energies": energies,
    "forces": forces,
    "stresses": stresses,
}

print(f"{len(structures)} downloaded from MP.")

element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = MGLDataset(
    threebody_cutoff=4.0, structures=structures, converter=converter, labels=labels, include_line_graph=True
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
    batch_size=2,
    num_workers=0,
)
model = M3GNet(
    element_types=element_types,
    is_intensive=False,
)
lit_module = PotentialLightningModule(model=model, include_line_graph=True)

# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_training")
# Inference mode = False is required for calculating forces, stress in test mode and prediction mode
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# test the model, remember to set inference_mode=False in trainer (see above)
trainer.test(dataloaders=test_loader)

# save trained model
model_export_path = "./trained_model/"
lit_module.model.save(model_export_path)

# load trained model
model = matgl.load_model(path=model_export_path)

# download a pre-trained M3GNet
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model
lit_module_finetune = PotentialLightningModule(model=model_pretrained, lr=1e-4, include_line_graph=True)

# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)

# save trained model
model_save_path = "./finetuned_model/"
lit_module_finetune.model.save(model_save_path)
# load trained model
trained_model = matgl.load_model(path=model_save_path)
