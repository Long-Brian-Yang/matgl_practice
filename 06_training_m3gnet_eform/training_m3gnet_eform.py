from __future__ import annotations
import os
import shutil
import warnings
import zipfile
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import lightning as pl
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.models import M3GNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")


def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    if not os.path.exists("mp.2018.6.1.json"):
        f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
        with zipfile.ZipFile(f.local_path) as zf:
            zf.extractall(".")

    data = pd.read_json("mp.2018.6.1.json")
    structures = []
    mp_ids = []
    i = 0
    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
        i = i + 1
        if i > 1000:
            break
    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


def main():
    structures, mp_ids, eform_per_atom = load_dataset()

    structures = structures[:100]
    eform_per_atom = eform_per_atom[:100]

    # get element types in the dataset
    elem_list = get_element_list(structures)
    # setup a graph converter
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
    # convert the raw dataset into M3GNetDataset
    mp_dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels={"eform": eform_per_atom},
        include_line_graph=True,
    )

    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )

    my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=my_collate_fn,
        batch_size=2,
        num_workers=1,
    )

    # setup the architecture of M3GNet model
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    # setup the M3GNetTrainer
    lit_module = ModelLightningModule(model=model, include_line_graph=True)

    logger = CSVLogger("logs", name="M3GNet_training")
    trainer = pl.Trainer(max_epochs=20, accelerator="cpu", logger=logger)
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Get the version directory
    version = logger.version  # Get the actual version number used
    metrics_path = f"logs/M3GNet_training/version_{version}/metrics.csv"

    try:
        metrics = pd.read_csv(metrics_path)

        # Create a new figure for clarity
        plt.figure(figsize=(10, 6))

        # Plot training and validation metrics
        metrics["train_MAE"].dropna().plot(label='Training MAE')
        metrics["val_MAE"].dropna().plot(label='Validation MAE')

        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # Save the plot before cleanup
        plt.savefig('training_progress.png')
        plt.close()

    except FileNotFoundError:
        print(f"Could not find metrics file at {metrics_path}")
        print("Available files in logs directory:")
        for root, dirs, files in os.walk("logs"):
            print(f"\nDirectory: {root}")
            for file in files:
                print(f"- {file}")

    # Cleanup
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

    shutil.rmtree("logs")


if __name__ == "__main__":
    main()
