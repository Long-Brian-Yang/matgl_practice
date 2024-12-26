from __future__ import annotations
import os
import shutil
import warnings
import json
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule


class FineTuningModelModule(ModelLightningModule):
    def __init__(self, model: torch.nn.Module, include_line_graph: bool = True,
                 learning_rate: float = 1e-4, frozen_layers: Optional[list[str]] = None):
        super().__init__(model=model, include_line_graph=include_line_graph)
        self.learning_rate = learning_rate

        # Freeze specified layers if any
        if frozen_layers:
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in frozen_layers):
                    param.requires_grad = False
                    print(f"Freezing layer: {name}")

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def load_pretrained_model(checkpoint_path: str, element_types: tuple) -> M3GNet:
    """Load pretrained M3GNet model with weights from checkpoint."""
    model = M3GNet(
        element_types=element_types,
        is_intensive=True,
        readout_type="set2set",
    )

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded pretrained weights from {checkpoint_path}")

    return model


def main(json_path: str = "dataset.json",
         pretrained_path: str = "pretrained_model.ckpt",
         num_samples: int = 100,
         max_epochs: int = 10,
         batch_size: int = 1,
         learning_rate: float = 1e-4,
         frozen_layers: Optional[list[str]] = None):
    """Main fine-tuning function."""

    # Load and prepare dataset (reusing existing code)
    structures, mp_ids, eform_per_atom = load_dataset(json_path)

    if num_samples is not None:
        structures = structures[:num_samples]
        eform_per_atom = eform_per_atom[:num_samples]

    print(f"Fine-tuning with {len(structures)} structures")

    # Validate structures and setup (reusing existing code)
    dataset_elements = validate_structures(structures)
    elem_list = get_element_list(structures)

    if not verify_element_coverage(dataset_elements, elem_list):
        raise ValueError("Element coverage verification failed")

    # Load pretrained model
    model = load_pretrained_model(pretrained_path, elem_list)

    # Setup dataset and data loaders (reusing existing code)
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
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
        batch_size=batch_size,
        num_workers=0,
    )

    # Setup fine-tuning module with custom learning rate and frozen layers
    lit_module = FineTuningModelModule(
        model=model,
        include_line_graph=True,
        learning_rate=learning_rate,
        frozen_layers=frozen_layers
    )

    # Setup logging and checkpointing
    logger = CSVLogger("logs", name="M3GNet_finetuning")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="m3gnet-finetuned-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    # Setup trainer with reduced learning rate and early stopping
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Fine-tuning
    try:
        trainer.fit(
            model=lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        # Save final model
        torch.save({
            'state_dict': model.state_dict(),
            'element_types': elem_list,
        }, 'final_finetuned_model.pt')

    except Exception as e:
        print(f"Fine-tuning failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    # Plot metrics and cleanup
    plot_training_metrics(logger.version)
    cleanup_files()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune M3GNet model on custom dataset')
    parser.add_argument('json_path', type=str, help='Path to the JSON dataset file')
    parser.add_argument('pretrained_path', type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use (default: all)')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for fine-tuning (default: 1e-4)')
    parser.add_argument('--freeze_layers', nargs='+', default=None,
                        help='List of layer names to freeze (default: None)')

    args = parser.parse_args()
    main(args.json_path, args.pretrained_path, args.num_samples,
         args.max_epochs, args.batch_size, args.learning_rate,
         args.freeze_layers)
