import os
import shutil
import lightning as pl
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.utils.training import PotentialLightningModule
from dataset import prepare_data, cleanup

def pretrain(max_epochs=1):
    # Prepare data
    train_loader, val_loader, _ = prepare_data(batch_size=16)
    
    # Load pre-trained model
    m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    model_pretrained = m3gnet_nnp.model
    
    # Get element reference energies and data normalization parameters
    element_refs_m3gnet = m3gnet_nnp.element_refs.property_offset
    data_std = m3gnet_nnp.data_std
    
    # Create lightning module
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
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=logger,
        inference_mode=False
    )
    
    # Train the model
    trainer.fit(
        model=lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Save the model
    model_export_path = "./trained_model/"
    lit_module.model.save(model_export_path)
    
    return model_export_path

if __name__ == "__main__":
    pretrain()
    cleanup()