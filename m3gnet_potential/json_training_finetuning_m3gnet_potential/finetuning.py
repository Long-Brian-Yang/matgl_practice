import os
import shutil
import lightning as pl
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.utils.training import PotentialLightningModule
from dataset_json import prepare_data, cleanup

def finetune(model_path=None, max_epochs=50):
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data("dataset.json", batch_size=1)
    
    # Load the base model for fine-tuning
    if model_path and os.path.exists(model_path):
        m3gnet_nnp = matgl.load_model(path=model_path)
    else:
        m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    
    model_pretrained = m3gnet_nnp.model
    property_offset = m3gnet_nnp.element_refs.property_offset
    
    # Create lightning module for fine-tuning
    lit_module_finetune = PotentialLightningModule(
        model=model_pretrained,
        element_refs=property_offset,
        lr=1e-4,
        include_line_graph=True,
        stress_weight=0.01
    )
    
    # Setup the trainer
    logger = CSVLogger("logs", name="M3GNet_finetuning")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
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
    
    return model_save_path

if __name__ == "__main__":
    # I havent test if I can use own trained model or only use pre-trianed model
    # model_path = "./trained_model/"  # Uncomment to use your trained model
    finetune()
    cleanup()