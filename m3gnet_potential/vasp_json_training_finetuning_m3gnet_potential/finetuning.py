import os
import shutil
from pathlib import Path
import logging
import argparse
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import matgl
from matgl.utils.training import PotentialLightningModule
from dataset_json import prepare_data, cleanup


def setup_logging(log_dir: str = "logs") -> None:
    """
    setup logging system

    Args:
        log_dir (str): log directory

    Returns:
        None
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/finetuning.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """
    Parse command line arguments

    Args:
        None

    Returns:
        args (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser(description='Fine-tune M3GNet model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the pre-trained model')
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--stress-weight', type=float, default=0.01,
                        help='Weight for stress loss')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--output-dir', type=str, default='./finetuned_model',
                        help='Directory to save outputs')
    parser.add_argument('--dataset-path', type=str, default='dataset.json',
                        help='Path to dataset JSON file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Training device (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


def finetune(args) -> str:
    """
    Fine-tune M3GNet model with specified parameters.

    Args:
        args (argparse.Namespace): parsed arguments

    Returns:
        model_save_path (str): path to the saved model
    """

    # setup logging
    setup_logging()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled")

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # prepare datasets
        logging.info("Preparing datasets...")
        train_loader, val_loader, test_loader = prepare_data(
            args.dataset_path,
            batch_size=args.batch_size,
        )

        if not all([train_loader, val_loader, test_loader]):
            raise ValueError("Failed to prepare datasets")

        # load pretrained model
        if args.model_path and os.path.exists(args.model_path):
            logging.info(f"Loading model from {args.model_path}")
            m3gnet_nnp = matgl.load_model(path=args.model_path)
        else:
            logging.info("Loading pretrained M3GNet model")
            m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        # get element reference energies and data normalization parameters
        model_pretrained = m3gnet_nnp.model
        property_offset = m3gnet_nnp.element_refs.property_offset

        # create lightning module
        lit_module_finetune = PotentialLightningModule(
            model=model_pretrained,
            element_refs=property_offset,
            lr=args.learning_rate,
            include_line_graph=True,
            stress_weight=args.stress_weight
        )

        # setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints"),
                filename="model-{epoch:02d}-{val_Total_Loss:.4f}",
                save_top_k=3,
                monitor="val_Total_Loss",
                mode="min"
            ),
            EarlyStopping(
                monitor="val_Total_Loss",
                patience=args.patience,
                mode="min"
            )
        ]

        logger = CSVLogger(str(output_dir / "logs"), name="M3GNet_finetuning")

        # setup trainer
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.device,
            logger=logger,
            callbacks=callbacks,
            inference_mode=False,
            deterministic=True
        )

        # fine-tuning
        logging.info("Starting fine-tuning...")
        trainer.fit(
            model=lit_module_finetune,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        # save model
        model_save_path = str(output_dir / "final_model")
        lit_module_finetune.model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")

        return model_save_path

    except Exception as e:
        logging.error(f"Fine-tuning failed: {str(e)}")
        raise
    finally:
        try:
            cleanup()
            logging.info("Cleanup completed")
        except Exception as e:
            logging.error(f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    args = parse_args()
    try:
        finetune(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
