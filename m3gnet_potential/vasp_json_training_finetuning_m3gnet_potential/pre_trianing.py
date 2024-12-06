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
            logging.FileHandler(f"{log_dir}/pretraining.log"),
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
    parser = argparse.ArgumentParser(description='Pretrain M3GNet model')
    parser.add_argument('--max-epochs', type=int, default=1,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for training')
    parser.add_argument('--force-weight', type=float, default=1.0,
                        help='Weight for force loss')
    parser.add_argument('--stress-weight', type=float, default=0.1,
                        help='Weight for stress loss')
    parser.add_argument('--decay-steps', type=int, default=100,
                        help='Learning rate decay steps')
    parser.add_argument('--decay-alpha', type=float, default=0.01,
                        help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--output-dir', type=str, default='./trained_model',
                        help='Directory to save outputs')
    parser.add_argument('--dataset-path', type=str, default='dataset.json',
                        help='Path to dataset JSON file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Training device (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


def pretrain(args) -> str:
    """
    Pretrain M3GNet model with specified parameters

    Args:
        args (argparse.Namespace): command line arguments

    Returns:
        str: Path to saved model
    """

    # setup logging
    setup_logging()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled")

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # prepare datasets
        logging.info("Preparing datasets...")
        train_loader, val_loader, test_loader = prepare_data(
            args.dataset_path,
            batch_size=args.batch_size
        )

        if not all([train_loader, val_loader, test_loader]):
            raise ValueError("Data loaders initialization failed")

        # load pretrained model
        logging.info("Loading pretrained M3GNet model...")
        m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        model_pretrained = m3gnet_nnp.model

        # get element reference energies and data normalization parameters
        element_refs_m3gnet = m3gnet_nnp.element_refs.property_offset
        data_std = m3gnet_nnp.data_std

        # create lightning module
        lit_module = PotentialLightningModule(
            model=model_pretrained,
            lr=args.learning_rate,
            include_line_graph=True,
            force_weight=args.force_weight,
            stress_weight=args.stress_weight,
            element_refs=element_refs_m3gnet,
            data_std=data_std,
            decay_steps=args.decay_steps,
            decay_alpha=args.decay_alpha
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

        logger = CSVLogger(str(output_dir / "logs"), name="M3GNet_training")

        # setup trainer
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.device,
            logger=logger,
            callbacks=callbacks,
            inference_mode=False,
            deterministic=True
        )

        # start pretraining
        logging.info("Starting pretraining...")
        trainer.fit(
            model=lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        # save model
        model_save_path = str(output_dir / "final_model")
        lit_module.model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")

        return model_save_path

    except Exception as e:
        logging.error(f"Pretraining failed: {str(e)}")
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
        pretrain(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
