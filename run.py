import argparse
import torch
import torch.nn as nn
import logging
import os

from utils.data_loader import create_data_loaders, save_augmentation_samples
from utils.train_evaluation import Trainer
from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Tea Leaf Disease Classes
# ============================================================

CLASSES = [
    'Anthracnose',
    'bird eye spot',
    'brown blight',
    'gray light',
    'healthy',
    'red leaf spot',
    'white spot'
]


# ============================================================
# Main Function
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    num_classes = len(CLASSES)

    logger.info(f"Using device: {device}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {CLASSES}")
    logger.info(f"Augmentations Strategy: {args.aug_type.upper()}")


    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_names=CLASSES,
        image_size=(224, 224),
        aug_type=args.aug_type
    )


    # --------------------------------------------------------
    # Calculate Class Weights
    # --------------------------------------------------------

    train_labels = train_loader.dataset.targets

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    logger.info(
        f"Class weights: {class_weights.numpy()}"
    )


    # --------------------------------------------------------
    # Create Model
    # --------------------------------------------------------

    model = get_model(
        model_name=args.model_name,
        num_classes=num_classes
    )

    model = model.to(device)


    # --------------------------------------------------------
    # Select Loss Function
    # --------------------------------------------------------

    if args.loss == "focal":

        criterion = FocalLoss(
            alpha=1.0,
            gamma=2.0
        )

    elif args.loss == "labelsmoothing":

        criterion = LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=0.1
        )

    else:

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
        )


    # --------------------------------------------------------
    # Initialize Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        device=device,
        class_names=CLASSES,
        model_name=args.model_name,
        class_weights=class_weights,
        aug_type=args.aug_type
    )


    # --------------------------------------------------------
    # Save Augmentation Samples
    # --------------------------------------------------------

    if args.save_samples > 0:

        save_augmentation_samples(
            train_loader=train_loader,
            save_dir=trainer.base_dir,
            num_samples=args.save_samples
        )


    # --------------------------------------------------------
    # Start Training
    # --------------------------------------------------------

    logger.info(
        f"Starting training: "
        f"{args.model_name} | "
        f"Aug: {args.aug_type} | "
        f"Loss: {args.loss}"
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        criterion=criterion,
        lr=args.lr
    )


    # --------------------------------------------------------
    # Test Model
    # --------------------------------------------------------

    test_metrics, test_loss = trainer.evaluate(
        test_loader
    )

    logger.info(
        f"Final Results - Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    logger.info(
        f"Final Results - Loss: "
        f"{test_loss:.4f}"
    )


# ============================================================
# Command Line Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Tea Leaf Disease Standalone Training"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset"
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model-name",
        type=str,
        default="customcnn",
        choices=[
            "customcnn",
            "vitb16",
            "efficientnetb3",
            "mobilenetv3",
            "densenet121",
            "vgg19",
            "resnet50",
            "hybridmodel"
        ],
        help="Model from factory"
    )


    # --------------------------------------------------------
    # Batch Size
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )


    # --------------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------------

    parser.add_argument(
        "--aug-type",
        type=str,
        default="standard",
        choices=[
            "none",
            "standard",
            "enhanced"
        ],
        help=(
            "Choose training pipeline: "
            "none, standard, or enhanced"
        )
    )


    # --------------------------------------------------------
    # Save Augmentation Samples
    # --------------------------------------------------------

    parser.add_argument(
        "--save-samples",
        type=int,
        default=20,
        help=(
            "Number of images to save for each "
            "augmentation type (0 to disable)"
        )
    )


    # --------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )


    # --------------------------------------------------------
    # Loss Function
    # --------------------------------------------------------

    parser.add_argument(
        "--loss",
        type=str,
        default="crossentropy",
        choices=[
            "crossentropy",
            "focal",
            "labelsmoothing"
        ]
    )


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping"
    )


    # --------------------------------------------------------
    # Save Directory
    # --------------------------------------------------------

    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints"
    )


    return parser.parse_args()


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()