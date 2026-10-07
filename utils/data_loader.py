"""Data loading utilities for the tea leaf disease detection model."""

# pylint: disable=no-member

import logging
import os
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Tea Leaf Dataset
# ============================================================

class TeaLeafDataset(Dataset):
    """Dataset class for loading and processing tea leaf images."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        subset: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the tea leaf dataset."""

        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset
        self.image_size = image_size

        # ----------------------------------------------------
        # Set class names
        # ----------------------------------------------------

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = sorted(
                [
                    entry
                    for entry in os.listdir(data_dir)
                    if os.path.isdir(
                        os.path.join(data_dir, entry)
                    )
                ]
            )

        # ----------------------------------------------------
        # Create class -> index mapping
        # ----------------------------------------------------

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.class_names)
        }

        # ----------------------------------------------------
        # Load samples only for full dataset
        # ----------------------------------------------------

        if subset == "full":
            self.samples = self.load_samples()

            self.targets = [
                sample[1]
                for sample in self.samples
            ]

            self.class_weights = self.calculate_class_weights()

        else:
            self.samples = []
            self.targets = []
            self.class_weights = None

    # ========================================================
    # Load Image Samples
    # ========================================================

    def load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and their corresponding class labels."""

        samples = []

        for class_name in self.class_names:
            class_path = os.path.join(
                self.data_dir,
                class_name,
            )

            # Skip missing class folders
            if not os.path.exists(class_path):
                logger.warning(
                    "Class folder not found: %s",
                    class_path,
                )
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    img_path = os.path.join(
                        class_path,
                        img_name,
                    )

                    samples.append(
                        (
                            img_path,
                            self.class_to_idx[class_name],
                        )
                    )

        return samples

    # ========================================================
    # Calculate Class Weights
    # ========================================================

    def calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights to handle class imbalance."""

        class_counts = np.zeros(
            len(self.class_names),
            dtype=np.int64,
        )

        for target in self.targets:
            class_counts[target] += 1

        total = len(self.targets)

        weights = [
            (
                total
                / (len(self.class_names) * count)
                if count > 0
                else 0.0
            )
            for count in class_counts
        ]

        return torch.tensor(
            weights,
            dtype=torch.float32,
        )

    # ========================================================
    # Dataset Summary
    # ========================================================

    def log_summary(self):
        """Log summary information about the dataset."""

        logger.info(
            "Dataset Summary: %s",
            self.subset,
        )

        logger.info(
            "Total samples: %s",
            len(self.samples),
        )

        counts = np.bincount(
            self.targets,
            minlength=len(self.class_names),
        )

        for name, count in zip(
            self.class_names,
            counts,
        ):
            logger.info(
                "  %s: %s",
                name,
                count,
            )

    # ========================================================
    # Dataset Length
    # ========================================================

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

        return len(self.samples)

    # ========================================================
    # Get Image
    # ========================================================

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, int]:
        """Return an image tensor and its corresponding label."""

        img_path, label = self.samples[idx]

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {img_path}"
            )

        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            self.image_size,
        )

        # ----------------------------------------------------
        # Apply Albumentations transform
        # ----------------------------------------------------

        if self.transform:
            transformed = self.transform(
                image=image,
            )

            image = transformed["image"]

        return image, label


# ============================================================
# Image Transformations
# ============================================================

def get_tea_leaf_transforms(
    image_size: Tuple[int, int],
    mode: str,
):
    """Create an Albumentations transformation pipeline."""

    base_norm = [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]

    # ========================================================
    # No Augmentation
    # ========================================================

    if mode == "none":
        return A.Compose(
            [
                A.Resize(
                    image_size[0],
                    image_size[1],
                ),
                *base_norm,
            ]
        )

    # ========================================================
    # Standard Augmentation
    # ========================================================

    if mode == "standard":
        return A.Compose(
            [
                A.Resize(
                    image_size[0],
                    image_size[1],
                ),
                A.Rotate(
                    limit=15,
                    p=0.7,
                ),
                A.HorizontalFlip(
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    p=0.2,
                ),
                *base_norm,
            ]
        )

    # ========================================================
    # Enhanced Augmentation
    # ========================================================

    if mode == "enhanced":
        return A.Compose(
            [
                A.Resize(
                    image_size[0],
                    image_size[1],
                ),
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=1.0,
                ),
                A.Sharpen(
                    alpha=(1.0, 1.0),
                    lightness=(1.0, 1.0),
                    p=1.0,
                ),
                A.Rotate(
                    limit=15,
                    p=0.7,
                ),
                A.HorizontalFlip(
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    p=0.2,
                ),
                *base_norm,
            ]
        )

    # ========================================================
    # Fallback
    # ========================================================

    return A.Compose(
        [
            A.Resize(
                image_size[0],
                image_size[1],
            ),
            *base_norm,
        ]
    )


# ============================================================
# Save Augmentation Samples
# ============================================================

def save_augmentation_samples(
    train_loader,
    save_dir,
    num_samples=10,
):
    """Save examples of images after different augmentations."""

    if num_samples <= 0:
        return

    # --------------------------------------------------------
    # Visualization pipelines
    # --------------------------------------------------------

    pipes = {
        "none": A.Compose(
            [
                A.Resize(
                    224,
                    224,
                )
            ]
        ),
        "standard": A.Compose(
            [
                A.Resize(
                    224,
                    224,
                ),
                A.Rotate(
                    limit=15,
                    p=1.0,
                ),
                A.HorizontalFlip(
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    p=0.5,
                ),
            ]
        ),
        "enhanced": A.Compose(
            [
                A.Resize(
                    224,
                    224,
                ),
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=1.0,
                ),
                A.Sharpen(
                    alpha=(1.0, 1.0),
                    lightness=(1.0, 1.0),
                    p=1.0,
                ),
                A.Rotate(
                    limit=15,
                    p=1.0,
                ),
            ]
        ),
    }

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for mode_name in pipes:
        path = os.path.join(
            save_dir,
            "aug_samples",
            mode_name,
        )

        os.makedirs(
            path,
            exist_ok=True,
        )

    print(
        f"INFO: Saving {num_samples} samples "
        f"per augmentation type to "
        f"{save_dir}/aug_samples/"
    )

    # --------------------------------------------------------
    # Access training dataset
    # --------------------------------------------------------

    dataset = train_loader.dataset

    # --------------------------------------------------------
    # Save samples
    # --------------------------------------------------------

    for i in range(
        min(num_samples, len(dataset))
    ):
        img_path, label_idx = dataset.samples[i]

        class_name = dataset.class_names[label_idx]

        # Load original image
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Apply each augmentation
        # ----------------------------------------------------

        for mode_name, pipe in pipes.items():
            transformed = pipe(
                image=image
            )["image"]

            # Convert RGB -> BGR
            save_img = cv2.cvtColor(
                transformed,
                cv2.COLOR_RGB2BGR,
            )

            filename = (
                f"sample_{i}_{class_name}.jpg"
            )

            save_path = os.path.join(
                save_dir,
                "aug_samples",
                mode_name,
                filename,
            )

            cv2.imwrite(
                save_path,
                save_img,
            )


# ============================================================
# Create Data Loaders
# ============================================================

def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    aug_type: str = "standard",
    class_names: Optional[List[str]] = None,
    image_size: Tuple[int, int] = (224, 224),
):
    """Create training, validation, and testing data loaders."""

    # --------------------------------------------------------
    # Load complete dataset once
    # --------------------------------------------------------

    full_ds = TeaLeafDataset(
        data_dir=data_dir,
        subset="full",
        class_names=class_names,
        image_size=image_size,
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if len(full_ds) == 0:
        raise ValueError(
            f"No images found in dataset directory: "
            f"{data_dir}"
        )

    logger.info(
        "Detected classes: %s",
        full_ds.class_names,
    )

    logger.info(
        "Total images: %s",
        len(full_ds),
    )

    # --------------------------------------------------------
    # Train / Validation / Test Split
    # --------------------------------------------------------

    train_idx, temp_idx = train_test_split(
        range(len(full_ds)),
        train_size=0.8,
        stratify=full_ds.targets,
        random_state=42,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        stratify=[
            full_ds.targets[i]
            for i in temp_idx
        ],
        random_state=42,
    )

    # --------------------------------------------------------
    # Build subset
    # --------------------------------------------------------

    def build_subset(
        indices,
        subset_name,
    ):
        """Build a dataset subset and its DataLoader."""

        # Training uses selected augmentation.
        # Validation and testing use only
        # resize + normalization.

        if subset_name == "train":
            mode = aug_type
        else:
            mode = "none"

        ds = TeaLeafDataset(
            data_dir=data_dir,
            transform=get_tea_leaf_transforms(
                image_size,
                mode,
            ),
            subset=subset_name,
            class_names=full_ds.class_names,
            image_size=image_size,
        )

        # ----------------------------------------------------
        # Use selected indices
        # ----------------------------------------------------

        ds.samples = [
            full_ds.samples[i]
            for i in indices
        ]

        ds.targets = [
            sample[1]
            for sample in ds.samples
        ]

        # ----------------------------------------------------
        # Log dataset information
        # ----------------------------------------------------

        ds.log_summary()

        # ----------------------------------------------------
        # Create DataLoader
        # ----------------------------------------------------

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(
                subset_name == "train"
            ),
        )

        return loader

    # --------------------------------------------------------
    # Create loaders
    # --------------------------------------------------------

    train_loader = build_subset(
        train_idx,
        "train",
    )

    val_loader = build_subset(
        val_idx,
        "val",
    )

    test_loader = build_subset(
        test_idx,
        "test",
    )

    # --------------------------------------------------------
    # Return loaders and class weights
    # --------------------------------------------------------

    return (
        train_loader,
        val_loader,
        test_loader,
        full_ds.class_weights,
    )
