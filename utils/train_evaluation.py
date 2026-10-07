import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=7,
        min_delta=0.001
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def __call__(self, val_loss):

        if self.best_loss is None:

            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:

            self.counter += 1

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


# ============================================================
# Metrics
# ============================================================

class Metrics:

    def __init__(self, class_names):

        self.class_names = class_names


    def calculate_metrics(
        self,
        y_true,
        y_pred,
        y_probs=None
    ):

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        acc = accuracy_score(
            y_true,
            y_pred
        )


        # ----------------------------------------------------
        # Precision, Recall, F1
        # ----------------------------------------------------

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average='macro',
                zero_division=0
            )
        )


        metrics = {

            'accuracy': acc,

            'precision_macro': precision,

            'recall_macro': recall,

            'f1_macro': f1

        }


        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        if y_probs is not None:

            try:

                metrics['auc'] = roc_auc_score(
                    y_true,
                    y_probs,
                    multi_class='ovr',
                    average='macro'
                )

            except ValueError:

                metrics['auc'] = 0.0

            except Exception:

                metrics['auc'] = 0.0

        else:

            metrics['auc'] = 0.0


        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=range(len(self.class_names))
        )

        metrics['cm_data'] = cm


        # ----------------------------------------------------
        # Per-class sensitivity and specificity
        # ----------------------------------------------------

        for i, name in enumerate(
            self.class_names
        ):

            tp = cm[i, i]

            fn = cm[i, :].sum() - tp

            fp = cm[:, i].sum() - tp

            tn = cm.sum() - (
                tp + fn + fp
            )


            # Sensitivity / Recall

            sensitivity = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )


            # Specificity

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0.0
            )


            # Make a safe metric name

            metric_name = (
                name.lower()
                .replace(' ', '_')
            )


            metrics[
                f'{metric_name}_sensitivity'
            ] = sensitivity


            metrics[
                f'{metric_name}_specificity'
            ] = specificity


        return metrics


# ============================================================
# Trainer
# ============================================================

class Trainer:

    def __init__(
        self,
        model,
        device,
        class_names,
        model_name='customcnn',
        aug_type='standard',
        class_weights=None
    ):

        self.model = model.to(device)

        self.device = device

        self.class_names = class_names

        self.aug_type = aug_type

        self.model_name = (
            f'{model_name}_{aug_type}'
        )

        self.class_weights = class_weights

        self.metrics_calc = Metrics(
            class_names
        )


        # ----------------------------------------------------
        # Result directories
        # ----------------------------------------------------

        self.base_dir = os.path.join(
            'Result',
            self.model_name
        )

        self.ckpt_dir = os.path.join(
            self.base_dir,
            'checkpoints'
        )

        self.logs_dir = os.path.join(
            self.base_dir,
            'logs'
        )


        os.makedirs(
            self.ckpt_dir,
            exist_ok=True
        )

        os.makedirs(
            self.logs_dir,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Training history
        # ----------------------------------------------------

        self.history = {

            'train_accuracy': [],

            'train_loss': [],

            'val_accuracy': [],

            'val_loss': [],

            'val_auc': [],

            'train_auc': []

        }


    # ========================================================
    # Training
    # ========================================================

    def train(
        self,
        train_loader,
        val_loader,
        epochs=50,
        criterion=None,
        lr=0.001
    ):

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        if criterion is None:

            criterion = nn.CrossEntropyLoss()


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        early_stop = EarlyStopping(
            patience=10
        )


        best_acc = 0.0


        # ====================================================
        # Epoch Loop
        # ====================================================

        for epoch in range(epochs):

            self.model.train()

            epoch_train_loss = 0.0

            all_train_preds = []

            all_train_labels = []

            all_train_probs = []


            # ------------------------------------------------
            # Training batches
            # ------------------------------------------------

            for imgs, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}"
            ):

                imgs = imgs.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )


                # Clear gradients

                optimizer.zero_grad()


                # Forward pass

                outputs = self.model(
                    imgs
                )


                # Calculate loss

                loss = criterion(
                    outputs,
                    labels
                )


                # Backpropagation

                loss.backward()


                # Update weights

                optimizer.step()


                # Track loss

                epoch_train_loss += (
                    loss.item()
                )


                # Predictions

                probs = torch.softmax(
                    outputs,
                    dim=1
                )

                _, preds = torch.max(
                    outputs,
                    1
                )


                all_train_preds.extend(
                    preds.detach()
                    .cpu()
                    .numpy()
                )

                all_train_labels.extend(
                    labels.detach()
                    .cpu()
                    .numpy()
                )

                all_train_probs.extend(
                    probs.detach()
                    .cpu()
                    .numpy()
                )


            # ------------------------------------------------
            # Average training loss
            # ------------------------------------------------

            if len(train_loader) > 0:

                avg_train_loss = (
                    epoch_train_loss /
                    len(train_loader)
                )

            else:

                avg_train_loss = 0.0


            # ------------------------------------------------
            # Training metrics
            # ------------------------------------------------

            train_acc = accuracy_score(
                all_train_labels,
                all_train_preds
            )


            try:

                train_auc = roc_auc_score(
                    np.array(all_train_labels),
                    np.array(all_train_probs),
                    multi_class='ovr',
                    average='macro'
                )

            except Exception:

                train_auc = 0.0


            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_results, val_loss = self.evaluate(
                val_loader,
                criterion,
                save_cm=False
            )


            # ------------------------------------------------
            # Save history
            # ------------------------------------------------

            self.history[
                'train_loss'
            ].append(
                avg_train_loss
            )

            self.history[
                'train_accuracy'
            ].append(
                train_acc
            )

            self.history[
                'train_auc'
            ].append(
                train_auc
            )

            self.history[
                'val_loss'
            ].append(
                val_loss
            )

            self.history[
                'val_accuracy'
            ].append(
                val_results['accuracy']
            )

            self.history[
                'val_auc'
            ].append(
                val_results.get(
                    'auc',
                    0.0
                )
            )


            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            logger.info(
                f"Epoch {epoch + 1}: "
                f"Train Acc={train_acc:.4f}, "
                f"Val Acc={val_results['accuracy']:.4f}"
            )

            logger.info(
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Train AUC={train_auc:.4f}, "
                f"Val AUC={val_results.get('auc', 0.0):.4f}"
            )


            # ------------------------------------------------
            # Save best model
            # ------------------------------------------------

            if (
                val_results['accuracy']
                > best_acc
            ):

                best_acc = (
                    val_results['accuracy']
                )


                best_model_path = os.path.join(
                    self.ckpt_dir,
                    'best_model.pth'
                )


                torch.save(
                    self.model.state_dict(),
                    best_model_path
                )


                logger.info(
                    f"Best model saved: "
                    f"{best_model_path}"
                )


            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if early_stop(val_loss):

                logger.info(
                    f"Early stopping at "
                    f"epoch {epoch + 1}"
                )

                break


        # ====================================================
        # Training Complete
        # ====================================================

        self.save_final_metrics()

        self.plot_history()


    # ========================================================
    # Save Training Metrics
    # ========================================================

    def save_final_metrics(self):

        final_path = os.path.join(
            self.base_dir,
            'final_trainMetrics.json'
        )


        self.history[
            'aug_type'
        ] = self.aug_type


        self.history[
            'model_name'
        ] = self.model_name


        # Convert NumPy values to Python values

        history_to_save = {}

        for key, value in self.history.items():

            if isinstance(value, list):

                history_to_save[key] = [
                    float(v)
                    if isinstance(
                        v,
                        (np.float32, np.float64)
                    )
                    else v
                    for v in value
                ]

            else:

                history_to_save[key] = value


        with open(
            final_path,
            'w'
        ) as f:

            json.dump(
                history_to_save,
                f,
                indent=4
            )


        logger.info(
            f"Training metrics saved to: "
            f"{final_path}"
        )


    # ========================================================
    # Plot Training History
    # ========================================================

    def plot_history(self):

        if len(
            self.history['train_loss']
        ) == 0:

            return


        # ----------------------------------------------------
        # Loss plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            self.history['train_loss'],
            label='Train Loss'
        )

        plt.plot(
            self.history['val_loss'],
            label='Validation Loss'
        )

        plt.title(
            f'Loss History - '
            f'{self.model_name}'
        )

        plt.xlabel('Epoch')

        plt.ylabel('Loss')

        plt.legend()

        plt.grid(True)

        plt.tight_layout()


        loss_path = os.path.join(
            self.base_dir,
            'loss_curve.png'
        )

        plt.savefig(
            loss_path
        )

        plt.close()


        # ----------------------------------------------------
        # Accuracy plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            self.history['train_accuracy'],
            label='Train Accuracy'
        )

        plt.plot(
            self.history['val_accuracy'],
            label='Validation Accuracy'
        )

        plt.title(
            f'Accuracy History - '
            f'{self.model_name}'
        )

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')

        plt.legend()

        plt.grid(True)

        plt.tight_layout()


        accuracy_path = os.path.join(
            self.base_dir,
            'accuracy_curve.png'
        )

        plt.savefig(
            accuracy_path
        )

        plt.close()


        logger.info(
            f"Training curves saved in: "
            f"{self.base_dir}"
        )


    # ========================================================
    # Save Confusion Matrix
    # ========================================================

    def save_confusion_matrix(
        self,
        cm,
        filename='confusion_matrix.png'
    ):

        plt.figure(
            figsize=(10, 8)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )


        plt.ylabel(
            'Actual'
        )

        plt.xlabel(
            'Predicted'
        )

        plt.title(
            f'Confusion Matrix - '
            f'{self.model_name}'
        )


        plt.tight_layout()


        save_path = os.path.join(
            self.base_dir,
            filename
        )


        plt.savefig(
            save_path
        )

        plt.close()


        logger.info(
            f"Confusion matrix saved to: "
            f"{save_path}"
        )


    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        loader,
        criterion=None,
        save_cm=True
    ):

        self.model.eval()


        all_preds = []

        all_labels = []

        all_probs = []


        total_loss = 0.0


        # ----------------------------------------------------
        # Disable gradients
        # ----------------------------------------------------

        with torch.no_grad():

            for imgs, labels in loader:

                imgs = imgs.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )


                # Forward pass

                outputs = self.model(
                    imgs
                )


                # Loss

                if criterion is not None:

                    loss = criterion(
                        outputs,
                        labels
                    )

                    total_loss += (
                        loss.item()
                    )


                # Probabilities

                probs = torch.softmax(
                    outputs,
                    dim=1
                )


                # Predictions

                _, preds = torch.max(
                    outputs,
                    1
                )


                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_probs.extend(
                    probs.cpu().numpy()
                )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        all_labels = np.array(
            all_labels
        )

        all_preds = np.array(
            all_preds
        )

        all_probs = np.array(
            all_probs
        )


        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = self.metrics_calc.calculate_metrics(

            all_labels,

            all_preds,

            all_probs

        )


        # ----------------------------------------------------
        # Save confusion matrix
        # ----------------------------------------------------

        if save_cm:

            self.save_confusion_matrix(
                metrics['cm_data']
            )


        # ----------------------------------------------------
        # Average loss
        # ----------------------------------------------------

        if len(loader) > 0:

            avg_loss = (
                total_loss /
                len(loader)
            )

        else:

            avg_loss = 0.0


        return metrics, avg_loss