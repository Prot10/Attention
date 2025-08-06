"""Training utilities for PyTorch attention models."""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass

from attention_research.shared.config import Config

logger = logging.getLogger(__name__)


class TorchTrainer:
    """PyTorch trainer for attention mechanisms."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: str = "cuda",
    ) -> None:
        """Initialize PyTorch trainer.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to train
        config : Config
            Configuration object
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        device : str, optional
            Device to use for training, by default "cuda"
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training configuration
        training_config = config.get_training_config()
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.warmup_steps = training_config.warmup_steps
        self.max_steps = training_config.max_steps
        self.gradient_clip_norm = training_config.gradient_clip_norm
        self.eval_every = training_config.eval_every
        self.save_every = training_config.save_every

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps,
        )

        # Mixed precision training
        self.use_mixed_precision = config.get("hardware.mixed_precision", True)
        self.scaler: GradScaler | None = GradScaler() if self.use_mixed_precision else None

        # Model compilation (PyTorch 2.0+)
        if config.get("hardware.compile_model", True):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                logger.info("Model compiled successfully")
            except (RuntimeError, ValueError, NotImplementedError) as e:
                logger.warning("Model compilation failed: %s", e)

        # Setup logging
        log_dir = Path(config.get("logging.log_dir", "./logs"))
        experiment_name = config.get("logging.experiment_name", "attention_experiment")
        self.writer = SummaryWriter(log_dir / experiment_name / "torch")

        # Setup wandb if enabled
        self.use_wandb = config.get("logging.use_wandb", False)
        if self.use_wandb and WANDB_AVAILABLE:
            wandb_project = config.get("logging.wandb_project", "attention-research")
            wandb.init(
                project=wandb_project,
                name=f"{experiment_name}-torch-{config.get_attention_type()}",
                config={
                    "framework": "pytorch",
                    "attention_type": config.get_attention_type(),
                    "model": dict(config.get_model_config()),
                    "training": dict(config.get_training_config()),
                    "dataset": dict(config.get_dataset_config()),
                },
                tags=["pytorch", config.get_attention_type()],
            )
            logger.info("Weights & Biases initialized")
        elif self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("wandb requested but not available, skipping...")
            self.use_wandb = False

        # Setup save directories
        self.model_save_dir = Path(config.get("paths.model_save_dir", "./models"))
        self.checkpoints_dir = Path(config.get("paths.checkpoints_dir", "./checkpoints"))
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform a single training step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch of training data

        Returns
        -------
        dict[str, float]
            Dictionary of training metrics
        """
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # Backward pass
        if self.use_mixed_precision:
            if self.scaler is None:
                msg = "Scaler is None despite mixed precision being enabled"
                raise RuntimeError(msg)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()

        # Update learning rate
        if self.step < self.warmup_steps:
            self.scheduler.step()

        self.optimizer.zero_grad()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform a single evaluation step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch of evaluation data

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        return {"loss": loss.item()}

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on validation set.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                metrics = self.eval_step(batch)
                total_loss += metrics["loss"]
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def save_checkpoint(self, metrics: dict[str, float] | None = None) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        metrics : dict[str, float], optional
            Current metrics to save with checkpoint, by default None
        """
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.config,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoints_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save as latest checkpoint
        latest_path = self.checkpoints_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        logger.info("Checkpoint saved at step %d", self.step)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info("Checkpoint loaded from step %d", self.step)

    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training for %d steps", self.max_steps)

        start_time = time.time()

        while self.step < self.max_steps:
            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break

                # Training step
                metrics = self.train_step(batch)
                self.step += 1

                # Log metrics
                self.writer.add_scalar("train/loss", metrics["loss"], self.step)
                self.writer.add_scalar("train/lr", metrics["lr"], self.step)

                # Log to wandb if available
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"train/loss": metrics["loss"], "train/lr": metrics["lr"], "step": self.step})

                # Evaluation
                if self.step % self.eval_every == 0:
                    eval_metrics = self.evaluate()

                    # Log evaluation metrics
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f"eval/{key}", value, self.step)

                    # Log evaluation metrics to wandb if available
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        wandb_eval_metrics["step"] = self.step
                        wandb.log(wandb_eval_metrics)

                    # Save best model
                    if eval_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val_loss"]
                        best_model_path = self.model_save_dir / "best_model.pt"
                        torch.save(self.model.state_dict(), best_model_path)
                        logger.info("New best model saved with val_loss: %.4f", self.best_val_loss)

                    logger.info(
                        "Step %d - Train Loss: %.4f, Val Loss: %.4f, Val PPL: %.2f",
                        self.step,
                        metrics["loss"],
                        eval_metrics["val_loss"],
                        eval_metrics["val_perplexity"],
                    )

                # Save checkpoint
                if self.step % self.save_every == 0:
                    self.save_checkpoint(metrics)

            self.epoch += 1

        # Final evaluation and save
        final_metrics = self.evaluate()
        self.save_checkpoint(final_metrics)

        # Save final model
        final_model_path = self.model_save_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_model_path)

        total_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", total_time)
        logger.info("Final validation loss: %.4f", final_metrics["val_loss"])

        self.writer.close()
