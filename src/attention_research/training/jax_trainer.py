"""Training utilities for JAX/Flax attention models."""

import logging
import time
from pathlib import Path
from typing import Any, NamedTuple

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

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from omegaconf import OmegaConf
from tqdm import tqdm

from attention_research.shared.config import Config

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """Extended training state for JAX trainer."""

    batch_stats: Any = None


class TrainMetrics(NamedTuple):
    """Training metrics structure."""

    loss: float
    accuracy: float
    learning_rate: float


class JaxTrainer:
    """JAX/Flax trainer for attention mechanisms."""

    def __init__(
        self,
        model: Any,
        config: Config,
        train_data: Any,
        val_data: Any,
    ) -> None:
        """Initialize JAX trainer.

        Parameters
        ----------
        model
            Flax model to train
        config : Config
            Configuration object
        train_data
            Training data
        val_data
            Validation data
        """
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        # Training configuration
        training_config = config.get_training_config()
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.warmup_steps = training_config.warmup_steps
        self.max_steps = training_config.max_steps
        self.gradient_clip_norm = training_config.gradient_clip_norm
        self.eval_every = training_config.eval_every
        self.save_every = training_config.save_every

        # Model configuration
        model_config = config.get_model_config()
        self.batch_size = training_config.batch_size
        self.vocab_size = model_config.vocab_size

        # Setup save directories
        self.model_save_dir = Path(config.get("paths.model_save_dir", "./models"))
        self.checkpoints_dir = Path(config.get("paths.checkpoints_dir", "./checkpoints"))
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if enabled
        self.use_wandb = config.get("logging.use_wandb", False)
        if self.use_wandb and WANDB_AVAILABLE:
            experiment_name = config.get("experiment.name", "attention_research")
            config_dict = OmegaConf.to_container(config.config, resolve=True)
            wandb.init(
                project="attention-research",
                name=f"{experiment_name}_jax",
                config=config_dict,  # type: ignore[arg-type]
                tags=["jax", "attention", config.get("model.attention_type", "vanilla")],
            )

        # Initialize training state
        self.state = None
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Setup optimizer
        self._setup_optimizer()

        # Initialize model
        self._initialize_model()

    def _setup_optimizer(self) -> None:
        """Set up the optimizer with learning rate schedule."""
        # Learning rate schedule with warmup
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.learning_rate,
            transition_steps=self.warmup_steps,
        )

        cosine_schedule = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=self.max_steps - self.warmup_steps,
            alpha=0.1,
        )

        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[self.warmup_steps],
        )

        # Optimizer with gradient clipping and weight decay
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.weight_decay,
            ),
        )

    def _initialize_model(self) -> None:
        """Initialize model parameters and training state."""
        # Get dummy input to initialize model
        model_config = self.config.get_model_config()
        dummy_input = {
            "input_ids": jnp.ones((1, model_config.max_sequence_length), dtype=jnp.int32),
            "attention_mask": jnp.ones((1, model_config.max_sequence_length), dtype=jnp.int32),
        }

        # Initialize parameters
        rng = jax.random.PRNGKey(self.config.get("training.seed", 42))
        variables = self.model.init(rng, **dummy_input)

        # Create training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=self.optimizer,
            batch_stats=variables.get("batch_stats"),
        )

    @staticmethod
    def _compute_loss(logits: jnp.ndarray, labels: jnp.ndarray, vocab_size: int) -> jnp.ndarray:
        """Compute cross-entropy loss.

        Parameters
        ----------
        logits : jnp.ndarray
            Model logits of shape (batch_size, seq_len, vocab_size)
        labels : jnp.ndarray
            Target labels of shape (batch_size, seq_len)
        vocab_size : int
            Vocabulary size

        Returns
        -------
        jnp.ndarray
            Computed loss
        """
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Compute cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)

        # Mask padding tokens (assuming 0 is padding)
        mask = labels_flat != 0
        loss = jnp.where(mask, loss, 0.0)

        return jnp.sum(loss) / jnp.sum(mask)

    @staticmethod
    def _compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute accuracy.

        Parameters
        ----------
        logits : jnp.ndarray
            Model logits of shape (batch_size, seq_len, vocab_size)
        labels : jnp.ndarray
            Target labels of shape (batch_size, seq_len)

        Returns
        -------
        jnp.ndarray
            Computed accuracy
        """
        predictions = jnp.argmax(logits, axis=-1)

        # Mask padding tokens
        mask = labels != 0
        correct = (predictions == labels) & mask

        return jnp.sum(correct) / jnp.sum(mask)

    def train_step(self, state: TrainState, batch: dict[str, jnp.ndarray]) -> tuple[TrainState, TrainMetrics]:
        """Perform a single training step.

        Parameters
        ----------
        state : TrainState
            Current training state
        batch : dict[str, jnp.ndarray]
            Batch of training data

        Returns
        -------
        tuple[TrainState, TrainMetrics]
            Updated training state and metrics
        """

        def loss_fn(params: Any) -> tuple[jnp.ndarray, tuple[jnp.ndarray, Any]]:
            variables = {"params": params}
            if state.batch_stats is not None:
                variables["batch_stats"] = state.batch_stats

            logits = state.apply_fn(
                variables,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                deterministic=False,
                mutable=["batch_stats"] if state.batch_stats is not None else False,
            )

            # Handle mutable state
            if isinstance(logits, tuple):
                logits, updates = logits
                new_batch_stats = updates.get("batch_stats")
            else:
                new_batch_stats = None

            loss = self._compute_loss(logits, batch["labels"], self.vocab_size)
            accuracy = self._compute_accuracy(logits, batch["labels"])

            return loss, (accuracy, new_batch_stats)

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (accuracy, new_batch_stats)), grads = grad_fn(state.params)

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        # Update batch stats if using batch norm
        if new_batch_stats is not None:
            new_state = new_state.replace(batch_stats=new_batch_stats)

        # Get current learning rate
        lr = (
            self.optimizer.learning_rate_fn(state.step)
            if hasattr(self.optimizer, "learning_rate_fn")
            else self.learning_rate
        )

        metrics = TrainMetrics(
            loss=float(loss),
            accuracy=float(accuracy),
            learning_rate=float(lr),
        )

        return new_state, metrics

    def eval_step(self, state: TrainState, batch: dict[str, jnp.ndarray]) -> TrainMetrics:
        """Perform a single evaluation step.

        Parameters
        ----------
        state : TrainState
            Current training state
        batch : dict[str, jnp.ndarray]
            Batch of evaluation data

        Returns
        -------
        TrainMetrics
            Evaluation metrics
        """
        variables = {"params": state.params}
        if state.batch_stats is not None:
            variables["batch_stats"] = state.batch_stats

        logits = state.apply_fn(
            variables,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            deterministic=True,
        )

        loss = self._compute_loss(logits, batch["labels"], self.vocab_size)
        accuracy = self._compute_accuracy(logits, batch["labels"])

        return TrainMetrics(
            loss=float(loss),
            accuracy=float(accuracy),
            learning_rate=0.0,  # Not relevant for evaluation
        )

    def evaluate(self, state: TrainState) -> dict[str, float]:
        """Evaluate the model on validation set.

        Parameters
        ----------
        state : TrainState
            Current training state

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics
        """
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch in tqdm(self.val_data, desc="Evaluating"):
            # Convert to JAX arrays
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}

            metrics = self.eval_step(state, batch_jax)
            total_loss += metrics.loss
            total_accuracy += metrics.accuracy
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        perplexity = jnp.exp(avg_loss).item()

        return {
            "val_loss": float(avg_loss),
            "val_accuracy": float(avg_accuracy),
            "val_perplexity": perplexity,
        }

    def save_checkpoint(self, state: TrainState, metrics: dict[str, float] | None = None) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        state : TrainState
            Current training state
        metrics : dict[str, float], optional
            Current metrics to save with checkpoint, by default None
        """
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "params": state.params,
            "batch_stats": state.batch_stats,
            "opt_state": state.opt_state,
            "best_val_loss": self.best_val_loss,
            "config": self.config.config,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        # Save checkpoint
        checkpoint_path = self.checkpoints_dir / f"checkpoint_step_{self.step}.npz"
        np.savez(checkpoint_path, **checkpoint)

        # Save as latest checkpoint
        latest_path = self.checkpoints_dir / "latest.npz"
        np.savez(latest_path, **checkpoint)

        logger.info("Checkpoint saved at step %d", self.step)

    def load_checkpoint(self, checkpoint_path: str) -> TrainState:
        """Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file

        Returns
        -------
        TrainState
            Loaded training state
        """
        checkpoint = np.load(checkpoint_path, allow_pickle=True)

        # Restore training state
        if self.state is None:
            msg = "Training state not initialized"
            raise RuntimeError(msg)

        state = self.state.replace(
            params=checkpoint["params"].item(),
            batch_stats=checkpoint["batch_stats"].item(),
            opt_state=checkpoint["opt_state"].item(),
            step=int(checkpoint["step"]),
        )

        self.step = int(checkpoint["step"])
        self.epoch = int(checkpoint["epoch"])
        self.best_val_loss = float(checkpoint["best_val_loss"])

        logger.info("Checkpoint loaded from step %d", self.step)
        return state

    def train(self) -> None:
        """Train the model."""
        logger.info("Starting JAX training for %d steps", self.max_steps)

        # Compile training step
        train_step_jit = jax.jit(self.train_step)

        start_time = time.time()
        state = self.state

        if state is None:
            msg = "Training state not initialized. Call create_train_state first."
            raise RuntimeError(msg)

        while self.step < self.max_steps:
            for batch in self.train_data:
                if self.step >= self.max_steps:
                    break

                # Convert to JAX arrays
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}

                # Training step
                state, metrics = train_step_jit(state, batch_jax)
                self.step += 1

                # Log training metrics to wandb if available
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "train/loss": float(metrics.loss),
                            "train/learning_rate": float(metrics.learning_rate),
                            "step": self.step,
                        }
                    )

                # Evaluation
                if self.step % self.eval_every == 0:
                    eval_metrics = self.evaluate(state)

                    # Log evaluation metrics to wandb if available
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        wandb_eval_metrics["step"] = self.step
                        wandb.log(wandb_eval_metrics)

                    # Save best model
                    if eval_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val_loss"]
                        best_model_path = self.model_save_dir / "best_model.npz"
                        np.savez(best_model_path, params=state.params, batch_stats=state.batch_stats)
                        logger.info("New best model saved with val_loss: %.4f", self.best_val_loss)

                    logger.info(
                        "Step %d - Train Loss: %.4f, Val Loss: %.4f, Val PPL: %.2f, Val Acc: %.4f",
                        self.step,
                        metrics.loss,
                        eval_metrics["val_loss"],
                        eval_metrics["val_perplexity"],
                        eval_metrics["val_accuracy"],
                    )

                # Save checkpoint
                if self.step % self.save_every == 0:
                    self.save_checkpoint(
                        state,
                        {
                            "train_loss": float(metrics.loss),
                            "train_accuracy": float(metrics.accuracy),
                        },
                    )

            self.epoch += 1

        # Final evaluation and save
        final_metrics = self.evaluate(state)
        self.save_checkpoint(state, final_metrics)

        # Save final model
        final_model_path = self.model_save_dir / "final_model.npz"
        np.savez(final_model_path, params=state.params, batch_stats=state.batch_stats)

        total_time = time.time() - start_time
        logger.info("Training completed in %.2f seconds", total_time)
        logger.info("Final validation loss: %.4f", final_metrics["val_loss"])

        self.state = state
