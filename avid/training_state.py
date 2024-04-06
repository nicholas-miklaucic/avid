import functools as ft
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from shutil import copytree
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyrallis
from clu import metrics
from flax import struct
from flax.training import train_state

from avid.checkpointing import best_ckpt, run_config
from avid.config import LossConfig, MainConfig
from avid.dataset import DataBatch, dataloader, load_file


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    rmse: metrics.Average.from_output('rmse')
    mae: metrics.Average.from_output('mae')
    grad_norm: metrics.Average.from_output('grad_norm')


class TrainState(train_state.TrainState):
    metrics: Metrics
    last_grad_norm: float


def create_train_state(module, optimizer, rng):
    params = module.init(rng, DataBatch.new_empty(52, 24, 5), training=False)
    tx = optimizer
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty(), last_grad_norm=0
    )


@ft.partial(jax.jit)
@chex.assert_max_traces(3)
def train_step(config: LossConfig, state: TrainState, batch: DataBatch, rng):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)

    def loss_fn(params):
        preds = state.apply_fn(
            params, batch, training=True, rngs={'dropout': dropout_train_key}
        ).squeeze()

        return config.regression_loss(preds, batch.e_form)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads, last_grad_norm=grad_norm)
    return state


@ft.partial(jax.jit)
@chex.assert_max_traces(3)
def compute_metrics(*, config: LossConfig, state: TrainState, batch: DataBatch):
    preds = state.apply_fn(state.params, batch, training=False).squeeze()

    loss = config.regression_loss(preds, batch.e_form)
    mae = jnp.abs(preds - batch.e_form).mean()
    rmse = jnp.sqrt(optax.losses.squared_error(preds, batch.e_form).mean())
    metric_updates = state.metrics.single_from_model_output(
        mae=mae, loss=loss, rmse=rmse, grad_norm=state.last_grad_norm
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@struct.dataclass
class Checkpoint:
    state: TrainState
    seed: int
    metrics_history: Mapping[str, Sequence[float]]
    curr_epoch: float


class TrainingRun:
    def __init__(self, config: MainConfig):
        self.seed = random.randint(100, 1000)
        self.rng = jax.random.key(self.seed)
        print(f'Seed: {self.seed}')
        self.config = config

        self.metrics_history = defaultdict(list)

        self.num_epochs = config.num_epochs
        self.steps_in_epoch, self.dl = dataloader(config, split='train', infinite=True)
        self.steps_in_test_epoch, self.test_dl = dataloader(config, split='valid', infinite=True)
        self.num_steps = self.steps_in_epoch * self.num_epochs
        self.steps = range(self.num_steps)
        self.curr_step = 0
        self.start_time = time.monotonic()
        self.scheduler = self.config.train.lr_schedule(
            num_epochs=self.num_epochs, steps_in_epoch=self.steps_in_epoch
        )
        self.optimizer = self.config.train.optimizer(self.scheduler)
        self.state = create_train_state(self.config.build_regressor(), self.optimizer, self.rng)

        if config.restart_from is not None:
            self.state = self.state.replace(
                params=best_ckpt(config.restart_from)['state']['params']
            )

        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            best_fn=lambda metrics: metrics['test_loss'],
            best_mode='min',
            max_to_keep=4,
            enable_async_checkpointing=False,
            keep_time_interval=timedelta(minutes=30),
        )
        self.mngr = ocp.CheckpointManager(
            ocp.test_utils.create_empty('/tmp/jax_ckpt'),
            options=opts,
        )

        self.test_loss = 1000

    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    def step(self, step, batch):
        self.curr_step = step
        if step >= self.num_steps:
            return None

        self.state = train_step(
            config=self.config.train.loss, state=self.state, batch=batch, rng=self.rng
        )  # get updated train state (which contains the updated parameters)
        self.state = compute_metrics(
            config=self.config.train.loss, state=self.state, batch=batch
        )  # aggregate batch metrics

        if self.should_log or self.should_ckpt:  # one training epoch has passed
            for metric, value in self.state.metrics.compute().items():  # compute metrics
                if metric == 'grad_norm':
                    self.metrics_history['grad_norm'].append(value.item())  # record metrics
                    continue
                self.metrics_history[f'train_{metric}'].append(value.item())  # record metrics

                if not self.should_ckpt:
                    if f'test_{metric}' in self.metrics_history:
                        self.metrics_history[f'test_{metric}'].append(
                            self.metrics_history[f'test_{metric}'][-1]
                        )
                    else:
                        self.metrics_history[f'test_{metric}'].append(0)

            self.state = self.state.replace(
                metrics=self.state.metrics.empty()
            )  # reset train_metrics for next training epoch

            self.metrics_history['lr'].append(self.lr)
            self.metrics_history['step'].append(self.curr_step)
            self.metrics_history['epoch'].append(self.curr_step / self.steps_in_epoch)
            self.metrics_history['rel_mins'].append((time.monotonic() - self.start_time) / 60)
            self.metrics_history['throughput'].append(
                self.curr_step * self.config.batch_size / self.metrics_history['rel_mins'][-1]
            )

        if self.should_ckpt:
            # Compute metrics on the test set after each training epoch
            self.test_state = self.state
            for _i, test_batch in zip(range(self.steps_in_test_epoch), self.test_dl):
                self.test_state = compute_metrics(
                    config=self.config.train.loss, state=self.test_state, batch=test_batch
                )

            for metric, value in self.test_state.metrics.compute().items():
                if metric == 'grad_norm':
                    continue
                self.metrics_history[f'test_{metric}'].append(value.item())
                if f'{metric}' == 'loss':
                    self.test_loss = value.item()
            self.mngr.save(
                self.curr_step,
                args=ocp.args.StandardSave(self.ckpt()),
                metrics={'test_loss': self.test_loss},
            )
            self.mngr.wait_until_finished()

        return self

    def step_until_done(self):
        for step, batch in zip(self.steps, self.dl):
            yield self.step(step, batch)

    def run_to_completion(self):
        for step, batch in zip(self.steps, self.dl):
            self.step(step, batch)

    @property
    def should_log(self):
        return (self.curr_step + 1) % (self.steps_in_epoch // self.config.log.logs_per_epoch) == 0

    @property
    def should_ckpt(self):
        return (self.curr_step + 1) % (2 * self.steps_in_epoch) == 0

    @property
    def lr(self):
        return self.scheduler(self.curr_step).item() * 100.0

    def ckpt(self):
        """Checkpoint PyTree."""
        return Checkpoint(
            self.state, self.seed, dict(self.metrics_history), self.curr_step / self.steps_in_epoch
        )

    def save_final(self, out_dir: str | PathLike):
        """Save final model to directory."""
        self.mngr.wait_until_finished()
        copytree(self.mngr.directory, Path(out_dir) / 'ckpts/')

    def finish(self):
        now = datetime.now()
        if self.config.log.exp_name is None:
            exp_name = now.strftime('%m-%d-%H')
        else:
            exp_name = self.config.log.exp_name

        folder = Path('logs/') / f'{exp_name}_{self.seed}'
        folder.mkdir(exist_ok=True)

        pd.DataFrame(self.metrics_history).to_feather(folder / 'metrics.feather')

        with open(folder / 'time_stopped.txt', 'w') as f:
            f.write(now.isoformat())

        with open(folder / 'config.toml', 'w') as outfile:
            pyrallis.cfgparsing.dump(self.config, outfile)

        self.save_final(folder / 'final_ckpt')

        return folder


class DiffusionDataParallelTrainer:
    """
    Trainer class using data parallelism with JAX.
    This trainer leverages JAX's `pmap` for parallel training across multiple devices (GPUs/TPUs).
    It handles the model training loop, including gradient computation, parameter updates, and evaluation.

    Attributes:
        model (Any): The model to be trained.
        input_shape (Tuple[int, ...]): The shape of the image input tensor.
        weights_filename (str): Filename where the trained model weights will be saved.
        learning_rate (float): Learning rate for the optimizer.
        params_path (Optional[str]): Path to pre-trained model parameters for initializing the model, if available.

    Methods:
        create_train_state(learning_rate, text_input_shape, image_input_shape): Initializes the training state, including parameters and optimizer.
        train_step(state, texts, images): Performs a single training step, including forward pass, loss computation, and gradients update.
        train(train_loader, num_epochs, val_loader): Runs the training loop over the specified number of epochs, using the provided data loaders for training and validation.
        evaluation_step(state, texts, images): Performs an evaluation step, computing forward pass and loss without updating model parameters.
        evaluate(test_loader): Evaluates the model performance on a test dataset.
        save_params(): Saves the model parameters to a file.
        load_params(filename): Loads model parameters from a file.
    """

    def __init__(
        self,
        config: MainConfig,
        weights_filename: str,
        encoder_path: PathLike = 'logs/e_form_equivariant_patch_235',
        params_path: Optional[str] = None,
    ) -> None:
        self.config = config
        enc_config = run_config(encoder_path)
        enc_mlp = enc_config.build_mlp()
        self.spec_encoder = enc_mlp.spec_embed
        self.patchifier = enc_mlp.im_embed
        ckpt = best_ckpt(encoder_path)
        params = ckpt['state']['params']['params']

        self.spec_emb_params = {'params': params['spec_embed']}
        self.patchifier_params = {'params': params['im_embed']}
        self.model = self.config.build_diffusion()
        self.params = None
        self.params_path = params_path
        self.num_parameters = None
        self.best_val_loss = float('inf')
        self.weights_filename = weights_filename
        self.train_step = DiffusionDataParallelTrainer.train_step
        self.evaluation_step = DiffusionDataParallelTrainer.evaluation_step

        self.num_epochs = config.num_epochs
        self.steps_in_epoch, self.dl = dataloader(config, split='train', infinite=True)
        self.steps_in_test_epoch, self.test_dl = dataloader(config, split='valid', infinite=True)
        self.num_steps = self.steps_in_epoch * self.num_epochs
        self.steps = range(self.num_steps)
        self.curr_step = 0
        self.start_time = time.monotonic()
        self.scheduler = self.config.train.lr_schedule(
            num_epochs=self.num_epochs, steps_in_epoch=self.steps_in_epoch
        )
        self.optimizer = self.config.train.optimizer(self.scheduler)
        self.state = self.create_train_state()

    def encode(self, batch: DataBatch):
        spec_emb = self.spec_encoder.apply(self.spec_emb_params, data=batch, training=False)
        return self.patchifier.apply(self.patchifier_params, im=spec_emb)

    def create_train_state(self) -> Any:
        rngs = {'params': jax.random.key(0), 'dropout': jax.random.key(1)}
        batch = load_file(self.config, 0)
        params = self.model.init(rngs, self.encode(batch), training=True)['params']

        if self.params_path is not None:
            params = self.load_params(self.params_path)

        self.num_parameters = sum(param.size for param in jax.tree_util.tree_leaves(params))
        print(f'Number of parameters: {self.num_parameters}')
        state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.optimizer
        )
        return state

    @staticmethod
    def train_step(state: Any, images: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        def loss_fn(params):
            key = jax.random.PRNGKey(int(time.time()))
            noises = jax.random.normal(key, shape=images.shape)
            pred_noises, pred_images = state.apply_fn(
                {'params': params},
                images,
                rngs={'dropout': jax.random.PRNGKey(int(time.time()))},
                training=True,
            )
            return jnp.mean(jnp.square(pred_noises - noises)) + jnp.mean(
                jnp.square(pred_images - images)
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            count = 0
            for step, batch in zip(range(self.steps_in_epoch), self.dl):
                encoded = self.encode(batch)
                self.state, loss = self.train_step(state=self.state, images=encoded)
                total_loss += jnp.mean(loss)
                count += 1

            mean_loss = total_loss / count
            print(f'Epoch {epoch+1}, Train Loss: {mean_loss}')

            val_loss = self.evaluate(self.steps_in_test_epoch, self.test_dl)
            print(f'Epoch {epoch+1}, Val Loss: {val_loss}')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            print('New best validation score achieved, saving model...')
            self.save_params()
        return

    @staticmethod
    def evaluation_step(state: Any, images: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=images.shape)
        pred_noises, pred_images = state.apply_fn(
            {'params': state.params},
            images,
            rngs={'dropout': jax.random.PRNGKey(int(time.time()))},
            training=False,
        )
        return jnp.mean(jnp.square(pred_noises - noises)) + jnp.mean(
            jnp.square(pred_images - images)
        )

    def evaluate(
        self, num_steps: int, test_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> None:
        total_loss = 0.0
        count = 0
        for step, batch in zip(range(num_steps), test_loader):
            encoded = self.encode(batch)
            loss = self.evaluation_step(self.state, encoded)
            total_loss += jnp.mean(loss)
            count += 1

        mean_loss = total_loss / count
        return mean_loss

    def get_ema_weights(self, params, ema=0.999):
        def func(x):
            return x * ema + (1 - ema) * x

        return jax.tree_util.tree_map(func, params)

    def save_params(self) -> None:
        self.params = flax.jax_utils.unreplicate(self.state.params)
        self.params = self.get_ema_weights(self.params)
        with open(self.weights_filename, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load_params(self, filename: str):
        with open(filename, 'rb') as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())

        return self.params
