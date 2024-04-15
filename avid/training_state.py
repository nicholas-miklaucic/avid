import functools as ft
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from shutil import copytree
from typing import Mapping, Sequence

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyrallis
from clu import metrics
from flax import struct
from flax.training import train_state

from avid.checkpointing import best_ckpt
from avid.config import LossConfig, MainConfig
from avid.dataset import DataBatch, dataloader


@struct.dataclass
class RegressionMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    rmse: metrics.Average.from_output('rmse')
    mae: metrics.Average.from_output('mae')
    grad_norm: metrics.Average.from_output('grad_norm')


@struct.dataclass
class DiffusionMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    rec_loss: metrics.Average.from_output('rec_loss')
    align_loss: metrics.Average.from_output('align_loss')
    diffuser_loss: metrics.Average.from_output('diffuser_loss')
    grad_norm: metrics.Average.from_output('grad_norm')


class TrainState(train_state.TrainState):
    metrics: metrics.Collection
    last_grad_norm: float


def create_train_state(module, optimizer, rng, metrics):
    params = module.init(rng, DataBatch.new_empty(52, 24, 5), training=False)
    tx = optimizer
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=metrics,
        last_grad_norm=0,
    )


@struct.dataclass
class Checkpoint:
    state: TrainState
    seed: int
    metrics_history: Mapping[str, Sequence[float]]
    curr_epoch: float


class TrainingRun:
    def __init__(self, config: MainConfig):
        self.seed = random.randint(100, 1000)
        self.rng = {'params': jax.random.key(self.seed)}
        if config.task == 'diled':
            self.rng['params'], self.rng['noise'], self.rng['time'] = jax.random.split(
                self.rng['params'], 3
            )
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
        self.state = self.create_train_state()

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

    @staticmethod
    @ft.partial(jax.jit, static_argnames=('task', 'config'))
    @chex.assert_max_traces(3)
    def compute_metrics(*, task: str, config: LossConfig, state: TrainState, batch: DataBatch, rng):
        if task == 'e_form':
            preds = state.apply_fn(state.params, batch, training=False, rngs=rng).squeeze()
            loss = config.regression_loss(preds, batch.e_form)
            mae = jnp.abs(preds - batch.e_form).mean()
            rmse = jnp.sqrt(optax.losses.squared_error(preds, batch.e_form).mean())
            metric_updates = state.metrics.single_from_model_output(
                mae=mae, loss=loss, rmse=rmse, grad_norm=state.last_grad_norm
            )
        elif task == 'diled':
            _loss, losses = state.apply_fn(state.params, batch, training=False, rngs=rng)
            losses = {k: jnp.mean(v) for k, v in losses.items()}
            metric_updates = state.metrics.single_from_model_output(
                grad_norm=state.last_grad_norm, **losses
            )
        else:
            msg = f'Unknown task {task}'
            raise ValueError(msg)

        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    @staticmethod
    @ft.partial(jax.jit, static_argnames=('task', 'config'))
    @chex.assert_max_traces(3)
    def train_step(task: str, config: LossConfig, state: TrainState, batch: DataBatch, rng):
        """Train for a single step."""
        rngs = {k: v for k, v in rng.items()} if isinstance(rng, dict) else {'params': rng}
        rngs['dropout'] = jax.random.fold_in(key=rng['params'], data=state.step)
        if task == 'diled':
            rngs['dropout'], rngs['noise'], rngs['time'] = jax.random.split(rngs['dropout'], 3)

        def loss_fn(params):
            if task == 'e_form':
                preds = state.apply_fn(params, batch, training=True, rngs=rngs).squeeze()
                return (config.regression_loss(preds, batch.e_form), {})
            else:
                loss, losses = state.apply_fn(params, batch, training=True, rngs=rngs)
                return loss.mean(), losses

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, losses = grad_fn(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads, last_grad_norm=grad_norm)
        return state

    def create_train_state(self):
        if self.config.task == 'e_form':
            return create_train_state(
                self.config.build_regressor(), self.optimizer, self.rng, RegressionMetrics.empty()
            )
        else:
            return create_train_state(
                self.config.build_diled(), self.optimizer, self.rng, DiffusionMetrics.empty()
            )

    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    def step(self, step, batch):
        self.curr_step = step
        if step >= self.num_steps:
            return None

        self.state = self.train_step(
            task=self.config.task,
            config=self.config.train.loss,
            state=self.state,
            batch=batch,
            rng=self.rng,
        )  # get updated train state (which contains the updated parameters)
        self.state = self.compute_metrics(
            task=self.config.task,
            config=self.config.train.loss,
            state=self.state,
            batch=batch,
            rng=self.rng,
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
                self.test_state = self.compute_metrics(
                    task=self.config.task,
                    config=self.config.train.loss,
                    state=self.test_state,
                    batch=test_batch,
                    rng=self.rng,
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
