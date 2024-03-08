"""Training dashboard."""

"""A longer-form example of using textual-plotext."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import cycle
from json import loads
from math import e
import random
import time
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from h11 import Data
import jax
import numpy as np
import pandas as pd
import pyrallis
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header, Markdown, DataTable
from textual.worker import get_current_worker, Worker
from typing_extensions import Final
import rho_plus as rp
from rich.table import Table

from textual_plotext import PlotextPlot

from avid.config import MainConfig
from avid.e_form_predictor import TrainingRun
from avid.utils import debug_structure

TEXTUAL_ICBM: Final[tuple[float, float]] = (55.9533, -3.1883)
"""The ICBM address of the approximate location of Textualize HQ."""


class Losses(PlotextPlot):
    """A widget for plotting losses during training."""

    def __init__(
        self,
        title: str,
        *,
        colors=rp.matplotlib.DARK_COLORS,
        name: str | None = None,
        id: str | None = None,  # pylint:disable=redefined-builtin
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialise the weather widget.

        Args:
            name: The name of the weather widget.
            id: The ID of the weather widget in the DOM.
            classes: The CSS classes of the weather widget.
            disabled: Whether the weather widget is disabled or not.
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._title = title
        self._unit = 'L2 Loss'
        self._data = {}
        self._colors = colors

    def on_mount(self) -> None:
        """Plot the data using Plotext."""
        self.plt.title(self._title)
        self.plt.xlabel('Time')

    def replot(self) -> None:
        """Redraw the plot."""
        if self._data and next(iter(self._data.values())):
            self.plt.clear_data()
            self.plt.xlim(0, len((list(self._data.values()) + [[0]])[0]))
            self.plt.ylabel(self._unit)
            # step = np.arange(len(next(iter(self._data))))
            max_data = 0
            for name, color in zip(self._data, cycle(self._colors)):
                yvals = self._data[name]
                max_data = max(max_data, max(yvals))
                self.plt.plot(np.arange(len(yvals)), yvals, color=color, label=name)

            self.plt.ylim(0, max_data * 1.01)

        self.refresh()

    def update_data(self, data: dict[str, Any]) -> None:
        """Update the data for the weather plot.

        Args:
            data: The data from the weather API.
            values: The name of the values to plot.
        """
        self._data = data
        self.replot()

    def update_colors(self, colors):
        self._colors = colors
        self.replot()


class TestRunner(TrainingRun):
    def __init__(self, delay=0.1, time=5):
        self.delay = delay
        self.num_steps = round(time / delay)
        self.curr_step = 0
        self.steps = range(self.num_steps)
        # self.metrics_history = {'train_loss': list(range(100)), 'test_loss': list(range(100))}
        self.metrics_history = {'train_loss': [], 'test_loss': []}

    def next_step(self):
        time.sleep(self.delay)
        self.curr_step += 1
        if self.curr_step >= self.num_steps:
            return None
        else:
            for i, k in enumerate(self.metrics_history):
                self.metrics_history[k].append(
                    0.2 + (np.exp(i) / (1 + np.exp(i))) * np.sin(self.curr_step * 2) * 0.1
                )
            return self

    def step_until_done(self):
        for _step in self.steps:
            yield self.next_step()


class Info(Widget):
    def __init__(self, dataa={}):
        super().__init__()

    def update_data(self, dataa):
        tab = self.query_one(DataTable)
        tab.clear(columns=True)
        df = pd.DataFrame(dataa)
        tab.add_columns(*df.columns)
        tab.add_rows(df.map(lambda x: '{:.03f}'.format(x)).values[-20:])

    def compose(self):
        yield DataTable(show_cursor=False, zebra_stripes=True)


class Dashboard(App):
    """An application for showing recent Textualize weather."""

    CSS = """
    Grid {
        grid-size: 2 1;
        grid-columns: 3fr 1fr;
    }

    Losses {
        padding: 1 2;
    }

    Info {
        text-style: bold;
    }
    """

    TITLE = 'Training'

    BINDINGS = [
        ('d', 'app.toggle_dark', 'Toggle light/dark mode'),
        ('q', 'app.quit', 'Quit the example'),
    ]

    data = reactive({})
    colors = reactive(lambda: rp.matplotlib.DARK_COLORS)

    def __init__(self, run: TrainingRun, plot_cols=('train_', 'test_', 'lr')) -> None:
        """Initialise the application."""
        super().__init__()
        self._run: TrainingRun = run
        self._plot_cols = plot_cols

    def on_mount(self):
        self.log(self._run)
        self.run_worker(lambda: self.stream_data(self._run), thread=True)

    def compose(self) -> ComposeResult:
        """Compose the display of the example app."""
        yield Header()
        with Grid():
            yield Losses('Losses')
            yield Info()
        yield Footer()

    def update_data(self, data):
        self.data = data
        self.watch_data()

    def watch_data(self) -> None:
        filtered_data = {
            k: v
            for k, v in self.data.items()
            if any(k.startswith(pref) for pref in self._plot_cols)
        }
        for plot in self.query(Losses).results(Losses):
            plot.update_data(filtered_data)

        self.query_one(Info).update_data(self.data)
        self.refresh(layout=True)

    def watch_colors(self) -> None:
        for plot in self.query(Losses).results(Losses):
            plot.update_colors(self.colors)

    def stream_data(self, run_state: TrainingRun):
        worker = get_current_worker()
        update_every_step = 1
        for state in run_state.step_until_done():
            if not worker.is_cancelled and state is not None:
                if state.curr_step % update_every_step == 0:
                    self.call_from_thread(self.update_data, state.metrics_history)
            else:
                break

        self.call_from_thread(self.update_data, run_state.metrics_history)

    def watch_dark(self, dark: bool) -> None:
        self.colors = rp.matplotlib.DARK_COLORS if dark else rp.matplotlib.LIGHT_COLORS
        return super().watch_dark(dark)


if __name__ == '__main__':
    config = pyrallis.argparsing.parse(MainConfig)
    from rich.pretty import pprint

    # with jax.profiler.trace('/tmp/jax-trace', create_perfetto_link=True):
    if config.do_profile:
        jax.profiler.start_trace('/tmp/jax-trace', create_perfetto_link=True)

    run = TrainingRun(config)
    app = Dashboard(run)
    app.run()

    if config.do_profile:
        jax.profiler.stop_trace()
