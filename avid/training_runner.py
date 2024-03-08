"""Training state runner interface."""

from avid.config import MainConfig
from avid.dashboard import Dashboard
import jax


def run_using_dashboard(config: MainConfig, run):
    app = Dashboard(run, config=config)
    app.run()
    return run


def run_using_progress(config: MainConfig, run):
    from rich.pretty import pprint
    import rich.progress as prog

    update_every = 1
    with prog.Progress(
        prog.TextColumn('[progress.description]{task.description}'),
        prog.BarColumn(80, 'light_pink3', 'deep_sky_blue4', 'green'),
        prog.MofNCompleteColumn(),
        prog.TimeElapsedColumn(),
        prog.TimeRemainingColumn(),
        prog.SpinnerColumn(),
        refresh_per_second=3,
        disable=not config.cli.show_progress,
    ) as progress:
        task = progress.add_task(
            '[bold] [deep_pink3] Training [/deep_pink3] [/bold]',
            total=run.num_steps // update_every,
        )
        for run_state in run.step_until_done():
            if run_state.curr_step % update_every == 0:
                progress.advance(task)

            if run_state.epoch_just_finished:
                status = []
                for k, v in run_state.metrics_history.items():
                    status.append(f'{k}={v[-1]:4.02f}')

                progress.update(task, description=''.join(status))

    return run
