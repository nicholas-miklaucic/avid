import subprocess

import jax
import pyrallis
from jax.lib import xla_client

from avid.config import MainConfig
from avid.dataset import load_file
from avid.utils import debug_structure, flax_summary


# https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html
def to_dot_graph(x):
    return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))


@pyrallis.argparsing.wrap()
def show_model(config: MainConfig):
    kwargs = dict(training=False)
    batch = load_file(config, 0)
    mod = config.vit.build()

    debug_structure(mod)
    out, params = mod.init_with_output(jax.random.key(0), im=batch, **kwargs)
    debug_structure(batch=batch, module=mod, out=out)
    flax_summary(mod, batch, **kwargs)

    def loss(params):
        preds = mod.apply(params, batch, training=False)
        return config.train.loss.regression_loss(preds, batch.e_form)

    grad_loss = jax.xla_computation(jax.value_and_grad(loss))(params)
    with open('model.hlo', 'w') as f:
        f.write(grad_loss.as_hlo_text())
    with open('model.dot', 'w') as f:
        f.write(grad_loss.as_hlo_dot_graph())

    grad_loss_opt = jax.jit(jax.value_and_grad(loss)).lower(params).compile()
    with open('model_opt.hlo', 'w') as f:
        f.write(grad_loss_opt.as_text())
    with open('model_opt.dot', 'w') as f:
        f.write(to_dot_graph(grad_loss_opt.as_text()))

    # debug_structure(grad_loss_opt.cost_analysis())

    for f in ('model.dot', 'model_opt.dot'):
        subprocess.run(['sfdp', f, '-Tsvg', '-O', '-x'])


if __name__ == '__main__':
    show_model()
