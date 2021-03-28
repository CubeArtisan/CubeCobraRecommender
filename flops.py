import sys

from src.ml.draftbots import DraftBot
from src.non_ml.parse_picks import FEATURES

import numpy as np
import tensorflow as tf

debugging = False
tf.config.optimizer.set_jit(not debugging)
tf.config.optimizer.set_experimental_options=({
    'layout_optimizer': True,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'debug_stripper': not debugging,
    'disable_model_pruning': False,
    'scoped_allocator_optimization': True,
    'pin_to_host_optimization': True,
    'implementation_selector': True,
    'auto_mixed_precision': True,
    'disable_meta_optimizer': False,
    'min_graph_nodes': 1,
})

def get_flops(concrete_func):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops
        

draftbot = DraftBot(np.zeros((21714,), dtype=np.float32), np.zeros((21714,64), dtype=np.float32), 1, 0.5, 0.5)

concrete = tf.function(lambda inputs: draftbot(inputs))
concrete_func = concrete.get_concrete_function(
    [tf.TensorSpec([int(sys.argv[1]), *shape], dtype=dtype) for dtype, shape, _ in FEATURES]
)

print(f'{get_flops(concrete_func):,}')