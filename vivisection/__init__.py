#!/usr/bin/env python3

__version__ = '0.1'
__description__ = 'Debug tools for Pytorch models'
__summary__ = 'Debug tools for Pytorch models'
__license__ = 'BSD'
__author__ = 'David Völgyes'
__email__ = 'david.volgyes@ieee.org'

# system modules
import torch
import sys
import os
import importlib
from contextlib import contextmanager
from pathlib import Path
import traceback

# 3rd party modules
from loguru import logger
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters.terminal import TerminalFormatter as fmt
from torch.utils.data.sampler import Sampler
from termcolor import colored

torch_enabled = 'torch' in sys.modules


class __vivisection_settings:

    def __init__(self):
        self.interactive = False
        self.abort_on_error = True
        self.line_length = 80

        self.tf = {}
        self.tf_names = {}
        self.tf['forward_in'] = [torch.isnan, torch.isinf]
        self.tf['forward_out'] = [torch.isnan, torch.isinf]
        self.tf['forward_pre'] = [torch.isnan, torch.isinf]
        self.tf['backward_in'] = [torch.isnan, torch.isinf]
        self.tf['backward_out'] = [torch.isnan, torch.isinf]
        self.tf['forward_attr'] = [torch.isnan, torch.isinf]
        self.tf['backward_attr'] = [torch.isnan, torch.isinf]
        self.tf['loss'] = [torch.isnan, torch.isinf]

        self.tf_names['forward_in'] = ['NaN', 'inf']
        self.tf_names['forward_out'] = ['NaN', 'inf']
        self.tf_names['forward_attr'] = ['NaN', 'inf']
        self.tf_names['forward_pre'] = ['NaN', 'inf']
        self.tf_names['backward_in'] = ['NaN', 'inf']
        self.tf_names['backward_out'] = ['NaN', 'inf']
        self.tf_names['backward_attr'] = ['NaN', 'inf']
        self.tf_names['loss'] = ['NaN', 'inf']

        self.forward_hook_flag = True
        self.forward_pre_hook_flag = True
        self.backward_hook_flag = True
        self.sample_logger = None

    def set_logger(self, logger):
        self.sample_logger = logger


_settings = __vivisection_settings()


@contextmanager
def local_env(name, value):
    defined = name in os.environ
    if defined:
        saved = os.environ[name]
    os.environ[name] = str(value)

    yield

    if defined:
        os.environ[name] = saved
    else:
        os.environ.pop(name)


conda = os.environ.get('CONDA_DEFAULT_ENV', 'Not in use')


def detect_libs(name):
    # os.environ.get('CHECK_INSTALLED_MODULES', False) in ['1','true','TRUE']
    check_existence = False
    enabled, exists, version = False, False, 'N/A'
    try:
        enabled = name in sys.modules
        if enabled or check_existence:
            importlib.import_module(name)
            exists = True
            try:
                version = sys.modules[name].__version__
            except AttributeError:
                pass
    except ImportError:
        pass
    if not exists:
        version = 'not installed'
    return exists, version, enabled


torch_exists, torch_version, torch_enabled = detect_libs('torch')
tf_exists, tf_version, tf_enabled = detect_libs('tensorflow')
keras_exists, keras_version, keras_enabled = detect_libs('keras')
fastai_exists, fastai_version, fastai_enabled = detect_libs('fastai')
TF_CUDA = False
CUDA_version = None

if torch_exists:
    CUDA_version = torch.version.cuda
if tf_exists:
    with local_env('TF_CPP_MIN_LOG_LEVEL', 2):
        import tensorflow as tf
        TF_CUDA = tf.test.is_gpu_available(cuda_only='True')

logger.info('System information')
logger.info(
    f'  Python     :    {sys.version[0:5]},'
    f'   Conda:{conda}, CUDA: {CUDA_version}')
logger.info(f'  Vivisection:    {__version__}      (imported: True)')

if torch_enabled:
    logger.info(
        f'  PyTorch    :    {torch_version:8s} (imported: {torch_enabled})')
if tf_enabled:
    logger.info(
        f'  Tensorflow : {tf_version:8s} '
        f'(imported: {tf_enabled}, CUDA capable:{TF_CUDA})')
if keras_enabled:
    logger.info(
        f'  Keras      :      {keras_version:8s}'
        f' (imported: {keras_enabled})')
if fastai_enabled:
    logger.info(
        f'  FastAI     :     {fastai_version:8s}'
        f' (imported: {fastai_enabled})')

if Path('requirements.txt').exists():
    with open('requirements.txt') as f:
        for line in map(str.strip, f.readlines()):
            if line in sys.modules:
                try:
                    module = sys.modules[line]
                    if hasattr(module, '__version__'):
                        version = module.__version__
                    else:
                        version = module.VERSION
                    if isinstance(version, tuple):
                        version = ".".join(map(str, version))
                    logger.info(f'  {line:11s}:    '
                                f'{version:8s} (imported: True)')
                except AttributeError:
                    version = 'unknown'
                    logger.info(f'  {line:11s}:    '
                                f'{version:8s} (imported: True)')
else:
    logger.warning(f'Missing requirements.txt!')


def forward_hooks(new_status=None):
    global _settings
    if new_status is None:
        new_status = not _settings.forward_hook_flag
    _settings.forward_hook_flag = new_status
    logger.info(f'Forward hooks: {new_status}')


def backward_hooks(new_status=None):
    global _settings
    if new_status is None:
        new_status = not _settings.backward_hook_flag
    _settings.backward_hook_flag = new_status
    logger.info(f'Forward hooks: {new_status}')


def forward_pre_hooks(new_status=None):
    global _settings
    if new_status is None:
        new_status = not _settings.forward_pre_hook_flag
    _settings.forward_pre_hook_flag = new_status
    logger.info(f'forward_pre hooks: {new_status}')


def free_space(dirname='.'):
    try:
        f = os.statvfs(dirname)
        return (f.f_bavail*f.f_bsize/(1024**3))
    except Exception:
        return -1


def find_global_variable_name(var):
    for k, v in globals().items():
        if v is var:
            return k
    return None


def path_difference(file1, file2):
    diff = str(Path(os.path.relpath(file1, file2)))
    cnt = 0
    for i, c in enumerate(diff):
        if c not in ['.', '/']:
            break
        if c == '/':
            cnt += 1
    return cnt


def format_trace(trace, skip=0, levels=None):
    global _settings
    lines = []
    f0 = Path(trace[0].filename)
    for idx, t in enumerate(trace):
        if idx < skip:
            continue
        if levels is not None and idx + levels >= len(trace):
            break
        e = Path(t.filename).name+":"+str(t.lineno)
        if path_difference(f0, Path(t.filename)) <= 3:
            e = colored(e, 'green')
        b = "  "*idx+t.line
        n = max(_settings.line_length - len(b), 0)
        b = "  "*idx+(highlight(t.line, PythonLexer(), fmt())).strip()
        e = e.strip()
        lines.append(b+(" "*n)+e)
    return "\n".join(lines)


class SampleLogger(Sampler):

    def __init__(self, sampler):
        global _settings
        self.sampler = sampler
        self.index_history = []
        _settings.set_logger(self)

    def clear(self):
        self.index_history.clear()

    def __iter__(self):
        for idx in self.sampler:
            self.index_history.insert(0, idx)
            yield idx

    def get_samples(self, size, indices):
        for idx in indices:
            if idx < len(self.index_history):
                yield self.index_history[idx]

    def __len__(self):
        return self.sampler.__len__()


def _get_location(module):
    result = []
    result.append(colored('Location in the model:', 'yellow'))
    result.append(module.print())
    result.append(colored('\nCall trace:', 'yellow'))
    s = traceback.extract_stack()
    result.append(format_trace(s, levels=2))
    return "\n".join(result)


def _batch_indices(array, test_function):
    global _settings
    indices = []
    for i in range(array.size()[0]):
        if torch.any(test_function(array[i])):
            indices.append(i)
    if _settings.sample_logger is not None:
        indices = list(_settings.sample_logger.get_samples(
            array.size()[0], indices))
    return list(map(str, indices))


def _forward_hook(module, input, output):
    global _settings
    if not _settings.forward_hook_flag:
        return
    if type(input) != tuple:
        input = (input,)

    if type(output) != tuple:
        output = (output,)

    for i in input:
        for idx, f in enumerate(_settings.tf['forward_in']):
            if torch.any(f(i)):
                varname = _settings.tf_names["forward_in"][idx]
                text = [f'  Forward hook: {varname}'
                        ' failed for the input!']
                text.append(colored('Affected sample indices: ', 'yellow')
                            + (", ").join(_batch_indices(i, f)))
                text.append("")
                text.append(_get_location(module))
                logger.error("\n".join(text)+"\n")

                if _settings.abort_on_error:
                    sys.exit(1)

    for attr in dir(module):
        if type(getattr(module, attr)) == torch.nn.parameter.Parameter:
            for idx, f in enumerate(_settings.tf['forward_attr']):
                if torch.any(f(getattr(module, attr))):
                    varname = _settings.tf_names["forward_attr"][idx]
                    text = [
                        f'  Forward hook: {varname}'
                        f' failed for the "{attr}"'
                        ' attribute of the input!']
                    text.append(_get_location(module))
                    logger.error("\n".join(text)+"\n")
                    if _settings.abort_on_error:
                        sys.exit(1)

    for o in output:
        for idx, f in enumerate(_settings.tf['forward_out']):
            if torch.any(f(o)):
                varname = _settings.tf_names["forward_out"][idx]
                text = [f'  Forward hook: {varname}'
                        ' failed for the output!']
                logger.error("\n".join(text)+"\n")
                if _settings.abort_on_error:
                    sys.exit(1)


def _backward_hook(module, input, output):
    if not _settings.backward_hook_flag:
        return
    if type(input) != tuple:
        input = (input,)

    if type(output) != tuple:
        output = (output,)

    for i in input:
        if i is not None:
            for idx, f in enumerate(_settings.tf['backward_in']):
                if torch.any(f(i)):
                    varname = _settings.tf_names["backward_in"][idx]
                    text = [f'  Backward hook: {varname}'
                            ' failed for the input!']
                    logger.error("\n".join(text)+"\n")
                    if _settings.abort_on_error:
                        sys.exit(1)

    for attr in dir(module):
        if type(getattr(module, attr)) == torch.nn.parameter.Parameter:
            for idx, f in enumerate(_settings.tf['backward_attr']):
                if torch.any(f(getattr(module, attr))):
                    varname = _settings.tf_names["backward_attr"][idx]
                    text = [f'  Backward hook: {varname} '
                            'failed for the "{attr}" attribute of the input!']
                    logger.error("\n".join(text)+"\n")
                    if _settings.abort_on_error:
                        sys.exit(1)

    for i in output:
        if i is not None:
            for idx, f in enumerate(_settings.tf['backward_attr']):
                if torch.any(f(i)):
                    varname = _settings.tf_names["backward_attr"][idx]
                    text = [f'  Backward hook: {varname}'
                            ' failed for the output!']
                    text.append(_get_location(module))
                    logger.error("\n".join(text)+"\n")
                    if _settings.abort_on_error:
                        sys.exit(1)


def _forward_pre_hook(module, input):
    if not _settings.forward_pre_hook_flag:
        return
    return


def _ghook(*args, **kwargs):
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            for idx, f in enumerate(_settings.tf['loss']):
                if torch.any(f(a)):
                    varname = _settings.tf_names["loss"][idx]
                    text = [f'Error in the loss function: {varname}']
                    if a.grad_fn is not None:
                        varname = str(type(a.grad_fn).__name__)
                        text.append(colored('', 'red') +
                                    f'The error originates around: {varname}')
                    else:
                        text.append(
                            f'Enable "create_graph" for '
                            'more detailed error message.')
                    logger.error("\n".join(text)+"\n")
                    if _settings.abort_on_error:
                        sys.exit(1)


def debug_model(model):
    class debug_instance:
        def __init__(self, m):
            self.model = m

        def __call__(self, *args):
            tensor = self.model(*args)
            tensor.register_hook(_ghook)
            return tensor

    global saved_model
    saved_model = model

    def _r2(self):
        return _r(saved_model, self)

    def _r(self, obj=None):
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = _r(module, obj)
            lines = '(' + key + '): ' + mod_str
            if obj is not None and module is obj:
                lines = colored(lines, 'red')
            child_lines.append(lines)
        lines = extra_lines + child_lines
        name = self._get_name()
        main_str = name + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        if obj is not None and self is obj:
            main_str = colored(main_str, 'red')
        return main_str

    def set_model_hooks(module):
        module.register_forward_hook(_forward_hook)
        module.register_forward_pre_hook(_forward_pre_hook)
        module.register_backward_hook(_backward_hook)
        module.print = _r2.__get__(module, module.__class__)
    model.apply(set_model_hooks)
    return debug_instance(model)
