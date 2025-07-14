#!/usr/bin/env python

"""
Utility lib for personal projects, supports py3 only.

Covering areas:
    - Logging;
    - Config save/load;
    - Decoupled parameter server-client arch;
"""
import ast
import cProfile as profile
# Import std-modules.
import collections
import concurrent.futures
import configparser
import copy
import csv
import datetime
import difflib
import fnmatch
import functools
import gettext
import glob
import hashlib
import importlib
import json
import linecache
import locale
import logging
import logging.config
import math
import multiprocessing
import operator
import os
import os.path as osp
import platform
import plistlib
import pprint as pp
import pstats
import queue
import re
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
import traceback
import types
import typing
import urllib.parse
import urllib.request
import uuid
import warnings
from types import SimpleNamespace

# region globals

_script_dir = osp.abspath(osp.dirname(__file__))
TXT_CODEC = 'utf-8'  # Importable.
LOCALE_CODEC = locale.getpreferredencoding()
MAIN_CFG_FILENAME = 'app.json'
DEFAULT_CFG_FILENAME = 'default.json'
PLATFORM = platform.system()
if PLATFORM == 'Windows':
    import winreg

# endregion


# region classes

class ClassicSingleton:
    _instances = {}

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Create instance using `object.__new__` directly to avoid triggering overridden `__new__`
            cls._instances[cls] = object.__new__(cls)
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]

    def __new__(cls, *args, **kwargs):
        if cls in cls._instances:
            # Allow the instance to exist if already created
            return cls._instances[cls]
            # Otherwise, raise error if someone tries to use `cls()` directly
        raise RuntimeError("Use `cls.instance()` to access the singleton instance.")


class MetaSingleton(type):
    """
    - usage: class MyClass(metaclass=MetaSingleton)
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BorgSingleton:
    """
    - Borg pattern: all instances share the same state, but not the same identity
    - override _shared_borg_state to avoid child polluting states of parent instances
    - ref: https://www.geeksforgeeks.org/singleton-pattern-in-python-a-complete-guide/
    """
    _shared_borg_state = {}

    def __new__(cls, *args, **kwargs):
        obj = super(BorgSingleton, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_borg_state
        return obj


class SingletonDecorator:
    """
    - Decorator to build Singleton class, single-inheritance only.
    - Usage:
        class MyClass: ...
        myobj = SingletonDecorator(MyClass, args, kwargs)
    """
    def __init__(self, klass, *args, **kwargs):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


class ExceptableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exception:
            raise self.exception


class LowPassLogFilter(object):
    """
    Logging filter: Show log messages below input level.
    - CRITICAL = 50
    - FATAL = CRITICAL
    - ERROR = 40
    - WARNING = 30
    - WARN = WARNING
    - INFO = 20
    - DEBUG = 10
    - NOTSET = 0
    """

    def __init__(self, level):
        self.__level = level

    def filter(self, log):
        return log.levelno <= self.__level


class HighPassLogFilter(object):
    """Logging filter: Show log messages above input level."""

    def __init__(self, level):
        self.__level = level

    def filter(self, log):
        return log.levelno >= self.__level


class BandPassLogFilter(object):
    def __init__(self, levelbounds):
        self.__levelbounds = levelbounds

    def filter(self, log):
        return self.__levelbounds[0] <= log.levelno <= self.__levelbounds[1]


class OfflineJSON:
    def __init__(self, file_path):
        self.path = file_path

    def exists(self):
        return osp.isfile(self.path)

    def load(self):
        return load_json(self.path) if self.exists() else None

    def save(self, data: dict):
        save_json(self.path, data)

    def merge(self, props: dict):
        data = self.load()
        if not data:
            return self.save(props)
        data.update(props)
        self.save(data)
        return data


def get_platform_tmp_dir():
    plat_dir_map = {
        'Windows': osp.join(str(os.getenv('LOCALAPPDATA')), 'Temp'),
        'Darwin': osp.expanduser('~/Library/Caches'),
        'Linux': '/tmp'
    }
    return plat_dir_map.get(PLATFORM)


class RerunLock:
    """
    - Lock process from reentering when seeing lock file on disk
    - use semaphore-like behaviour with an instance limit
    - Because lockfile is created by pyutil, we also save the occupier pid and .py path (name) in it
    - if name is a path, e.g., __file__, then lockfile will be named after its basename
    """

    def __init__(self, name, folder=None, logger=None, max_instances=1, max_retries=3, retrieve_delay_sec=0.1):
        folder = folder or osp.join(get_platform_tmp_dir(), '_util')
        filename = f'lock_{extract_path_stem(name)}.{os.getpid()}.lock.json'
        self.name = name
        self.lockFile = osp.join(folder, filename)
        self.nMaxInstances = max_instances
        self.logger = logger or glogger
        self.maxRetries = max_retries
        self.retrieveDelaySec = retrieve_delay_sec
        self._setup_signal_handlers()
        self._cleanup_zombie_locks()

    def lock(self):
        for retry in range(self.maxRetries + 1):  # +1 to include the initial attempt
            locks = self._get_existing_locks()
            is_locked = len(locks) >= self.nMaxInstances
            
            if is_locked:
                # PID is at index 1 after splitting by '.' e.g. "lock_test.12345.lock.json" -> PID is 12345
                locker_pids = [int(lock.split(".")[1]) for lock in locks]
                if retry < self.maxRetries:
                    self.logger.info(f'{self.name} is locked by processes: {locker_pids}. Retry {retry + 1}/{self.maxRetries} after {self.retrieveDelaySec}s')
                    time.sleep(self.retrieveDelaySec)
                    continue
                else:
                    self.logger.warning(f'{self.name} is locked by processes: {locker_pids}. Max retries ({self.maxRetries}) reached. Will block new instances until unlocked.')
                    return False
            
            # Attempt to acquire the lock
            save_json(self.lockFile, {
                'pid': os.getpid(),
                'name': self.name,
            })
            
            # Double-check: verify we didn't exceed max instances after creating our lock
            # This handles the race condition where multiple processes create locks simultaneously
            locks_after = self._get_existing_locks()
            if len(locks_after) > self.nMaxInstances:
                # exceeded the limit, remove our lock and retry
                self.logger.info(f'Race condition detected: {len(locks_after)} locks found after creation (max: {self.nMaxInstances}). Removing our lock and retrying.')
                self.unlock()
                if retry < self.maxRetries:
                    time.sleep(self.retrieveDelaySec)
                    continue
                else:
                    return False
            
            return True

    def unlock(self):
        try:
            os.remove(self.lockFile)
        except FileNotFoundError:
            self.logger.warning(f'{self.name} already unlocked. Safely ignored.')
            return False
        except Exception:
            failure = traceback.format_exc()
            self.logger.error(f""""\
Failed to unlock {self.name}:
Details:
{failure}

Advice: 
- Delete the lock by hand: {self.lockFile}""")
            return False
        return True

    def unlock_all(self):
        locks = glob.glob(osp.join(osp.dirname(self.lockFile), f'lock_{osp.basename(self.name)}.*.lock.json'))
        for lock in locks:
            os.remove(lock)
        return True

    def is_locked(self):
        return osp.isfile(self.lockFile)

    def _setup_signal_handlers(self):
        """Setup signal handlers to ensure proper cleanup on termination"""
        # CAUTION:
        # - windows grpc server crashes with signals:
        #   - ValueError: signal only works in main thread of the main interpreter
        # - signals are disabled for windows
        if threading.current_thread() is threading.main_thread():
            common_sigs = [
                signal.SIGABRT,
                signal.SIGFPE,
                signal.SIGILL,
                signal.SIGINT,
                signal.SIGSEGV,
                signal.SIGTERM,
            ]
            plat_sigs = [
                signal.SIGBREAK,
                # CAUTION
                # - CTRL_C_EVENT, CTRL_BREAK_EVENT not working on Windows
                # signal.CTRL_C_EVENT,
                # signal.CTRL_BREAK_EVENT,
            ] if PLATFORM == 'Windows' else [
                # CAUTION:
                # - SIGCHLD as an alias is safe to ignore
                # - SIGKILL must be handled by os.kill()
                signal.SIGALRM,
                signal.SIGBUS,
                # signal.SIGCHLD,
                # - SIGCONT: CTRL+Z is allowed for bg process
                # signal.SIGCONT,
                signal.SIGHUP,
                # signal.SIGKILL,
                signal.SIGPIPE,
            ]
            for sig in common_sigs + plat_sigs:
                signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, sig, frame):
        msg = f'Terminated due to signal: {signal.Signals(sig).name}; Will unlock'
        self.logger.warning(msg)
        self.unlock()
        raise RuntimeError(msg)

    def _cleanup_zombie_locks(self):
        """Clean up lock files from processes that are no longer running"""
        locks = self._get_existing_locks()
        zombie_locks = [lock for lock in locks if not is_pid_running(int(lock.split(".")[1]))]
        for lock in zombie_locks:
            safe_remove(osp.join(osp.dirname(self.lockFile), lock))
    
    def _get_existing_locks(self):
        """Get a list of existing lock filenames for this lock name"""
        return [osp.basename(lock) for lock in glob.glob(osp.join(osp.dirname(self.lockFile), f'lock_{extract_path_stem(self.name)}.*.lock.json'))]


class Tracer:
    """
    - custom module-ignore rules
    - trace calls and returns
    - exclude first, then include
    - usage: use in source code
      - tracer = util.Tracer(exclude_funcname_pattern='stop')
      - tracer.start()
      - # add traceable code here
      - tracer.stop()
    """

    def __init__(self,
                 excluded_modules: set[str] = None,
                 exclude_filename_pattern: str = None,
                 include_filename_pattern: str = None,
                 exclude_funcname_pattern: str = None,
                 include_funcname_pattern: str = None,
                 trace_func=None,
                 exclude_builtins=True):
        self.exclMods = {'builtins'} if excluded_modules is None else excluded_modules
        self.exclFilePatt = re.compile(exclude_filename_pattern) if exclude_filename_pattern else None
        self.inclFilePatt = re.compile(include_filename_pattern) if include_filename_pattern else None
        self.exclFuncPatt = re.compile(exclude_funcname_pattern) if exclude_funcname_pattern else None
        self.inclFuncPatt = re.compile(include_funcname_pattern) if include_funcname_pattern else None
        self.traceFunc = trace_func
        if exclude_builtins:
            self.ignore_stdlibs()

    def start(self):
        sys.settrace(self.traceFunc or self._trace_calls_and_returns)

    @staticmethod
    def stop():
        sys.settrace(None)

    def ignore_stdlibs(self):
        def _get_stdlib_module_names():
            import distutils.sysconfig
            stdlib_dir = distutils.sysconfig.get_python_lib(standard_lib=True)
            return {f.replace(".py", "") for f in os.listdir(stdlib_dir)}

        py_ver = sys.version_info
        std_libs = set(sys.stdlib_module_names) if py_ver.major >= 3 and py_ver.minor >= 10 else _get_stdlib_module_names()
        self.exclMods.update(std_libs)

    def _trace_calls_and_returns(self, frame, event, arg):
        """
        track hook for function calls. Usage:
        sys.settrace(trace_calls_and_returns)
        """
        if event not in ('call', 'return'):
            return
        module_name = frame.f_globals.get('__name__')
        if module_name is not None and module_name in self.exclMods:
            return
        filename = frame.f_code.co_filename
        if self.exclFilePatt and self.exclFuncPatt.search(filename):
            return
        if self.inclFilePatt and not self.inclFilePatt.search(filename):
            return
        func_name = frame.f_code.co_name
        if self.exclFuncPatt and self.exclFuncPatt.search(func_name):
            return
        if self.inclFuncPatt and not self.inclFuncPatt.search(func_name):
            return
        line_number = frame.f_lineno
        line = linecache.getline(filename, line_number).strip()
        if event == 'call':
            args = ', '.join(f'{arg}={repr(frame.f_locals[arg])}' for arg in frame.f_code.co_varnames[:frame.f_code.co_argcount])
            print(f'Call: {module_name}.{func_name}({args}) - {line}')
            return self._trace_calls_and_returns
        print(f'Call: {module_name}.{func_name} => {arg} - {line}')


class Cache:
    """
    cross-session caching: using temp-file to retrieve data based on hash changes
    - constraints:
      - data retrieval/parsing is expensive
      - one cache per data-source
    - cache is a mediator b/w app and data-source as a retriever only, cuz user's saving intent is always towards source, no need to cache a saving action
    - for cross-session caching, save hash into cache, then when instantiate cache object, always load hash from cache to compare with incoming hash
    - app must provide retriever function: retriever(src) -> json_data
      - because it'd cost the same to retrieve data from a json-file source as from cache, so no json default is provided
    - e.g., loading a complex tree-structure from a file:
      - tree_cache = Cache('/path/to/file.tree', lambda: src: load_data(src), '/tmp/my_app')
      - # ... later
      - cached_tree_data = tree_cache.retrieve()
    """

    def __init__(self, data_source, data_retriever, cache_dir=get_platform_tmp_dir(), cache_type='cache', algo='checksum', source_seed='6ba7b810-9dad-11d1-80b4-00c04fd430c8'):
        assert algo in ['checksum', 'mtime']
        self.srcURL = data_source
        self.retriever = data_retriever
        # use a fixed namespace for each data-source to ensure inter-session consistency
        namespace = uuid.UUID(str(source_seed))
        uid = str(uuid.uuid5(namespace, self.srcURL))
        self.cacheFile = osp.join(cache_dir, f'{uid}.{cache_type}.json')
        self.hashAlgo = algo
        # first comparison needs
        self.prevSrcHash = load_json(self.cacheFile).get('hash') if osp.isfile(self.cacheFile) else None

    def retrieve(self):
        if self._compare_hash():
            return self.update()
        return load_json(self.cacheFile)['data']

    def update(self):
        """
        - update cache directly
        - useful when app needs to force update cache
        """
        data = self.retriever(self.srcURL)
        container = {
            'data': data,
            'hash': self.prevSrcHash,
        }
        save_json(self.cacheFile, container)
        return data

    def _compare_hash(self):
        in_src_hash = self._compute_hash()
        if changed := in_src_hash != self.prevSrcHash or self.prevSrcHash is None:
            self.prevSrcHash = in_src_hash
        return changed

    def _compute_hash(self):
        hash_algo_map = {
            'checksum': self._compute_hash_as_checksum,
            'mtime': self._compute_hash_as_modified_time,
        }
        return hash_algo_map[self.hashAlgo]()

    def _compute_hash_as_checksum(self):
        return get_md5_checksum(self.srcURL)

    def _compute_hash_as_modified_time(self):
        try:
            return osp.getmtime(self.srcURL)
        except FileNotFoundError:
            return None

# endregion


# region functions

def get_platform_home_dir():
    home_envvar = 'USERPROFILE' if PLATFORM == 'Windows' else 'HOME'
    return os.getenv(home_envvar)


def get_platform_appdata_dir(winroam=True):
    plat_dir_map = {
        'Windows': os.getenv('APPDATA' if winroam else 'LOCALAPPDATA'),
        'Darwin': osp.expanduser('~/Library/Application Support'),
        'Linux': osp.expanduser('~/.config')
    }
    return plat_dir_map.get(PLATFORM)


def get_posix_shell_cfgfile():
    return os.path.expanduser('~/.bash_profile' if os.getenv('SHELL') == '/bin/bash' else '~/.zshrc')


def build_default_logger(logdir, name=None, verbose=False, use_rotating_handler=True, max_mb=5, backup_count=5):
    """
    Create logger sharing global logging config except log file path
    - 'filename' in config is a filename; must prepend folder path to it.
    - name is log-id in config, and will get overwritten by subsequent in-process calls; THEREFORE, never build logger with the same name twice!

    Parameters:
    - logdir: Directory to store log files
    - name: Logger name, defaults to directory basename
    - verbose: Whether to enable verbose logging
    - use_rotating_handler: Whether to use RotatingFileHandler to limit log file size
    - max_mb: Maximum size of log file before rotation in MB
    - backup_count: Number of backup files to keep
    """
    os.makedirs(logdir, exist_ok=True)
    filename = name or osp.basename(osp.basename(logdir.strip('\\/')))
    log_path = osp.join(logdir, f'{filename}.log')
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "info_lpf": {
                "()": "xypyutil.LowPassLogFilter",
                "level": 10 if verbose else 20,
            },
            "info_bpf": {
                "()": "xypyutil.BandPassLogFilter",
                "levelbounds": [10, 20] if verbose else [20, 20],
            },
            "warn_hpf": {
                "()": "xypyutil.HighPassLogFilter",
                "level": 30
            }
        },
        "formatters": {
            "console": {
                "format": "%(asctime)s: %(levelname)s: %(module)s: %(lineno)d: \n%(message)s\n"
            },
            "file": {
                "format": "%(asctime)s: %(levelname)s: %(pathname)s: %(lineno)d: \n%(message)s\n"
            }
        },
        "handlers": {
            "console": {
                "level": "DEBUG" if verbose else "INFO",
                "formatter": "console",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["info_bpf"]
            },
            "console_err": {
                "level": "WARN",
                "formatter": "console",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "filters": ["warn_hpf"]
            },
            # filename gets overwritten every call
            "file": {
                "level": "DEBUG",
                "formatter": "file",
                "class": "logging.handlers.RotatingFileHandler" if use_rotating_handler else "logging.FileHandler",
                "encoding": "utf-8",
                "filename": log_path,
                "maxBytes": max_mb * 1024 * 1024 if use_rotating_handler else 0,
                "backupCount": backup_count if use_rotating_handler else 0
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "console_err", "file"],
                "level": "INFO",
                "propagate": True
            },
            # do not show log in parent loggers
            "default": {
                "handlers": ["console", "console_err", "file"],
                "level": "DEBUG",
                "propagate": False
            }
        }
    }
    if name:
        logging_config['loggers'][name] = logging_config['loggers']['default']
    logging.config.dictConfig(logging_config)
    return logging.getLogger(name or 'default')


def find_log_path(logger):
    return next((handler.baseFilename for handler in logger.handlers if isinstance(handler, logging.FileHandler)), None)


glogger = build_default_logger(logdir=osp.join(get_platform_tmp_dir(), '_util'), name='util', verbose=True)
glogger.setLevel(logging.DEBUG)


def catch_unknown_exception(exc_type, exc_value, exc_traceback):
    """Global exception to handle uncaught exceptions"""
    exc_info = exc_type, exc_value, exc_traceback
    glogger.error('Unhandled exception: ', exc_info=exc_info)
    # _logger.exception('Unhandled exception: ')  # try-except block only.
    # sys.__excepthook__(*exc_info)  # Keep commented out to avoid msg dup.


sys.excepthook = catch_unknown_exception


def format_brief(title='', bullets=()):
    """
    - create readable brief:
      title:
      - bullet 1
      - bullet 2
    - indent: 1 indent = 2 spaces
    - if brief is nested, we don't add bullet to title for simplicity
    """
    ttl = title if title else ''
    if not bullets:
        return ttl
    lst = '\n'.join([f'- {p}' for p in bullets])
    return f"""{ttl}:
{lst}""" if ttl else lst


def indent_lines(lines, indent=1):
    """
    - indent: 1 indent = 2 spaces
    """
    return [f'{"  " * indent}{line}' for line in lines]


def format_log(situation, detail=None, advice=None, reso=None):
    """
    generic log message for all error levels
    - situation: one sentence about what happened (e.g. 'file not found', 'op completed'), useful for info level if used alone
    - detail: usually a list of facts, can be list or brief
    - advice: how to fix the problem, useful for warning/error levels
    - reso: what program ends up doing to resolve a problem, useful for error level
    """
    def _lazy_create_list(atitle, content):
        # if content is a list, we first format_brief it, then indent it
        if isinstance(content, list):
            return format_brief(atitle, content)
        # if content is a string, we indent it as lines
        le = '\n'
        return f"""\
{atitle}:
{le.join(indent_lines(content.splitlines() if isinstance(content, str) else content))}"""
    title = situation
    body = ''
    if detail is not None:
        body += f"{_lazy_create_list('Detail', detail)}\n"
    if advice is not None:
        body += f"{_lazy_create_list('Advice', advice)}\n"
    if reso is not None:
        body += f"{_lazy_create_list('Done-for-you', reso)}\n"
    if body:
        title += ':'
    return f"""\
{title}
{body}"""


def format_error(expected, got):
    """
    - indent by 1 level because titled-diagnostics usually appear below a parent title as part of a detail listing
    """
    log_expected = format_brief('Expected', expected if isinstance(expected, list) else [expected])
    log_got = format_brief('Got', got if isinstance(got, list) else [got])
    return f"""\
{log_expected}
{log_got}"""


def format_xml(elem, indent='    ', encoding='utf-8'):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    rough_string = ET.tostring(elem, encoding)
    reparsed = minidom.parseString(rough_string)
    raw = reparsed.toprettyxml(indent=indent, encoding=encoding).decode(encoding)
    lines = raw.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines)


def format_callstack():
    """
    - traceback in worker thread is hard to propagate to main thread
    - so we wrap around inspect.stack() before passing it
    """
    import inspect
    stack = inspect.stack()
    formatted_stack = []
    for frame_info in reversed(stack):  # Reverse to match traceback's order
        filename = frame_info.filename
        lineno = frame_info.lineno
        function = frame_info.function
        code_context = frame_info.code_context[0].strip() if frame_info.code_context else ''
        # Format each frame like traceback
        formatted_stack.append(f'  File "{filename}", line {lineno}, in {function}\n    {code_context}\n')
    return ''.join(formatted_stack)


def throw(err_cls, detail, advice):
    raise err_cls(f"""
{format_brief('Detail', detail if isinstance(detail, list) else [detail])}
{format_brief('Advice', advice if isinstance(advice, list) else [advice])}""")


def is_python3():
    return sys.version_info[0] > 2


def load_json(path, as_namespace=False, encoding=TXT_CODEC):
    """
    - Load Json configuration file.
    - supports UTF-8 only, due to no way to support mixed encodings
    - most usecases involve either utf-8 or mixed encodings
    - windows users must fix their region and localization setup via control panel
    """
    with open(path, 'r', encoding=encoding, errors='backslashreplace', newline=None) as f:
        text = f.read()
    return json.loads(text) if not as_namespace else json.loads(text, object_hook=lambda d: SimpleNamespace(**d))


def save_json(path, config, encoding=TXT_CODEC):
    """
    Use io.open(), aka open() with py3 to produce a file object that encodes
    Unicode as you write, then use json.dump() to write to that file.
    Validate keys to avoid JSON and program out-of-sync.
    """
    dict_config = vars(config) if isinstance(config, types.SimpleNamespace) else config
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    with open(path, 'w', encoding=encoding) as f:
        return json.dump(dict_config, f, ensure_ascii=False, indent=4)


def get_md5_checksum(file):
    """Compute md5 checksum of a file."""
    if not osp.isfile(file):
        return None
    myhash = hashlib.md5()
    with open(file, 'rb') as f:
        while True:
            b = f.read(8096)
            if not b:
                break
            myhash.update(b)
    return myhash.hexdigest()


def logcall(msg='trace', logger=glogger):
    """
    decorator for tracing app-domain function calls. Usage
        @logcall(msg='my task', logger=my_logger)
        def my_func():
           ...
    - only shows enter/exit
    - can be interrupted by exceptions
    """

    def wrap(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            logger.debug(f"Enter: '{function.__name__}' <= args={args}, kwargs={kwargs}: {msg}")
            ret = function(*args, **kwargs)
            logger.debug(f"Exit: '{function.__name__}' => {ret}")
            return ret

        return wrapper

    return wrap


def is_toplevel_function(func):
    return func.__qualname__ == func.__name__


def concur_map(worker, coll, worker_count=None, iobound=True, logger=None):
    """
    - concurrent version of builtin map()
    - due to GIL, threading is only good for io-bound tasks
    - map function interface: worker((index, elem)) -> processed_elem
    """
    if not iobound:
        assert is_toplevel_function(worker), 'must use top-level function as multiprocessing worker'
    max_workers = 10 if iobound else multiprocessing.cpu_count() - 1
    n_workers = worker_count or max_workers
    executor_class = concurrent.futures.ThreadPoolExecutor if iobound else concurrent.futures.ProcessPoolExecutor
    if logger:
        logger.debug(f'Concurrently run task: {worker.__name__} on collection, using {n_workers} {"threads" if iobound else "processes"} ...')
    with executor_class(max_workers=n_workers) as executor:
        return list(executor.map(worker, enumerate(coll)))


def profile_runs(funcname, modulefile, nruns=5, outdir=None):
    """
    - modulefile: script containing profileable wrapper functions
    - funcname: arg-less wrapper function that instantiate modules/functions as the actual profiling target
    """
    out_dir = outdir or osp.dirname(modulefile)
    module_name = osp.splitext(osp.basename(modulefile))[0]
    mod = safe_import_module(module_name, osp.dirname(modulefile))
    stats_dir = osp.join(out_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    for r in range(nruns):
        stats_file = osp.abspath(f'{stats_dir}/profile_{funcname}_{r}.pstats.log')
        profile.runctx('import {}; print({}, {}.{}())'.format(module_name, r, module_name, funcname), globals(), locals(), stats_file)
    # Read all 5 stats files into a single object
    stats = pstats.Stats(osp.abspath(f'{stats_dir}/profile_{funcname}_0.pstats.log'))
    for r in range(1, nruns):
        stats.add(osp.abspath(f'{stats_dir}/profile_{funcname}_{r}.pstats.log'))
    # Clean up filenames for the report
    stats.strip_dirs()
    # Sort the statistics by the cumulative time spent in the function
    stats.sort_stats('cumulative')
    stats.print_stats()
    return stats


def load_plist(path, binary=False):
    fmt = plistlib.FMT_BINARY if binary else plistlib.FMT_XML
    with open(path, 'rb') as fp:
        return plistlib.load(fp, fmt=fmt)


def save_plist(path, my_map, binary=False):
    fmt = plistlib.FMT_BINARY if binary else plistlib.FMT_XML
    par_dir = osp.dirname(path)
    os.makedirs(par_dir, exist_ok=True)
    with open(path, 'wb') as fp:
        plistlib.dump(my_map, fp, fmt=fmt)


def substitute_keywords_in_file(file, str_map, useliteral=False, encoding=TXT_CODEC):
    with open(file, encoding=encoding) as f:
        original = f.read()
        updated = substitute_keywords(original, str_map, useliteral)
    with open(file, 'w', encoding=encoding) as f:
        f.write(updated)


def substitute_keywords(text, str_map, useliteral=False):
    if not useliteral:
        return text % str_map
    updated = text
    for src, dest in str_map.items():
        updated = updated.replace(src, dest)
    return updated


def is_uuid(text, version: int = 0):
    try:
        uuid_obj = uuid.UUID(text)
    except ValueError:
        # not an uuid
        return False
    return uuid_obj.version == version if (any_version := version != 0) else True


def get_uuid_version(text):
    try:
        uuid_obj = uuid.UUID(text)
    except ValueError:
        # not an uuid
        return None
    return uuid_obj.version


def create_guid(uuid_version=4, uuid5_name=None):
    if uuid_version not in [1, 4, 5]:
        return None
    if uuid_version == 5:
        if uuid5_name is None:
            return None
        namespace = uuid.UUID(str(uuid.uuid4()))
        uid = uuid.uuid5(namespace, uuid5_name)
        return get_guid_from_uuid(uid)
    uid = eval(f'uuid.uuid{uuid_version}()')
    return get_guid_from_uuid(uid)


def get_guid_from_uuid(uid):
    return f'{{{str(uid).upper()}}}'


def get_clipboard_content():
    import tkinter as tk
    root = tk.Tk()
    # keep the window from showing
    root.withdraw()
    content = root.clipboard_get()
    root.quit()
    return content


def alert(content, title='Debug', action='Close'):
    """
    - on Windows, mshta msgbox does not support custom button text
    - so "action" is ignored on windows
    - multiline message uses \n as line separator
    - mshta (ms html application host) tend to open vbs using text editor; so we use dedicated vbscript cli instead
    """
    if PLATFORM == 'Windows':
        # vbs uses its own line-end
        lines = [f'"{line}"' for line in content.split('\n')]
        vbs_lines = ' & vbCrLf & '.join(lines)
        # Construct the VBScript command for the message box
        vbs_content = f'MsgBox {vbs_lines}, vbOKOnly, "{title}"'
        vbs = osp.join(get_platform_tmp_dir(), 'msg.vbs')
        save_text(vbs, vbs_content)
        cmd = ['cscript', '//Nologo', vbs]
        subprocess.run(cmd)
        os.remove(vbs)
        return cmd
    if PLATFORM == 'Darwin':
        cmd = ['osascript', '-e', f'display alert "{title}" message "{content}"']
    else:
        cmd = ['echo', f'{title}: {content}: {action}']
    return subprocess.run(cmd)


def confirm(situation, question='Do you want to proceed?', title='Question'):
    if PLATFORM == 'Windows':
        # Escaping double quotes within the VBScript
        escaped_sit = situation.replace('"', '""')
        escaped_question = question.replace('"', '""')
        escaped_title = title.replace('"', '""')

        # PowerShell command to execute VBScript code in-memory
        ps_command = (
            f"$wshell = New-Object -ComObject WScript.Shell; "
            f"$result = $wshell.Popup(\"{escaped_sit}\n\n{escaped_question}\", 0, \"{escaped_title}\", 4); "
            f"exit $result"
        )

        # Running the PowerShell command
        try:
            result = subprocess.run(["powershell", "-Command", ps_command], capture_output=True, text=True)

            # VBScript Popup returns 6 for "Yes" and 7 for "No"
            return result.returncode == 6
        except subprocess.CalledProcessError:
            return False
    elif PLATFORM == 'Darwin':
        try:
            cmd = f'osascript -e \'tell app "System Events" to display dialog "{situation}\n\n{question}" with title "{title}" buttons {{"No", "Yes"}} default button "Yes"\''
            result = subprocess.check_output(cmd, shell=True).decode().strip()
            return 'Yes' in result  # Returns True if 'Yes' was clicked
        except subprocess.CalledProcessError:
            return False  # Handle case where dialog is closed without making a selection
    else:
        # Other OS implementation (if needed)
        pass


def convert_to_wine_path(path, drive=None):
    """
    - path is a macOS-style POSIX full path, e.g.
      - ~/path/to/file
      - /path/to
    - on windows, ~/path/to/file is
    """
    full_path = osp.expanduser(path)
    assert osp.isabs(full_path), f'expected absolute paths, got: {full_path}'
    home_folder = os.environ['USERPROFILE'] if PLATFORM == 'Windows' else os.environ['HOME']
    if leading_homefolder := full_path.startswith(home_folder):
        mapped_drive = drive or 'Y:'
        full_path = full_path.removeprefix(home_folder)
    else:
        mapped_drive = drive or 'Z:'
    full_path = full_path.replace('/', '\\')
    return mapped_drive + full_path


def convert_from_wine_path(path):
    """
    - on windows, we still expand to macOS virtual drive letters
    - use POSIX path separator /
    """
    path = path.strip()
    if path.startswith('Z:') or path.startswith('z:'):
        return path[2:].replace('\\', '/') if len(path) > 2 else '/'
    elif path.startswith('Y:') or path.startswith('y:'):
        home_folder = '~/' if PLATFORM == 'Windows' else os.environ['HOME']
        return osp.join(home_folder, path[2:].replace('\\', '/').strip('/'))
    return path


def kill_process_by_name(name, forcekill=False):
    cmd_map = {
        'Windows': {
            'softKill': ['taskkill', '/IM', name],
            'hardKill': ['wmic', 'process', 'where', f"name='{name}'", 'delete'],
        },
        "*": {
            'softKill': ['pkill', name],
            'hardKill': ['pkill', '-9', name],
        }
    }
    return_codes = {
        'success': 0,
        'procNotFound': 1,
        'permissionDenied': 2,
        'unknownError': 3,  # triggered when softkilling a system process
    }
    plat = PLATFORM if PLATFORM in cmd_map else '*'
    cmd = cmd_map[plat]['hardKill'] if forcekill else cmd_map[plat]['softKill']
    if plat == '*':
        proc = run_cmd(["pgrep", "-x", name], check=False)
        if proc.returncode != 0:
            return return_codes['procNotFound']
        proc = run_cmd(cmd, check=False)
        if 'not permitted' in (err_log := safe_decode_bytes(proc.stderr).lower()):
            return return_codes['permissionDenied']
        if err_log:
            return return_codes['unknownError']
        return return_codes['success']
    # Windows: wmic cmd can kill admin-level process
    cmd = cmd_map[plat]['hardKill'] if forcekill else cmd_map[plat]['softKill']
    proc = run_cmd(cmd, check=False)
    if 'not found' in (err_log := safe_decode_bytes(proc.stderr).lower()) or 'no instance' in safe_decode_bytes(proc.stdout).lower():
        return return_codes['procNotFound']
    if 'denied' in err_log:
        return return_codes['permissionDenied']
    if proc.returncode != 0:
        return return_codes['unknownError']
    return return_codes['success']


def init_translator(localedir, domain='all', langs=None):
    """
    - select locale and set up translator based on system language
    - the leading language in langs, if any, is selected to override current locale
    - locale code differs b/w Windows and macOS
      - Windows: refer to: https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/available-language-packs-for-windows?view=windows-11
      - macOS: same as gettext
    """

    def _get_locale_code(loc):
        locale_code_map = {
            'Chinese (Simplified)_China': 'zh_CN',
            'Chinese (Traditional)_Taiwan': 'zh_TW',
            'English_United States': 'en_US',
        }
        return locale_code_map.get(loc, 'en_US')

    if langs:
        cur_langs = langs
    else:
        cur_locale, encoding = locale.getlocale()
        if PLATFORM == 'Windows':
            cur_locale = _get_locale_code(cur_locale)
        cur_langs = [cur_locale] if cur_locale else ['en']
    try:
        translator = gettext.translation(domain, localedir=localedir, languages=cur_langs)
        translator.install()
        trans = translator.gettext
    except FileNotFoundError as e:
        # No translation files found for domain. 
        # Ignore this message if called for the first time.
        trans = str

    return trans


def match_files_except_lines(file1, file2, excluded=None):
    with open(file1) as fp:
        content1 = fp.readlines()
    with open(file2) as fp:
        content2 = fp.readlines()
    if excluded:
        excluded = [excluded] if isinstance(excluded, int) else excluded
        content1 = [cont for c, cont in enumerate(content1) if c not in excluded]
        content2 = [cont for c, cont in enumerate(content2) if c not in excluded]
    return content1 == content2


def rerun_lock(name, folder=None, logger=glogger, max_instances=1, max_retries=3, retrieve_delay_sec=0.1):
    """Decorator for reentrance locking on functions"""

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            my_lock = None
            try:
                my_lock = RerunLock(name, folder, logger, max_instances, max_retries, retrieve_delay_sec)
                if not my_lock.lock():
                    return 1
                try:
                    ret = f(*args, **kwargs)
                except KeyboardInterrupt as e:
                    my_lock.unlock()
                    raise e
                my_lock.unlock()
            except Exception as e:
                if my_lock:
                    my_lock.unlock()
                # leave exception to its upper handler or let the program crash
                raise e
            return ret

        return wrapper

    return decorator


def is_pid_running(pid):
    """
    Determines if a process with the specified PID is currently running.

    This function checks for the existence of a process given its process ID (PID).
    - On Windows, it utilizes the `tasklist` command to verify if the process is active
    - On Unix/Linux, it uses the `os.kill` method with signal 0 to check for the process's existence

    Parameters:
    - pid (int): The process ID to check

    Returns:
    - bool: True if the process is running, False otherwise.
    """
    if platform.system() == 'Windows':
        try:
            # Use tasklist command to find the process
            # /FI filter condition "PID eq <pid>" to only look for specific PID
            # /NH no header line
            # /FO CSV output in CSV format for easier parsing
            output = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}', '/NH', '/FO', 'CSV'],
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True)
            # If the output contains the PID, the process exists
            return str(pid) in output
        except (subprocess.SubprocessError, OSError):
            # Command execution failed, assume process doesn't exist
            return False
    
    # Linux / macOS, send signal 0 to check if process exists
    try:
        # os.kill(pid, 0) doesn't actually kill the process but sends a harmless signal
        # This will throw an OSError exception if the PID is not running, and do nothing otherwise
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def await_while(condition, timeout_ms, step_ms=10):
    """
    - condition must implement methods:
      - .met()
      - .update()
    - under concurrency, caller must ensure thread/process safety, e.g., implement locking in condition
    """
    waited_ms = 0
    while waited_ms < timeout_ms and not condition.met():
        condition.update()
        time.sleep(step_ms / 1000)
        waited_ms += step_ms
    return waited_ms < timeout_ms


def await_lockfile(lockpath, until_gone=True, timeout_ms=float('inf'), step_ms=10):
    """
    - while lockfile exists, we wait
    """

    class UntilLockGoneCondition:
        def __init__(self, path):
            self.path = path

        def met(self):
            return not osp.exists(self.path)

        def update(self):
            pass

    class UntilLockAppearCondition:
        def __init__(self, path):
            self.path = path

        def met(self):
            return osp.exists(self.path)

        def update(self):
            pass

    condition_cls = UntilLockGoneCondition if until_gone else UntilLockAppearCondition
    return await_while(condition_cls(lockpath), timeout_ms, step_ms)


def append_to_os_paths(bindir, usesyspath=True, inmemonly=False):
    """
    On macOS, PATH update will only take effect after calling `source ~/.bash_profile` directly in shell. It won't work 
    """
    path_var = 'PATH'
    os.environ[path_var] += os.pathsep + bindir
    if inmemonly:
        return os.environ[path_var]
    if PLATFORM == 'Windows':
        root_key = 'HKEY_LOCAL_MACHINE' if usesyspath else 'HKEY_CURRENT_USER'
        full_key = f'{root_key}\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'
        env_paths = load_winreg_record(full_key, path_var)
        # SoC: here bindir must be a newcomer
        if env_paths[-1] != os.pathsep:
            env_paths += os.pathsep
        env_paths += f'{bindir}{os.pathsep}'
        save_winreg_record(full_key, path_var, env_paths)
        return os.environ[path_var]
    # posix
    cfg_file = get_posix_shell_cfgfile()
    save_lines(cfg_file, [
        '',
        f'export {path_var}="${path_var}:{bindir}"',
        '',
    ], toappend=True, addlineend=True)
    return os.environ[path_var]


def prepend_to_os_paths(bindir, usesyspath=True, inmemonly=False):
    path_var = 'PATH'
    os.environ[path_var] = bindir + os.pathsep + os.environ[path_var]
    if inmemonly:
        return os.environ[path_var]
    if PLATFORM == 'Windows':
        root_key = 'HKEY_LOCAL_MACHINE' if usesyspath else 'HKEY_CURRENT_USER'
        full_key = f'{root_key}\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'
        env_paths = load_winreg_record(full_key, path_var)
        # SoC: here bindir must be a newcomer
        env_paths = f'{bindir}{os.pathsep}{env_paths}'
        save_winreg_record(full_key, path_var, env_paths)
        return os.environ[path_var]
    # posix
    cfg_file = get_posix_shell_cfgfile()
    save_lines(cfg_file, [
        '',
        f'export {path_var}="{bindir}:${path_var}"',
        '',
    ], toappend=True, addlineend=True)
    return os.environ[path_var]


def remove_from_os_paths(bindir, usesyspath=True, inmemonly=False):
    """
    - on windows, bindir is missing from PATH before restarting explorer if freshly added
    - so we don't lazy remove to avoid miss
    """
    path_var = 'PATH'
    os.environ[path_var] = os.environ[path_var].replace(bindir, '')
    if inmemonly:
        return os.environ[path_var]
    if PLATFORM == 'Windows':
        root_key = 'HKEY_LOCAL_MACHINE' if usesyspath else 'HKEY_CURRENT_USER'
        full_key = f'{root_key}\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'
        env_paths = load_winreg_record(full_key, path_var)
        keepers = [path for path in env_paths.split(os.pathsep) if path.lower() != bindir.lower()]
        env_paths = os.pathsep.join(keepers)
        save_winreg_record(full_key, path_var, env_paths)
        return os.environ[path_var]
    cfg_file = get_posix_shell_cfgfile()
    # escape to handle metachars
    pattern_prepend = f'export {path_var}="{bindir}:${path_var}"'
    pattern_append = f'export {path_var}="${path_var}:{bindir}"'
    str_map = {
        pattern_append: '',
        pattern_prepend: '',
    }
    substitute_keywords_in_file(cfg_file, str_map, useliteral=True)
    return os.environ[path_var]


def load_winreg_record(full_key, var):
    """
    - windows registry record path
      - root_key/sub_key/var
    - input should be the full path to the key, and the variable name
    - support posix-style path
    """
    reg_key = normalize_path(full_key, 'win')
    key_comps = reg_key.split(os.sep)
    root_key = getattr(winreg, key_comps[0])
    sub_key = os.sep.join(key_comps[1:])
    with winreg.ConnectRegistry(None, root_key):
        with winreg.OpenKey(root_key, sub_key, 0, winreg.KEY_ALL_ACCESS) as key:
            value, _ = winreg.QueryValueEx(key, var)
    return value


def save_winreg_record(full_key, var, value, value_type=winreg.REG_EXPAND_SZ if PLATFORM == 'Windows' else None):
    """
    refer value_type to: https://docs.python.org/3/library/winreg.html#value-types
    """
    reg_key = normalize_path(full_key, 'win')
    key_comps = reg_key.split(os.sep)
    root_key = getattr(winreg, key_comps[0])
    sub_key = os.sep.join(key_comps[1:])
    with winreg.ConnectRegistry(None, root_key):
        with winreg.OpenKey(root_key, sub_key, 0, winreg.KEY_ALL_ACCESS) as key:
            winreg.SetValueEx(key, var, 0, value_type, value)


def run_cmd(cmd, cwd=None, logger=None, check=True, shell=False, verbose=False, useexception=True, env=None, hidedoswin=True):
    """
    - Use shell==True with autotools where new shell is needed to treat the entire command option sequence as a command,
    e.g., shell=True means running sh -c ./configure CFLAGS="..."
    - we do not use check=False to supress exception because that'd leave app no way to tell if child-proc succeeded or not
    - instead, we catch CallProcessError but avoid rethrow, and then return error code and other key diagnostics to app
    - allow user to input non-str options, e.g., int, bool, etc., and auto-convert to str for subprocess
    """
    cmd = [comp if isinstance(comp, str) else str(comp) for comp in cmd]
    logger = logger or glogger
    console_info = logger.info if logger and verbose else logger.debug
    # show cmdline with or without exceptions
    cmd_log = f"""\
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
"""
    logger.info(cmd_log)
    try:
        if hidedoswin and PLATFORM == 'Windows':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            proc = subprocess.run(cmd, check=check, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env, startupinfo=startupinfo)
        else:
            proc = subprocess.run(cmd, check=check, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env)
        stdout_log = safe_decode_bytes(proc.stdout)
        stderr_log = safe_decode_bytes(proc.stderr)
        if stdout_log:
            console_info(f'stdout:\n{stdout_log}')
        if stderr_log:
            logger.error(f'stderr:\n{stderr_log}')
    # subprocess started but failed halfway: check=True, proc returns non-zero
    # won't trigger this exception when useexception=True
    except subprocess.CalledProcessError as e:
        # generic error, grandchild_cmd error with noexception enabled
        stdout_log = f'stdout:\n{safe_decode_bytes(e.stdout)}'
        stderr_log = f'stderr:\n{safe_decode_bytes(e.stderr)}'
        logger.info(stdout_log)
        logger.error(stderr_log)
        if useexception:
            raise e
        return types.SimpleNamespace(returncode=1, stdout=e.stdout, stderr=e.stderr)
    # subprocess fails to start
    except Exception as e:
        # cmd missing ...FileNotFound
        # PermissionError, OSError, TimeoutExpired
        logger.error(e)
        if useexception:
            raise e
        return types.SimpleNamespace(returncode=2, stdout='', stderr=safe_encode_text(str(e), encoding=LOCALE_CODEC))
    return proc


def run_daemon(cmd, cwd=None, logger=None, shell=False, useexception=True, env=None, hidedoswin=True):
    """
    - if returned proc is None, means
    """
    cmd = [comp if isinstance(comp, str) else str(comp) for comp in cmd]
    logger = logger or glogger
    logger.debug(f"""run in background:
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
""")
    # fake the same proc interface
    proc = None
    try:
        if hidedoswin and PLATFORM == 'Windows':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env, startupinfo=startupinfo)
        else:
            proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env)
        # won't be able to retrieve log from background
    # subprocess fails to start
    except Exception as e:
        # cmd missing ...FileNotFound
        # PermissionError, OSError, TimeoutExpired
        logger.error(e)
        if useexception:
            raise e
        return types.SimpleNamespace(returncode=2, stdout='', stderr=safe_encode_text(str(e), encoding=LOCALE_CODEC))
    return proc


def watch_cmd(cmd, cwd=None, logger=None, shell=False, verbose=False, useexception=True, prompt=None, timeout=None, env=None, hidedoswin=True):
    """
    realtime output
    """

    def read_stream(stream, output_queue):
        for line in iter(stream.readline, b''):
            output_queue.put(line)
    cmd = [comp if isinstance(comp, str) else str(comp) for comp in cmd]
    logger = logger or glogger
    # show cmdline with or without exceptions
    cmd_log = f"""\
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
"""
    logger.info(cmd_log)
    try:
        if hidedoswin and PLATFORM == 'Windows':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # Start the subprocess with the slave ends as its stdout and stderr
            proc = subprocess.Popen(cmd, cwd=cwd, shell=shell, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, startupinfo=startupinfo)
        else:
            proc = subprocess.Popen(cmd, cwd=cwd, shell=shell, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout_queue, stderr_queue = queue.Queue(), queue.Queue()
        # Start separate threads to read from stdout and stderr
        stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_queue))
        stdout_thread.start()
        stderr_thread.start()
        res_stdout = []
        res_stderr = []
        # Read and print stdout and stderr in real-time
        while True:
            try:
                stdout_line = safe_decode_bytes(stdout_queue.get_nowait())
                res_stdout.append(stdout_line)
                sys.stdout.write(stdout_line)
                sys.stdout.flush()
            except queue.Empty:
                pass
            try:
                stderr_line = safe_decode_bytes(stderr_queue.get_nowait())
                res_stderr.append(stderr_line)
                sys.stderr.write(stderr_line)
                sys.stderr.flush()
            except queue.Empty:
                pass
            if proc.poll() is not None and stdout_queue.empty() and stderr_queue.empty():
                break
        # Wait for the threads to finish
        stdout_thread.join()
        stderr_thread.join()
        # both are empty at this point
        stdout, stderr = proc.communicate()
        proc.stdout, proc.stderr = safe_encode_text(''.join(res_stdout), encoding=LOCALE_CODEC), safe_encode_text(''.join(res_stderr), encoding=LOCALE_CODEC)
        return proc
    # subprocess fails to start
    except Exception as e:
        # no need to have header, exception has it all
        logger.error(e)
        if useexception:
            raise e
        return types.SimpleNamespace(returncode=2, stdout='', stderr=safe_encode_text(str(e), encoding=LOCALE_CODEC))


def extract_call_args(file, caller, callee):
    """
    - only support literal args
    - will throw if an arg value is a function call itself
    """

    def _get_arg_value(argument):
        """
        kwarg.value is arg
        """
        if isinstance(argument, ast.Constant):
            return argument.value
        elif negative_num := isinstance(argument, ast.UnaryOp) and isinstance(argument.op, ast.USub):
            return -argument.operand.value
        elif func_cls_name := isinstance(argument, ast.Name):
            return argument.id
        elif isinstance(argument, (ast.List, ast.Tuple)):
            return [elem.value if isinstance(elem, ast.Constant) else None for elem in argument.elts]
        elif use_type_map := isinstance(argument, ast.Attribute):
            return argument.attr
        elif use_typed_list := isinstance(argument, ast.Subscript):
            coll_type = argument.value.id
            elem_type = argument.slice.id if coll_type.startswith('list') or coll_type.startswith('tuple') else argument.slice.dims[0].id
            return f'{coll_type}[{elem_type}]'
        glogger.error(f'Unsupported syntax node: {argument}. Will fallback to None.')
        return None

    def _extract_caller_def(cls, func):
        if not cls:
            return next((node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == func), None)
        class_def = next((node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == cls), None)
        return next((node for node in class_def.body if isinstance(node, ast.FunctionDef) and node.name == func), None)

    import inspect
    mod_name = osp.splitext(osp.basename(file))[0]
    mod = safe_import_module(mod_name, osp.dirname(file))
    parsed = ast.parse(inspect.getsource(mod))
    # caller can be class.method or function
    spl = caller.split('.')
    if len(spl) > 1:
        class_name, func_name = spl[0], spl[1]
    else:
        class_name, func_name = None, spl[0]
    caller_def = _extract_caller_def(class_name, func_name)
    # lineno, args, keywords
    if not caller_def:
        return None, None
    func_method_calls = [def_node for def_node in caller_def.body if 'value' in dir(def_node) and isinstance(def_node.value, ast.Call)]
    raw_calls = {
        'func': [def_node for def_node in func_method_calls if isinstance(def_node.value.func, ast.Name) and def_node.value.func.id == callee],
        'method': [def_node for def_node in func_method_calls if isinstance(def_node.value.func, ast.Attribute) and def_node.value.func.attr == callee]
    }
    # collect args and kwargs: lineno, args, keywords
    calls = {
        'func': [],
        'method': [],
    }
    for calltype, rc in raw_calls.items():
        for call in rc:
            record = {
                'args': [_get_arg_value(arg) for arg in call.value.args],
                'kwargs': {kw.arg: _get_arg_value(kw.value) for kw in call.value.keywords},
                'lineno': call.lineno,
                'end_lineno': call.end_lineno,
            }
            calls[calltype].append(record)
    if mod_name in sys.modules:
        sys.modules.pop(mod_name)
    return calls['func'], calls['method']


def extract_class_attributes(file, classname):
    """
    assume
    - class is defined at the top-level of source file
    - all attributes are defined in constructor
    - all assignments must be about attributes, no local variables are allowed
    - attributes can use type-annotated assignments (taa)
    - builtin types of attributes without taa can be inferred from constant values
    - type-annotation can use built-in primitive types, typed-collection, and typing.TypeVar
    """

    def _get_attr_by_type(node):
        is_type_annotated = isinstance(node, ast.AnnAssign)
        is_assigned_with_const = isinstance(node.value, ast.Constant)
        is_assigned_with_seq = isinstance(node.value, (ast.List, ast.Tuple))
        is_typed_coll = is_type_annotated and isinstance(node.annotation, ast.Subscript) and not isinstance(node.annotation.value, ast.Attribute)
        use_typemap = is_type_annotated and isinstance(node.annotation, ast.Attribute)
        if use_typemap:
            attr_type = node.annotation.attr
        elif is_typed_coll:
            coll_type = node.annotation.value.id
            elem_type = node.annotation.slice.id if coll_type.startswith('list') or coll_type.startswith('tuple') else node.annotation.slice.dims[0].id
            attr_type = f'{coll_type}[{elem_type}]'
        elif is_type_annotated:
            if is_builtin_type := isinstance(node.annotation, ast.Name):
                attr_type = node.annotation.id
            else:  # NoneType
                attr_type = None
        elif is_assigned_with_const:
            # without type annotation, infer from default
            attr_type = type(node.value.value).__name__
        else:  # assigned with variable (non-constant)
            attr_type = None
        if not is_typed_coll and is_assigned_with_seq:
            attr_type = 'list' if isinstance(node.value, ast.List) else 'tuple'
        # only support constants inside list
        # non-consts are taken as None
        attr_value = node.value.value if is_assigned_with_const \
            else [elem.value if isinstance(elem, ast.Constant) else None for elem in node.value.elts] if is_assigned_with_seq \
            else None
        if coll_type_value_mismatch := attr_type == 'tuple' and isinstance(attr_value, list):
            attr_value = tuple(attr_value)
        # CAUTION: must not merge with type above, but as 2nd-pass after value-fix
        if can_infer_elem_type := not is_typed_coll and is_assigned_with_seq and len(attr_value) > 0:
            elem_type = type(attr_value[0]).__name__
            attr_type = f'{attr_type}[{elem_type}]'
        return attr_type, attr_value

    import inspect
    mod_name = osp.splitext(osp.basename(file))[0]
    mod = safe_import_module(mod_name, osp.dirname(file))
    parsed = ast.parse(inspect.getsource(mod))

    class_node = next((node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == classname), None)
    if not class_node:
        return None
    ctor = next((node for node in ast.walk(class_node) if isinstance(node, ast.FunctionDef) and node.name == '__init__'), None)
    if not ctor:
        return None
    # parse ctor
    attrib_names = [node.attr for node in ast.walk(ctor) if isinstance(node, ast.Attribute) and node.value.id == 'self']
    assigns = [node for node in ast.walk(ctor) if isinstance(node, (ast.AnnAssign, ast.Assign))]
    dtypes, values, linenos, end_linenos = [], [], [], []
    for node in assigns:
        atype, avalue = _get_attr_by_type(node)
        dtypes.append(atype)
        values.append(avalue)
        linenos.append(node.lineno)
        end_linenos.append(node.end_lineno)
    attributes = [{'name': n, 'type': t, 'default': v, 'lineno': l, 'end_lineno': e} for n, t, v, l, e in zip(attrib_names, dtypes, values, linenos, end_linenos)]
    if mod_name in sys.modules:
        sys.modules.pop(mod_name)
    return attributes


def extract_local_var_assignments(file, caller, varname):
    """
    - only support literal assignments (var_name = literal_value)
    """
    import inspect
    mod_name = osp.splitext(osp.basename(file))[0]
    mod = safe_import_module(mod_name, osp.dirname(file))
    parsed = ast.parse(inspect.getsource(mod))

    caller_def = next((node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == caller), None)
    if not caller_def:
        return None
    single_literal_assigns = [node for node in ast.walk(caller_def) if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Constant)]
    assigns = [{
        'name': varname,
        'lineno': node.lineno,
        'end_lineno': node.end_lineno,
        'value': node.value.value,
    } for node in single_literal_assigns
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == varname
    ]
    if mod_name in sys.modules:
        sys.modules.pop(mod_name)
    return assigns


def extract_imported_modules(file):
    def _get_import_aliases(importmod):
        return [alias.name for alias in importmod.names]

    def _extract_from_module_import(modnode):
        return modnode.module

    imported = []
    import inspect
    mod_name = osp.splitext(osp.basename(file))[0]
    mod = safe_import_module(mod_name, osp.dirname(file))
    parsed = ast.parse(inspect.getsource(mod))
    for node in ast.walk(parsed):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Import):
            extracted = _get_import_aliases(node)
            if extracted:
                imported += extracted
            continue
        extracted = _extract_from_module_import(node)
        if extracted:
            imported.append(extracted)
    if mod_name in sys.modules:
        sys.modules.pop(mod_name)
    return sorted(list(set(imported)))


def extract_sourcecode_comments(file):
    """
    comments = {"(row, col)": "# ...."}
    """
    with open(file) as fp:
        comments = {str(start): tok for toktype, tok, start, end, line in tokenize.generate_tokens(fp.readline) if toktype == tokenize.COMMENT}
    return comments


def find_first_line_in_range(lines, keyword, linerange=(0,), algo='startswith'):
    if not isinstance(lines, list):
        raise TypeError('lines must be list[str]')
    is_bandpass = len(linerange) > 1
    if is_bandpass:
        assert linerange[1] > linerange[0]
    criteria = {
        'startswith': lambda l, k: l.strip().startswith(k),
        'endswith': lambda l, k: l.strip().endswith(k),
        'contains': lambda l, k: k in l,
    }
    subs = lines[linerange[0]: linerange[1]] if is_bandpass else lines[linerange[0]:]
    lineno_in_between = next((ll for ll, line in enumerate(subs) if criteria[algo](line, keyword)), None)
    return lineno_in_between + linerange[0] if lineno_in_between is not None else None


def substitute_lines_between_cues(inserts, iolines, startcue, endcue, startlineno=0, removecues=False, withindent=True, useappend=False, skipdups=False):
    """
    - lazy-create list if input lines is a string (a single line)
    - smart-indent lines according to tag indentation
    - optimize with search range slicing
    - do not add lineends, treat everything as generic strings
    - returns start/end indices of inserted lines in resulted lines
    """
    inserts = [inserts] if isinstance(inserts, str) else inserts
    # slice: from offset to tail
    focus_lines = iolines[startlineno:] if startlineno > 0 else iolines
    # find range
    # always use startswith, because when we leave a cue, we want it to appear first and foremost
    startcue_ln = next((ln for ln, line in enumerate(focus_lines) if line.strip().startswith(startcue)), None)
    if startcue_ln is None:
        return None, None
    # relative to start cue
    endcue_ln = next((ln for ln, line in enumerate(focus_lines[startcue_ln:]) if line.strip().startswith(endcue)), None)
    # shift by search-start as offset
    startcue_ln += startlineno
    ins_start_ln = startcue_ln + 1
    if endcue_ln is None:
        return ins_start_ln, None
    endcue_ln += startcue_ln
    if withindent:
        # cue indents by the same amount as the followup line
        n_indent_chars = iolines[startcue_ln].find(startcue)
        indent_by_spaces = 0
        for ic in range(n_indent_chars):
            indent_by_spaces += 4 if iolines[startcue_ln][ic] == '\t' else 1
        inserts = ['{}{}'.format(' ' * indent_by_spaces, line) for line in inserts]
    # append to current content between cues or not
    lines_to_insert = iolines[ins_start_ln: endcue_ln] + inserts if useappend else inserts
    if skipdups:
        lines_to_insert = list(dict.fromkeys(lines_to_insert))
    # remove lines in b/w
    if has_lines_between_keywords := endcue_ln - startcue_ln > 1:
        rm_start_ln = startcue_ln if removecues else ins_start_ln
        rm_end_ln = endcue_ln + 1 if removecues else endcue_ln
        # CAUTION:
        # - iolines changes in-place
        del iolines[rm_start_ln: rm_end_ln]
    if removecues:
        ins_start_ln = startcue_ln
    for il in reversed(lines_to_insert):
        iolines.insert(ins_start_ln, il)
    rg_inserted = [ins_start_ln, ins_start_ln + len(lines_to_insert) - 1]
    return rg_inserted


def substitute_lines_in_file(inserts, file, startcue, endcue, startlineno=0, removecues=False, withindent=True, useappend=False, skipdups=False):
    """inserts don't have lineends"""
    lines = load_lines(file)
    rg_inserted = substitute_lines_between_cues(inserts, lines, startcue, endcue, startlineno, removecues, withindent, useappend, skipdups)
    save_lines(file, lines)
    return rg_inserted


def wrap_lines_with_tags(lines: list[str], starttag: str, endtag: str, withindent=False):
    """
    - caller must strip off line-ends beforehand
    - and add back line-ends later when needed
    """
    assert isinstance(lines, list)
    head_line, tail_line = [starttag], [endtag]
    if withindent:
        n_indent_chars = len(lines[0]) - len(lines[0].lstrip())
        head_line = [f'{" " * n_indent_chars}{starttag}']
        tail_line = [f'{" " * n_indent_chars}{endtag}']
    return head_line + lines + tail_line


def convert_compound_cases(text, style='pascal', instyle='auto'):
    def _assert_alphanumeric_hyphen_underscore(input_string):
        # Pattern to match alphanumeric characters, hyphens, and underscores
        pattern = r'^[a-zA-Z0-9-_]+$'

        # Assert that the input_string matches the pattern
        assert re.match(pattern, input_string), f'Unsupported string: {input_string}; expected alphanumeric, hyphen, and underscore characters only.'

    def _detect_casing(txt):
        case_patterns = {
            'snake': r'^[a-z][a-z0-9]*(_[a-zA_Z0-9]+)*$',
            'SNAKE': r'^[A-Z]+(_[A-Z0-9]+)*$',
            'camel': r'^[a-z]+([A-Z][a-z0-9]*)*$',
            'kebab': r'^[a-zA-Z0-9]+(-[a-zA-Z0-9]+)+$',
            'pascal': r'^[A-Z][a-z0-9]+([A-Z][a-z0-9]*)*$',
            'phrase': r'^[a-z]+( [a-z]+)*$',
            'title': r'^[A-Z][a-z]*([ ][A-Z][a-z]*)*$',
        }
        for case_style, pattern in case_patterns.items():
            if re.match(pattern, txt):
                return case_style
        glogger.warning(f'Unsupported casing: {txt}, expected: {case_patterns.keys()}; will treat as camelCase')
        return 'camel'

    assert style in ('camel', 'kebab', 'oneword', 'ONEWORD', 'pascal', 'phrase', 'snake', 'SNAKE', 'title')
    in_style = _detect_casing(text) if instyle == 'auto' else instyle
    if in_style == style:
        return text
    snake_text = text
    if in_style in ('snake', 'SNAKE'):
        snake_text = text if in_style == 'snake' else text.lower()
    elif in_style == 'kebab':
        snake_text = text.replace('-', '_')
    elif in_style in ('camel', 'pascal'):
        anchors = [c for c, char in enumerate(text) if char.isupper()]
        # prefix _ before anchors
        chars = list(text)
        for c in reversed(anchors):
            chars.insert(c, '_')
        snake_text = ''.join(chars).lstrip('_').lower()
    elif in_style in ('phrase', 'title'):
        snake_text = text.replace(' ', '_').lower()
    else:
        raise KeyError(f'Unknown input casing style: {instyle}, expected: camel, kebab, pascal, phrase, snake, SNAKE, title')
    if style == 'snake':
        return snake_text
    # convert to snake first
    if style == 'oneword':
        return snake_text.replace('_', '').lower()
    if style == 'ONEWORD':
        return snake_text.replace('_', '').upper()
    if style == 'SNAKE':
        return snake_text.upper()
    if style == 'kebab':
        return snake_text.replace('_', '-')
    split_strs = snake_text.split('_')
    if style == 'title':  # en_US => en US, this_is_title => This is Title
        return ' '.join([part[0].title() + part[1:] if part else part.title() for part in split_strs])
    if style == 'phrase':
        return ' '.join(split_strs).lower()
    out_text = [s.capitalize() for s in split_strs]
    if style == 'camel':
        out_text[0] = out_text[0].lower()
    return ''.join(out_text)


def append_lineends_to_lines(lines: list[str], style='posix'):
    """
    - because append lineend is a strong decision
    - often followed by saving-to-file action
    - the lines are not reused afterward
    - so for efficiency we allow for modifying in-lines
    """
    lineend = '\r\n' if style in ('windows', 'win') else '\n'
    return [line + lineend for line in lines]


def zip_dir(srcdir, dstbasename=None):
    """
    zip the entire folder into a zip file under the same parent folder.
    """
    src_par, src_name = osp.split(srcdir)
    dstbn = dstbasename or src_name
    out_zip = osp.join(src_par, dstbn)
    shutil.make_archive(out_zip, format='zip', root_dir=src_par, base_dir=src_name)
    return out_zip + '.zip'


def unzip_dir(srcball, destpardir=None):
    """
    - assume srcball has a top-level folder "product".
    - unzip to destpardir/product.
    - unsupported file extension will fall back to default behaviour of shutil.unpack_archive()
    """
    ext = osp.splitext(srcball)[1]
    ext_fmt_map = {
        '.zip': 'zip',
        '.tar': 'tar',
        '.xz': 'xztar',
        '.xzip': 'xztar',
        '.txz': 'xztar',
        '.gz': 'gztar',
        '.gzip': 'gztar',
        '.tgz': 'gztar',
        '.bz': 'bztar',
        '.bzip': 'bztar',
        '.tbz': 'bztar',
    }
    fmt = ext_fmt_map.get(ext)
    if destpardir is None:
        destpardir = osp.dirname(srcball)
    shutil.unpack_archive(srcball, extract_dir=destpardir, format=fmt)
    return osp.join(destpardir, osp.splitext(osp.basename(srcball))[0])


def compare_textfiles(file1, file2, showdiff=False, contextonly=True, ignoredlinenos=None, logger=None):
    """
    - ignoredlinenos: 0-based
    """
    logger = logger or glogger
    with open(file1) as fp1, open(file2) as fp2:
        lines1 = fp1.readlines()
        lines2 = fp2.readlines()
        if showdiff:
            diff_func = difflib.context_diff if contextonly else difflib.Differ().compare
            diff = diff_func(lines1, lines2)
            logger.info(f"""***
{file1} vs.
{file2}
***""")
            logger.info(''.join(diff), logger)
    if ignoredlinenos:
        lines1 = [line for ln, line in enumerate(lines1) if ln not in ignoredlinenos]
        lines2 = [line for ln, line in enumerate(lines2) if ln not in ignoredlinenos]
    return lines1 == lines2


def is_float_text(text):
    """
    - observation:
        >> float('99e-0')
        99.0
        >> float('99e-9')
        9.9e-08
        >> float('99e+9')
        99000000000.0
        >> float('99e9')
        99000000000.0
    """
    if '.' not in text and 'e' not in text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def is_number_text(text):
    """
    - Check if the string `s` is a number, including
      - Negative numbers (e.g., '-123')
      - Floats with a trailing dot (e.g., '123.')
      - Regular integers and floats
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    if not text:
        return False
    # Check for negative numbers
    if text.startswith('-'):
        text = text[1:]

    # Check for integers or floats (with a possible trailing dot)
    if text.isdigit() or (text.count('.') == 1 and text.replace('.', '', 1).isdigit()):
        return True
    return False


def is_bool_text(text):
    return text.lower() in ('true', 'false')


def create_parameter(name, default: str, val_range=None, step=0.1, precision: int = 2, delim=' '):
    """
    - a single user text input may carry polymorphic primitive data types
    - so we need to convert it to a good enough data record that a frontend can understand
    - step is for numbers only, and precision is for floats only
    - some frontend may offer fine-tuning, a good practice is to use step/10 for that, but this low-level API does not offer fine-tuning for minimalism
    - for numbers, null xrange or null component means no range limit
    - for options, range is a tuple of options; single-selection uses a literal str as default; multi-selection uses a space-separated str 'opt1 opt2'
    """
    if options := isinstance(val_range, tuple):
        assert len(val_range)
        try:
            i_default = val_range.index(default)
            # single-select
            default_opts = default
        except ValueError:
            # multi-select
            default_opts = tuple([opt.strip() for opt in default.split(delim)])
        return {'name': name, 'type': 'option', 'default': default_opts, 'range': val_range}
    if is_bool_text(default):
        return {'name': name, 'type': 'bool', 'default': default.lower() == 'true'}
    if is_number_text(default):
        # edge cases: None, (None, 1.0), (1.0, None), (None, None)
        if val_range is None:
            val_range = [float('-inf'), float('inf')]
        else:
            assert len(val_range) == 2
            if val_range[0] is None:
                val_range[0] = float('-inf')
            if val_range[1] is None:
                val_range[1] = float('inf')
        if is_float_text(default):
            val_range = [float(val_range[0]), float(val_range[1])]
            return {'name': name, 'type': 'float', 'default': float(default), 'range': val_range, 'step': step, 'precision': precision}
        val_range = [-2 ** 32 + 1 if val_range[0] in (float('-inf'), float('inf')) else int(val_range[0]), 2 ** 32 - 1 if val_range[1] in (float('inf'), float('-inf')) else int(val_range[1])]
        return {'name': name, 'type': 'int', 'default': int(default), 'range': val_range, 'step': max(int(step), 1)}
    return {'name': name, 'type': 'str', 'default': default}


def compare_dsv_lines(line1, line2, delim=' ', float_rel_tol=1e-6, float_abs_tol=1e-6, striptext=True, randomidok=False, logger=None):
    """
    - compare two lines of delimiter-separated values, with numerical and uuid comparison in mind
    - integers are compared as strings
    - strings use case-sensitive comparison
    - early-out at first mismatch
    - randomidok: if True, only compare uuid versions; accepts raw uuid and guid ({...})
    """
    logger = logger or glogger
    cmp1 = (line1.strip() if striptext else line1).split(delim)
    cmp2 = (line2.strip() if striptext else line2).split(delim)
    if len(cmp1) != len(cmp2):
        logger.error(f'number of fields mismatch: {len(cmp1)} vs. {len(cmp2)}')
        return False
    for v, (value1, value2) in enumerate(zip(cmp1, cmp2)):
        log_header = f'[Field {v}]'
        if striptext:
            value1, value2 = value1.strip(), value2.strip()
        if (v1_is_float := is_float_text(value1)) != (v2_is_float := is_float_text(value2)):
            logger.error(f'{log_header}: type mismatch, mixed float with non-float: {value1} vs. {value2}')
            return False
        if v1_is_float and v2_is_float:
            if not math.isclose(float(value1), float(value2), rel_tol=float_rel_tol, abs_tol=float_abs_tol):
                logger.error(f'{log_header}: float mismatch: {value1} vs. {value2}')
                return False
            continue
        if (uuid_ver1 := get_uuid_version(value1)) != (uuid_ver2 := get_uuid_version(value2)):
            logger.error(f'{log_header}: uuid version mismatch {uuid_ver1} vs. {uuid_ver2}: {value1} vs. {value2}')
            return False
        if both_are_uuids_and_same_versions := uuid_ver1 is not None and uuid_ver2 is not None:
            if randomidok:
                continue
            if value1 == value2:
                continue
        if value1 != value2:
            logger.error(f'{log_header}: string mismatch: {value1} vs. {value2}')
            return False
    return True


def copy_file(src, dst, isdstdir=False, keepmeta=False):
    par_dir = dst if isdstdir else osp.dirname(dst)
    os.makedirs(par_dir, exist_ok=True)
    copyfunc = shutil.copy2 if keepmeta else shutil.copy
    try:
        copyfunc(src, dst)
    except shutil.SameFileError:
        glogger.warning(f'source and destination are identical. will SKIP: {osp.abspath(src)} -> {osp.abspath(dst)}.')
    return dst if not isdstdir else osp.join(dst, osp.basename(src))


def move_file(src, dst, isdstdir=False):
    """
    - no SameFileError will be raised from shutil
    """
    par_dir = dst if isdstdir else osp.dirname(dst)
    os.makedirs(par_dir, exist_ok=True)
    try:
        shutil.move(src, dst)
    except (FileExistsError, shutil.Error) as win_err:
        glogger.debug(f'On Windows, use POSIX mv convention to overwrite existing file: {dst}')
        copy_file(src, dst, isdstdir)
        try:
            os.remove(src)
        except Exception as e:
            glogger.error(f"""Failed to remove source: {src};
- detail: {e}
- advice: manually remove source
- ignored""")
    return dst if not isdstdir else osp.join(dst, osp.basename(src))


def compare_dirs(dir1, dir2, ignoreddirpatterns=(), ignoredfilepatterns=(), showdiff=True):
    """
    - filecmp.dircmp() supports explicit name ignores only
    - this function supports glob-pattern ignores
    """

    def _collect_folders_files(my_dir):
        my_dir_contents = {
            'dirs': [],
            'files': [],
        }
        for root, folders, files in os.walk(my_dir):
            # my_dir_contents['dirs'] = {osp.join(root, folder).removeprefix(my_dir+os.sep) for folder in folders}
            # my_dir_contents['files'] = {osp.join(root, file).removeprefix(my_dir+os.sep) for file in files}

            for folder in folders:
                should_ignore_folder = next((pat for pat in ignoreddirpatterns if pat in folder), None)
                if should_ignore_folder:
                    continue
                my_dir_contents['dirs'].append(osp.join(root, folder).removeprefix(my_dir + os.sep))
            for file in files:
                should_ignore_file = next((pat for pat in ignoredfilepatterns if fnmatch.fnmatch(file, pat)), None)
                if should_ignore_file:
                    continue
                my_dir_contents['files'].append(osp.join(root, file).removeprefix(my_dir + os.sep))
        return my_dir_contents

    def _get_formatted_coll(coll):
        return pp.pformat(coll, indent=2)

    dir1_contents = _collect_folders_files(dir1)
    dir2_contents = _collect_folders_files(dir2)
    dir_names_match = dir1_contents['dirs'] == dir2_contents['dirs']
    file_names_match = dir1_contents['files'] == dir2_contents['files']
    if showdiff and not dir_names_match:
        print(f"""\
==================
folder comparison:
==================
in dir1 only:
{_get_formatted_coll(set(dir1_contents['dirs']) - set(dir2_contents['dirs']))}
vs.
in dir2 only:
{_get_formatted_coll(set(dir2_contents['dirs']) - set(dir1_contents['dirs']))}""")
    if showdiff and not file_names_match:
        print(f"""\
==================
file comparison:
==================
in dir1 only:
{_get_formatted_coll(set(dir1_contents['files']) - set(dir2_contents['files']))}
vs.
in dir2 only:
{_get_formatted_coll(set(dir2_contents['files']) - set(dir2_contents['files']))}""")
    return dir_names_match and file_names_match


def pack_obj(obj, topic=None, envelope=('<KK-ENV>', '</KK-ENV>'), classes=(), ensure_ascii=False):
    """
    for cross-language rpc only, so no need for an unpack()
    pickle is already good enough for local transmission.
    classes must be fully qualified class object, not from importlib.import_module()
    """
    if not topic:
        topic = type(obj).__name__
    msg = types.SimpleNamespace(payload=obj, topic=topic)
    if classes:
        class CustomJsonEncoder(json.JSONEncoder):
            def default(self, o):
                return o.__dict__ if isinstance(o, classes) else json.JSONEncoder.encode(self, o)

        classes = tuple([SimpleNamespace] + list(classes))
        msg_str = json.dumps(msg, cls=CustomJsonEncoder, ensure_ascii=ensure_ascii)
    else:
        msg_str = json.dumps(msg, default=lambda o: o.__dict__, ensure_ascii=ensure_ascii)

    return f'{envelope[0]}{msg_str}{envelope[1]}'


def lazy_extend_sys_path(paths):
    sys.path = list(dict.fromkeys(sys.path + paths))


def lazy_prepend_sys_path(paths):
    sys.path = list(dict.fromkeys(paths + sys.path))


def lazy_remove_from_sys_path(paths):
    for path in paths:
        try:
            sys.path.remove(path)
        except Exception as e:
            pass


def safe_import_module(modname, path=None, prepend=True, reload=False):
    """
    - importlib.reload(mod) does not work with dynamic import
    - importlib.reload(mod) expects import statement's binding with __spec__ object
    - imporlib.import_module() returns an object without __spec__
    - so we have to remove cache by hand for reloading
    """
    if path:
        path_hacker = lazy_prepend_sys_path if prepend else lazy_extend_sys_path
        path_hacker([path])
    if reload:
        try:
            del sys.modules[modname]
            glogger.debug(f'reloading: {modname}')
        except KeyError as e:
            pass
    mod = importlib.import_module(modname)
    if path:
        lazy_remove_from_sys_path([path])
    return mod


def get_parent_dirs(file_or_dir, subs=(), depth=1):
    """
    - usually file_or_dir is simply __file__
    - but __file__ is n/a when running under embedded python
    - instead, give the file's folder, i.e., osp.dirname(file_or_dir), then append this dir to result on app side
    """
    deprecate('get_parent_dirs()', 'get_ancestor_dirs()')
    script_dir = osp.abspath(osp.dirname(file_or_dir))
    par_seq = osp.normpath('../' * depth)
    root = osp.abspath(osp.join(script_dir, par_seq)) if depth > 0 else script_dir
    return script_dir, root, *[osp.join(root, sub) for sub in subs]


def get_ancestor_dirs(file_or_dir, depth=1):
    """
    - given structure: X > Y > Z > file,
    - return folder sequence: Z, Y, X
    - usually file_or_dir is simply __file__
    - but __file__ is n/a when running under embedded python
    - instead, give the file's folder, i.e., osp.dirname(file_or_dir), then append this dir to result on app side
    """
    par_dir = osp.abspath(osp.dirname(file_or_dir))
    if depth < 2:
        return [par_dir]
    # From pardir(i.e., depth 1), single-step track back up
    return [par_dir] + [osp.abspath(osp.join(par_dir, osp.normpath('../' * (dp + 1)))) for dp in range(depth - 1)]


def get_child_dirs(root, subs=()):
    return [osp.join(root, sub) for sub in subs]


def get_drivewise_commondirs(paths: list[str]):
    """
    - windows paths are drive-bound
    - paths must be either all absolute or all relative
    - on windows, we distinguish b/w UNC and relative paths
    - on posix, drive letter is always '' (empty str) for absolute and relative paths
      - so we differentiate them for clarity: '/' and ''
    - posix: common dir of below: a single drive map: {'/': 'path/to'}
      - /path/to/dir1/file1
      - /path/to/dir2/
      - /path/to/dir3/dir4/file2
    - posix: common dir of below: a single drive map: {'': 'path/to'}
      - path/to/dir1/file1
      - path/to/dir2/
      - path/to/dir3/dir4/file2
    - windows: common dirs of below: a multi-drive map: {'c:': 'c:\\path\\to', 'd:': 'd:\\path\\to', '\\\\network\\share': '\\\\network\\share\\path\\to\\dir7', '': 'path\\to\\dir9'}
      - C:\\path\\to\\dir1\\file1
      - C:\\path\\to\\dir2\\
      - D:\\path\\to\\dir3\\dir4\\file2
      - D:\\path\\to\\dir5\\dir6\\file3
      - \\\\network\\share\\path\\to\\dir7\\file4
      - \\\\network\\share\\path\\to\\dir7\\dir8\\file5
      - path\\to\\dir9\\file6
      - path\\to\\dir9\\file7
    - on windows, we normalize all paths to lowercase
    """
    if is_posix := PLATFORM != 'Windows':
        single_cm_dir = osp.commonpath(paths)
        if len(paths) == 1:
            single_cm_dir = osp.dirname(single_cm_dir)
        root = '/'
        drive = root if single_cm_dir.startswith(root) else ''
        return {drive: single_cm_dir}
    # windows
    if len(paths) == 1:
        drive, relpath = osp.splitdrive(paths[0])
        drive = drive.lower()
        single_cm_dir = osp.dirname(relpath).strip('\\').lower()
        # join('d:', 'relpath') -> 'd:relpath'
        # join('d:\\', 'relpath') -> 'd:\\relpath'
        return {drive: osp.join(drive + '\\', single_cm_dir) if drive else single_cm_dir}
    paths_sorted_by_drive = sorted(paths)
    # collect paths into map by drive
    drive_path_map = {}
    for p, winpath in enumerate(paths_sorted_by_drive):
        assert winpath, f'Invalid path at line {p}; must not be empty'
        drive, relpath = osp.splitdrive(winpath)
        drive = drive.lower()
        if must_lazy_init_for_drive := drive not in drive_path_map:
            drive_path_map[drive] = []
        drive_path_map[drive].append(relpath.strip('\\'))
    drive_relpath_map = {drive: osp.dirname(winpaths[0]).strip('\\') if (is_single_file := len(winpaths) == 1 and osp.splitdrive(winpaths[0])[1]) else osp.commonpath(winpaths).strip('\\') for drive, winpaths in drive_path_map.items()}
    return {drive: osp.join(drive + '\\', relpath).lower() if drive else relpath.lower() for drive, relpath in drive_relpath_map.items()}


def split_platform_drive(path):
    """
    - windows paths are drive-bound
    - POSIX paths are not
    - but to refer to drive-wise common dirs, we need to split the drive
    - so we define drive for POSIX in drive-wise common-dir map
      - '/' for absolute paths
      - '' for relative paths
    - on windows, convert drive letter to use lower case
    """
    if PLATFORM != 'Windows':
        drive, relpath = osp.splitdrive(path)
        drive = '/' if relpath.startswith('/') else ''
        return drive, relpath[1:] if drive else relpath
    drive, relpath = osp.splitdrive(path)
    return drive.lower(), relpath


def open_in_browser(path, window='tab', islocal=True, foreground=False):
    """
    - path must be absolute
    - widows path must be converted to posix
    - on windows, file without extension prompts for associated program; so notepad is used to avoid blocking
    """
    import webbrowser as wb
    import urllib.parse
    if no_ext_on_windows := PLATFORM == 'Windows' and not osp.splitext(path)[1]:
        open_in_editor(path, foreground)
        return path
    url_path = urllib.parse.quote(normalize_path(path, mode='posix').lstrip('/')) if islocal else path
    if islocal:
        # fix drive-letter false-alarm: 'file://C%3A/
        url_path = re.sub(r'^([A-Za-z])%3A', r'\1:', url_path)
    url = f'file:///{url_path}' if islocal else path
    api = {
        'current': wb.open,
        'tab': wb.open_new_tab,
        'window': wb.open_new,
    }
    api[window](url)
    return url


def open_in_editor(path, foreground=False):
    """
    - support both file and folder
    """
    is_extless_file = osp.isfile(path) and not osp.splitext(path)[1]
    cmds = {
        'Windows': 'notepad' if is_extless_file else 'explorer',
        'Darwin': 'open',
        'Linux': 'xdg-open',  # ubuntu
    }
    # explorer.exe only supports \
    # start.exe supports / and \, but is not an app cmd but open a prompt
    path = normalize_paths([path])[0]
    cmd = [cmds[PLATFORM], path]
    if foreground:
        run_cmd(cmd, check=PLATFORM != 'Windows', hidedoswin=False)
        return
    run_daemon(cmd, hidedoswin=False)


def flatten_nested_lists(mylist):
    return functools.reduce(operator.iconcat, mylist, [])


def show_results(succeeded, detail, advice, dryrun=False):
    banner = '** DRYRUN **' if dryrun else '*** SUCCEEDED ***' if succeeded else '* FAILED *'
    detail_title = 'Detail:'
    detail_block = detail if detail else '- (N/A)'
    advice_title = 'Next:' if succeeded else 'Advice:'
    advice_block = advice if advice else '- (N/A)'
    report = f"""
{banner}

{detail_title}
{detail_block}

{advice_title}
{advice_block}"""
    print(report)
    return report


def init_repo(srcfile_or_dir, appdepth=2, repodepth=3, organization='mycompany', logname=None, verbose=False, uselocale=False):
    """
    help source-file refer to its project-tree and utilities
    - based on a 3-level folder structure: repo > app > sub-dirs
    - sub-dirs are standard open-source folders, e.g., src, test, temp, locale, ...,
    - by default, assume srcfile is under repo > app > src
    - deeper files such as test-case files may tweak appdepth and repodepth for pointing to a correct tree-level
    - set flag uselocale to use gettext localization, by using _T() function around non-fstrings
    - set verbose to show debug log in console
    - set inter-app sharable tmp folder to platform_cache > organization > app
    """
    assert appdepth <= repodepth
    app = types.SimpleNamespace()
    app.ancestorDirs = get_ancestor_dirs(srcfile_or_dir, depth=repodepth)
    # CAUTION:
    # - do not include repo to sys path here
    # - always use lazy_extend and lazy_remove
    # just have fixed initial folders to meet most needs in core and tests
    app.locDir, app.srcDir, app.tmpDir, app.testDir = get_child_dirs(app_root := app.ancestorDirs[appdepth - 1], subs=('locale', 'src', 'temp', 'test'))
    app.pubTmpDir = osp.join(get_platform_tmp_dir(), organization, osp.basename(app_root))
    app.stem = osp.splitext(osp.basename(srcfile_or_dir))[0]
    app.logger = build_default_logger(app.tmpDir, name=logname if logname else app.stem, verbose=verbose)
    if uselocale:
        app.translator = init_translator(app.locDir)
    return app


def backup_file(file, dstdir=None, suffix='.1', keepmeta=True):
    """
    save numeric backup in dstdir or same dir
    - preserve metadata
    - always overwrite non-numeric backup
    """
    bak_dir = dstdir if dstdir else osp.dirname(file)
    bak = osp.join(bak_dir, osp.basename(file) + suffix)
    num = suffix[1:]
    if not num.isnumeric():
        copy_file(file, bak, keepmeta=keepmeta)
        return bak
    bn = osp.basename(file)
    cur_numeric_suffixes = [int(_sfx) for bkfile in glob.glob(osp.join(bak_dir, f'{bn}.*')) if (_sfx := osp.splitext(bkfile)[1][1:]).isnumeric()]
    bak = osp.join(bak_dir, f'{bn}.{max(cur_numeric_suffixes) + 1}') if cur_numeric_suffixes else osp.join(bak_dir, f'{bn}{suffix}')
    copy_file(file, bak, keepmeta=keepmeta)
    return bak


def recover_file(file, bakdir=None, suffix=None, keepmeta=True):
    """
    recover file from backup in bakdir or same dir
    - if no suffix is given, find the latest numeric backup
    """
    bak_dir = bakdir if bakdir else osp.dirname(file)
    assert osp.isdir(bak_dir)
    bn = osp.basename(file)
    baks = glob.glob(osp.join(bak_dir, f'{bn}.*'))
    if not baks:
        return None
    if suffix:
        bak = osp.join(bak_dir, bn + suffix)
        copy_file(bak, file, keepmeta=keepmeta)
        return bak
    cur_numeric_suffixes = [int(_sfx) for bkfile in glob.glob(osp.join(bak_dir, f'{bn}.*')) if (_sfx := osp.splitext(bkfile)[1][1:]).isnumeric()]
    if may_have_bak_but_not_numeric := not cur_numeric_suffixes:
        return None
    suffix = max(cur_numeric_suffixes)
    bak = osp.join(bak_dir, f'{bn}.{suffix}')
    copy_file(bak, file, keepmeta=keepmeta)
    glogger.info(f'no suffix given, recovered from latest backup: {bak}')
    return bak


def deprecate(old, new):
    msg = f'{old} is deprecated; use {new} instead'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return msg


def load_lines(path, rmlineend=False, encoding=TXT_CODEC):
    with open(path, encoding=encoding) as fp:
        lines = fp.readlines()
        if rmlineend:
            lines = [line.rstrip('\n').rstrip('\r') for line in lines]
    return lines


def save_lines(path, lines, toappend=False, addlineend=False, style='posix', encoding=TXT_CODEC):
    lines_to_write = copy.deepcopy(lines)
    if isinstance(lines, str):
        lines_to_write = [lines_to_write]
    mode = 'a' if toappend else 'w'
    if addlineend:
        line_end = '\n' if style == 'posix' else '\r\n'
        lines_to_write = [line + line_end for line in lines_to_write]
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    with open(path, mode, encoding=encoding) as fp:
        fp.writelines(lines_to_write)
    return lines_to_write


def load_text(path, encoding=TXT_CODEC):
    with open(path, encoding=encoding) as fp:
        text = fp.read()
    return text


def save_text(path, text: str, toappend=False, encoding=TXT_CODEC):
    mode = 'a' if toappend else 'w'
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    with open(path, mode, encoding=encoding) as fp:
        fp.write(text)


def find_duplication(mylist):
    """
    - find and index all duplicates in a list
      - in: [1, 2, 3, 2, 4, 1, 5]
      - out: {1: [0, 5], 2: [1, 3]}
    - useful for later conflict resolution
    """
    seen, dups = set(), set()
    for idx, item in enumerate(mylist):
        if item in seen:
            dups.add(item)
        else:
            seen.add(item)
    return {dup: [idx for idx, item in enumerate(mylist) if item == dup] for dup in dups}


def remove_duplication(mylist):
    return list(dict.fromkeys(mylist))


def find_runs(lst):
    """
    indexing contiguous elem sequences (runs)
    """
    runs = []
    cur_run = []
    for i, value in enumerate(lst):
        if i > 0 and value == lst[i - 1]:
            cur_run.append(i)
        else:  # current run ends
            if len(cur_run) > 1:
                runs.append(cur_run)
            # new candidate run
            cur_run = [i]
    # tail run
    if len(cur_run) > 1:
        runs.append(cur_run)
    return runs


def install_by_macports(pkg, lazybin=None):
    """
    Homebrew has the top priority.
    Macports only overrides in memory on demand.
    """
    macports = shutil.which('port')
    if not macports or not osp.isfile(macports):
        raise FileNotFoundError('Missing MacPorts; Retry after installing MacPorts')
    os_paths = os.environ['PATH']
    prepend_to_os_paths('/opt/local/sbin', inmemonly=True)
    prepend_to_os_paths('/opt/local/bin', inmemonly=True)
    if lazybin and (exe := shutil.which(lazybin)):
        print(f'Found binary: {exe}; skipped installing package: {pkg}')
        return exe
    run_cmd(['sudo', macports, 'install', pkg])
    os.environ['PATH'] = os_paths
    binary = pkg
    return shutil.which(binary)


def uninstall_by_macports(pkg):
    """
    Homebrew has the top priority.
    Macports only overrides in memory on demand.
    """
    macports = shutil.which('port')
    if not macports or not osp.isfile(macports):
        raise FileNotFoundError('Missing MacPorts; Retry after installing MacPorts')
    os_paths = os.environ['PATH']
    prepend_to_os_paths('/opt/local/sbin', inmemonly=True)
    prepend_to_os_paths('/opt/local/bin', inmemonly=True)
    run_cmd(['sudo', macports, 'uninstall', pkg])
    os.environ['PATH'] = os_paths


def install_by_homebrew(pkg, ver=None, lazybin=None, cask=False, buildsrc=False):
    """
    always upgrade homebrew to the latest
    """
    if lazybin and (exe := shutil.which(lazybin)):
        print(f'Found binary: {exe}, and skipped installing package: {pkg}')
        return
    pkg_version = pkg if not ver else pkg + f'@{ver}'
    cmd = ['brew', 'install']
    if cask:
        cmd += ['--cask']
    if buildsrc:
        cmd += ['--build-from-source']
    run_cmd(cmd + [pkg_version])


def uninstall_by_homebrew(pkg, lazybin=None):
    if lazybin and not shutil.which(lazybin):
        print(f'Missing binary: {lazybin}, and skipped uninstalling package: {pkg}')
        return
    run_cmd(['brew', 'remove', pkg])


def validate_platform(supported_plats):
    if isinstance(supported_plats, str):
        supported_plats = [supported_plats]
    if (plat := PLATFORM) in supported_plats:
        return
    raise NotImplementedError(f'Expected to run on {supported_plats}, but got {plat}')


def touch(file, withmtime=True):
    par_dir = osp.dirname(file)
    os.makedirs(par_dir, exist_ok=True)
    with open(file, 'a'):
        if withmtime:
            os.utime(file, None)
    return file


def lazy_load_listfile(single_or_listfile: typing.Union[str, list[str]], ext=('.list', '.csv', '.tsv')):
    """
    - we don't force return type-hint to be -> list for reusing args.path str
    - assume list can be text of any nature, i.e., not just paths
    - allow for single-item list containing a listfile so that frontend can offer a listview for this case
    """
    def _load_listfile(lst_file):
        if not osp.isfile(lst_file):
            raise FileNotFoundError(f'Missing list file: {lst_file}')
        return load_lines(lst_file, rmlineend=True)
    if isinstance(ext, str):
        ext = [ext]
    if isinstance(single_or_listfile, (list, tuple,)) and single_or_listfile:
        # e.g. [my_name]
        # e.g. [my_name.txt]
        if osp.splitext(possible_lst_file := single_or_listfile[0])[1] not in ext:
            return single_or_listfile
        # e.g. [my_name.list]
        return _load_listfile(possible_lst_file)
    if is_single_item := osp.splitext(single_or_listfile)[1] not in ext:
        # we don't care whether it exists or not
        return [single_or_listfile]
    return _load_listfile(single_or_listfile)


def normalize_path(path, mode='native'):
    if mode == 'native':
        return path.replace('/', '\\') if PLATFORM == 'Windows' else path.replace('\\', '/')
    if mode == 'posix':
        return path.replace('\\', '/')
    if mode == 'win':
        return path.replace('/', '\\')
    raise NotImplementedError(f'Unsupported path normalization mode: {mode}')


def normalize_paths(paths, mode='native'):
    """
    - modes:
      - auto: use platform pathsep
      - posix: use /
      - win: use \\
    """
    return [normalize_path(p, mode) for p in paths]


def lazy_load_filepaths(single_or_listfile: typing.Union[str, list[str]], ext=('.list', '.csv', '.tsv'), root=''):
    """
    - we don't force return type-hint to be -> list for reusing args.path str
    - listfile can have \\ or /, so can root and litfile path
    - we must normalize for file paths
    """
    def _load_listfile(lst_file, root_path):
        if not osp.isfile(lst_file):
            raise FileNotFoundError(f'Missing list file: {lst_file}')
        # native win-paths remain the same;
        # posix-format win-paths are converted to native
        paths = [osp.normpath(path) for path in load_lines(lst_file, rmlineend=True)]
        return [path if osp.isabs(path) else osp.abspath(f'{root_path}/{path}') for path in paths]
    if not single_or_listfile:
        return []
    # if not file path, then user must give root for relative paths
    root = root or os.getcwd()
    # prepare for path normalization: must input posix paths for windows
    root = normalize_path(root, mode='posix')
    if isinstance(ext, str):
        ext = [ext]
    if isinstance(single_or_listfile, (list, tuple,)) and single_or_listfile:
        # e.g. [my_name.txt]
        # e.g. [my_name.list]
        abs_file = single_or_listfile[0]
        if not osp.isabs(abs_file):
            abs_file = osp.abspath(f"{root}/{normalize_path(abs_file, mode='posix')}")
        if is_single := osp.splitext(abs_file)[1] not in ext:
            # we don't care whether it exists or not
            return single_or_listfile
        return _load_listfile(abs_file, root)
    abs_file = single_or_listfile
    if not osp.isabs(abs_file):
        abs_file = osp.abspath(f"{root}/{normalize_path(abs_file, mode='posix')}")
    if is_single := osp.splitext(abs_file)[1] not in ext:
        # we don't care whether it exists or not
        return [single_or_listfile]
    return _load_listfile(abs_file, root)


def read_link(link_path, encoding=TXT_CODEC):
    """
    cross-platform symlink/shortcut resolver
    - Windows .lnk can be a command, thus can contain source-path and arguments
    """
    if link_path.endswith('.lnk') and PLATFORM != 'Windows':
        return link_path
    try:
        # all symlinks got resolved here
        return os.readlink(link_path)
    except OSError as is_win_lnk:
        # consistent with os.readlink(symlink) on Windows
        pass
    # symlink on window
    # windows: .lnk does not resolve as a file/dir
    # macos: .lnk resolves as a file; app detects .lnk by validation: link_path == read_link(link_path)?
    if not link_path.endswith('.lnk') and osp.isfile(link_path) or osp.isdir(link_path):
        return link_path
    # windows: not a symlink or
    # because results of osp.islink(), os.readlink() always mix up with .lnk
    # get_target implementation by hannes, https://gist.github.com/Winand/997ed38269e899eb561991a0c663fa49
    ps_command = \
        "$WSShell = New-Object -ComObject Wscript.Shell;" \
        "$Shortcut = $WSShell.CreateShortcut(\"" + str(link_path) + "\"); " \
                                                                    "Write-Host $Shortcut.TargetPath ';' $shortcut.Arguments "
    output = subprocess.run(["powershell.exe", ps_command], capture_output=True)
    raw = safe_decode_bytes(output.stdout)
    src_path, args = [x.strip() for x in raw.split(';', 1)]
    return src_path


def is_link(path, encoding=TXT_CODEC):
    """
    support .lnk and symlink:
    on windows
    - osp.islink(path) always returns False
    - os.readlink(path.symlink) returns empty str ''
    - os.readlink(path) throws when link itself does not exist
    - osp.isdir(path) returns True only when linked source is an existing dir
    - os.readlink(file_under_linked) raises OSError WinError 4390
    - os.readlink(file.lnk) raises OSError WinError 4390
    - osp.isfile(file.lnk) returns True
    on mac
    - osp.islink(path.lnk) returns false
    - osp.islink(path) returns True when symlink exists
    - osp.isdir(path) / osp.exists(path) returns True only when linked source is an existing dir
    """
    if PLATFORM != 'Windows':
        return osp.islink(path)
    # windows
    # - check symlinks first, posix-based tool such as npm-link can create symlinks on windows
    # - then with .lnk
    try:
        src = os.readlink(path)
        return True
    except OSError as e:
        # not a symlink, or path does not exist, or is a file
        pass
    src = read_link(path, encoding)
    return src != '' and src != path


def raise_error(errcls, detail, advice):
    raise errcls(f"""\
Detail:
{detail}

Advice:
{advice}""")


def sanitize_text_as_path(text: str, fallback_char='_'):
    """
    text is a part of path, excluding os.sep
    """
    invalid_chars_pattern = r'[\\\/:*?"<>|\x00-\x1F]'
    return re.sub(invalid_chars_pattern, fallback_char, text)


def safe_remove(path, logger=None):
    if not osp.exists(path):
        logger = logger or glogger
        logger.debug(f'Missing file/folder: {path}; skipped removing')
        return
    if osp.isdir(path):
        remove_tree(path, safe=True)
    else:
        # no need to safe-check again
        remove_file(path, safe=False)


def remove_file(file, safe=True):
    if safe and not osp.isfile(file):
        return
    os.remove(file)


def remove_tree(root, safe=True):
    shutil.rmtree(root, ignore_errors=safe)


def is_non_ascii_text(text):
    return any(char not in string.printable for char in text)


def inspect_obj(obj):
    """
    - return dict instead pf namespace for easier data-exchange via json serialization
    """
    type_name = type(obj).__name__
    try:
        attrs = vars(obj)
    except TypeError:
        attrs = {}
    details = dir(obj)
    return {'type': type_name, 'attrs': attrs, 'repr': repr(obj), 'details': details}


def mem_caching(maxsize=None):
    """
    - per-process lru caching for multiple data sources
    - cache is outdated when process exits
    """

    def decorator(func):
        cache = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cache(*args, **kwargs)

        return wrapper

    return decorator


def find_invalid_path_chars(path, mode='native'):
    """
    - posix: / is not allowed
    - windows: "\\/:*?"<>|" are not allowed
    - consider all ascii control characters are invalid
    - result: {path_comp1: [pos1, pos2, ...], ...}
    """
    sep_map = {
        'native': os.sep,
        'posix': '/',
        'win': '\\',
        'windows': '\\',
    }
    path_comps = [pc for pc in path.split(sep_map[mode]) if pc]
    invalid_pattern = r'[\\\/:*?"<>|\x00-\x1F]'
    invalid_registry = {}
    for pc in path_comps:
        invalid_matches = re.finditer(invalid_pattern, pc)
        invalid_positions = [(match.start(), match.group()) for match in invalid_matches if match]
        if invalid_positions:
            invalid_registry[pc] = invalid_positions
    return invalid_registry


def extract_path_stem(path):
    return osp.splitext(osp.basename(path))[0]


def extract_docstring(path, target=None, encoding=TXT_CODEC, envelope=None):
    """
    - returns docstring as a single string
    - target: None means module-level docstring
    - envelope: None means .py file, otherwise, it's a non-python file that has a python-like docstring using its own comment syntax
    TODO:
    - support class/method/function-level docstrings
    """
    def _extract_nonpy_docstring(code, target, envelope):
        lines = code.splitlines()
        start_ln = find_first_line_in_range(lines, envelope)
        if start_ln is None:
            return None, None, None
        end_ln = find_first_line_in_range(lines, envelope, linerange=(start_ln+1,))
        if end_ln is None:
            return None, None, None
        return '\n'.join(code.splitlines()[start_ln+1:end_ln]), start_ln+1, end_ln-1
    code = load_text(path, encoding=encoding)
    try:
        docstring = ast.get_docstring(node := ast.parse(code))
    except SyntaxError as not_py_err:
        if not envelope:
            raise not_py_err
        return _extract_nonpy_docstring(code, target, envelope)
    if docstring is None:
        # possible cases
        # - empty file
        # - top module-level const is not a str, e.g, a number
        return None, None, None
    # The module-level docstring is stored in the `body` of the module AST as the first item,
    # if it exists. Otherwise, it's None.
    docstring_node = node.body[0] if (node.body and isinstance(node.body[0], (ast.Expr, ast.Constant))) else None
    assert docstring_node
    # found docstring candidate
    # if isinstance(docstring_node, ast.Expr) and isinstance(docstring_node.value, (ast.Str, ast.Constant)):
    start_lineno = docstring_node.lineno
    # We'll approximate end_lineno by splitting the docstring by lines and adding the count
    end_lineno = start_lineno + len(docstring.splitlines()) - 1
    return docstring, start_lineno, end_lineno


def inject_docstring(path, lines_no_lineends, target=None, encoding=TXT_CODEC, envelope=None, style='posix'):
    """
    - add docstring to top if no target is given
    TODO:
    - support class/method/function-level docstrings
    """

    if isinstance(lines_no_lineends, str):
        lines_no_lineends = [lines_no_lineends]
    doc_lines = copy.deepcopy(lines_no_lineends)
    envelope = envelope or '"""'
    doc_lines = [envelope] + doc_lines + [envelope]
    code_lines = load_lines(path, encoding=encoding, rmlineend=True)
    code_with_docstring = doc_lines + code_lines
    return save_lines(path, code_with_docstring, encoding=encoding, addlineend=True)


def load_dsv(path, delimiter=',', encoding=TXT_CODEC):
    """
    - strip of leading and trailing spaces for each row
    TODO:
    - support csv's dialect
    """
    avoid_extra_blankline_on_win = '' if PLATFORM == 'Windows' else None
    with open(path, newline=avoid_extra_blankline_on_win, encoding=encoding) as fp:
        reader = csv.reader(fp, delimiter=delimiter, skipinitialspace=True)
        rows = [row for row in reader]
    return rows


def save_dsv(path, rows, delimiter=',', encoding=TXT_CODEC):
    """
    - strip of leading and trailing spaces for each row
    """
    avoid_extra_blankline_on_win = '' if PLATFORM == 'Windows' else None
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', newline=avoid_extra_blankline_on_win, encoding=encoding) as fp:
        writer = csv.writer(fp, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)


def say(text, voice='Samantha', outfile=None):
    """
    - voice:
      - zhs: Tingting
      - zht: Meijia
      - zhc: Sinji
      - jp: Kyoko
      - kr: Yuna
      - fr_CA: Amélie
      - fr_FR: Thomas
    """
    if PLATFORM not in ['Darwin', 'Windows']:
        raise NotImplementedError(f'Unsupported platform: {PLATFORM}')
    out_file = outfile or osp.join(get_platform_tmp_dir(), '_util', 'say.wav')
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    speak_cmd = ["powershell", "-File", osp.abspath(f'{_script_dir}/xypyutil_helper/windows/kkttsspeak.ps1'), text] if PLATFORM == 'Windows' else ['say', '-v', voice, text]
    save_cmd = ["powershell", "-File", osp.abspath(f'{_script_dir}/xypyutil_helper/windows/kkttssave.ps1'), "-text", text, "-filepath", out_file] if PLATFORM == 'Windows' else ['say', '-v', voice, '-o', out_file, '--data-format', 'LEI16@48000', text]
    run_cmd(speak_cmd)
    run_cmd(save_cmd)
    return out_file


def http_get(url, encoding=TXT_CODEC):
    with urllib.request.urlopen(url) as response:
        html = response.read()
    return safe_decode_bytes(html, encoding=encoding)


def http_post(url, data: dict, encoding=TXT_CODEC):
    encoded = json.dumps(data).encode(encoding)
    req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, encoded) as response:
        resp = types.SimpleNamespace(
            status_code=response.status,
            url=response.url,
            content=response.read()
        )
    return resp


def lazy_download(local_file, url, file_open_mode='wb', logger=None):
    if osp.isfile(local_file):
        return local_file
    os.makedirs(osp.dirname(local_file), exist_ok=True)
    logger = logger or glogger
    with open(local_file, file_open_mode) as fp:
        with urllib.request.urlopen(url) as response:
            logger.info(f'Downloading: {url} => {local_file}')
            fp.write(response.read())
    return local_file


def get_environment():
    return {
        'os': platform.platform(),
        'pyExe': sys.executable,
        'pyVersion': platform.python_version(),
        'pyPath': sys.path,
        'osPath': os.environ['Path'] if PLATFORM == 'Windows' else os.environ['PATH'],
    }


def safe_decode_bytes(byte_obj, encoding=LOCALE_CODEC, errors="backslashreplace"):
    return byte_obj.decode(encoding, errors=errors)


def safe_encode_text(text, encoding=TXT_CODEC, errors="backslashreplace"):
    return text.encode(encoding, errors=errors)


def report_duration(start, end, fmt='compact'):
    duration = end - start
    days, seconds = duration.days, duration.seconds
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if fmt == 'compact':
        return f"{days}.{hours}:{minutes}:{secs}.{duration.microseconds // 1000}"
    if fmt == 'human':
        return f"{days} days, {hours} hours, {minutes} minutes, {secs} seconds, and {duration.microseconds//1000} milliseconds"
    return duration


def safe_get_element(mylist, idx, default=None):
    try:
        return mylist[idx]
    except IndexError:
        return default


def safe_index(mylist, item, default=None):
    try:
        return mylist.index(item)
    except ValueError:
        return default


def format_now(dt=datetime.datetime.now(), fmt='%Y_%m_%d-%H_%M_%S'):
    return dt.strftime(fmt)


def load_ini(path, *args, **kwargs):
    config = configparser.ConfigParser(*args, **kwargs)
    config.read(path)
    return config


def indent(code_or_lines, spaces_per_indent=4):
    lines = code_or_lines.splitlines() if isinstance(code_or_lines, str) else code_or_lines
    indented = [f'{" " * spaces_per_indent}{line}' for line in lines]
    return '\n'.join(indented) if isinstance(code_or_lines, str) else indented


def collect_file_tree(root):
    return [file for file in glob.glob(osp.join(root, '**'), recursive=True) if osp.isfile(file)]


def merge_namespaces(to_ns: types.SimpleNamespace, from_ns: types.SimpleNamespace, trim_from=False):
    """
    - merge from_ns into to_ns
    - to_ns must be a namespace
    - from_ns can be a dict or a namespace
    """
    from_ = vars(from_ns)
    to_ = vars(to_ns)
    from_keys = [key for key in from_ if key in to_] if trim_from else list(from_.keys())
    for k in from_keys:
        setattr(to_ns, k, from_[k])
    return to_ns


def thread_timeout(seconds, bypass=False):
    """
    - for single-process function only
    - will not work if decorated function spawns subprocesses
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if bypass:
                return func(*args, **kwargs)
            # Create a thread to run the function
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.start()
            # Wait for the thread to finish or timeout
            thread.join(seconds)
            if thread.is_alive():
                # Terminate the thread if it's still running
                raise TimeoutError(f"{func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator


def _run_container(func, cont_queue, args, kwargs):
    module_name, func_name = func.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    try:
        result = func(*args, **kwargs)
        cont_queue.put(result)
    except Exception as e:
        cont_queue.put(e)


def process_timeout(seconds, bypass=False):
    def decorator(func):
        func_name = f'{func.__module__}.{func.__name__}'

        def wrapper(*args, **kwargs):
            if bypass:
                return func(*args, **kwargs)
            # Create a thread to run the function
            cont_queue = multiprocessing.Queue()
            proc = multiprocessing.Process(target=_run_container, args=(func_name, cont_queue, args,), kwargs=kwargs)
            proc.start()
            try:
                result = cont_queue.get(timeout=seconds)
                return result
            except multiprocessing.queues.Empty:
                proc.terminate()
                proc.join()
                raise TimeoutError(f"{func_name} timed out after {seconds} seconds")
        return wrapper
    return decorator


def remove_unsupported_dict_keys(mydict: dict, supported_keys: set):
    # use set arithmetic to remove unsupported keys
    unsupported = set(mydict) - supported_keys
    for key in unsupported:
        mydict.pop(key)
    return mydict


def json_to_text(obj, use_unicode=True, pretty=False):
    try:
        json_str = json.dumps(obj, ensure_ascii=not use_unicode, indent=4 if pretty else None)
        return json_str, None
    except TypeError as e:
        return None, e


def json_from_text(json_str):
    try:
        data = json.loads(json_str)
        return data, None
    except json.JSONDecodeError as e:
        return None, e

# endregion


def _test():
    # print(say('hello'))
    print(create_guid())


if __name__ == '__main__':
    _test()
