#!/usr/bin/env python

"""
Utility lib for personal projects, supports py3 only.

Covering areas:
    - Logging;
    - Config save/load;
    - Decoupled parameter server-client arch;
"""

# Import std-modules.
import collections
import cProfile as profile
import concurrent.futures
import copy
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
import queue
import re
import string
import time
import tokenize
import types
import platform
import plistlib
import pprint as pp
import pstats
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
import warnings
from types import SimpleNamespace
import uuid

#
# Globals
#
_script_dir = osp.abspath(osp.dirname(__file__))
TXT_CODEC = 'utf-8'  # Importable.
LOCALE_CODEC = locale.getpreferredencoding()
MAIN_CFG_FILENAME = 'app.json'
DEFAULT_CFG_FILENAME = 'default.json'


class SingletonDecorator:
    """
    Decorator to build Singleton class, single-inheritance only.
    Usage:
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


def get_platform_home_dir():
    plat = platform.system()
    if plat == 'Windows':
        return os.getenv('USERPROFILE')
    elif plat == 'Darwin':
        return osp.expanduser('~')
    elif plat == 'Linux':
        return osp.expanduser('~')
    raise NotImplementedError(f'unsupported platform: {plat}')


def get_platform_appdata_dir(winroam=True):
    plat = platform.system()
    if plat == 'Windows':
        return os.getenv('APPDATA' if winroam else 'LOCALAPPDATA')
    elif plat == 'Darwin':
        return osp.expanduser('~/Library/Application Support')
    raise NotImplementedError(f'unsupported platform: {plat}')


def get_platform_tmp_dir():
    plat = platform.system()
    if plat == 'Windows':
        return osp.join(os.getenv('LOCALAPPDATA'), 'Temp')
    elif plat == 'Darwin':
        return osp.join(osp.expanduser('~'), 'Library', 'Caches')
    elif plat == 'Linux':
        return '/tmp'
    raise NotImplementedError(f'unsupported platform: {plat}')


def build_default_logger(logdir, name=None, verbose=False):
    """
    create logger sharing global logging config except log file path
    - 'filename' in config is a filename; must prepend folder path to it.
    - name is log-id in config, and will get overwritten by subsequent in-process calls; THEREFORE, never build logger with the same name twice!
    """
    os.makedirs(logdir, exist_ok=True)
    filename = name or osp.basename(osp.basename(logdir.strip('\\/')))
    log_path = osp.join(logdir, f'{filename}.log')
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "info_lpf": {
                "()": "kkpyutil.LowPassLogFilter",
                "level": 10 if verbose else 20,
            },
            "info_bpf": {
                "()": "kkpyutil.BandPassLogFilter",
                "levelbounds": [10, 20] if verbose else [20, 20],
            },
            "warn_hpf": {
                "()": "kkpyutil.HighPassLogFilter",
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
                "class": "logging.FileHandler",
                "encoding": "utf-8",
                "filename": log_path
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


glogger = build_default_logger(logdir=osp.join(get_platform_tmp_dir(), '_util'), name='util', verbose=False)
glogger.setLevel(logging.DEBUG)


def catch_unknown_exception(exc_type, exc_value, exc_traceback):
    """Global exception to handle uncaught exceptions"""
    exc_info = exc_type, exc_value, exc_traceback
    glogger.error('Unhandled exception: ', exc_info=exc_info)
    # _logger.exception('Unhandled exception: ')  # try-except block only.
    # sys.__excepthook__(*exc_info)  # Keep commented out to avoid msg dup.


sys.excepthook = catch_unknown_exception


def build_logger(srcpath, logpath=None):
    """
    Build per-file logger.
    :param srcpath: Path to source file.
    :param logpath: Path to log file, default to /same/dir/basename.log.
    :return: Logger object.
    """
    src_basename = osp.basename(srcpath)

    # Must have to see DEBUG/INFO at all
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(src_basename)
    logger.setLevel(logging.DEBUG)

    # Hide dependency module's logging
    logger.propagate = False

    # Avoid redundant logs from duplicated handlers created by other modules.
    if len(logger.handlers) > 1:
        return logger

    # Console log for end-users: no debug messages.
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(name)s: %(message)s')
    )
    logger.addHandler(handler)

    logpath = logpath or osp.abspath(f'{osp.dirname(srcpath)}/{osp.splitext(src_basename)[0]}.log')

    # Log file for coders: with debug messages.
    logdir = osp.abspath(osp.dirname(logpath))
    if not osp.exists(logdir):
        os.makedirs(logdir)
    handler = logging.FileHandler(logpath)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(pathname)s: %(lineno)d: %(asctime)s: \n%(message)s\n')
    )
    logger.addHandler(handler)

    return logger


def format_error_message(situation, expected, got, advice, reaction):
    return f"""\
{situation}:
- Expected: {expected}
- Got: {got}
- Advice: {advice}
- Reaction: {reaction}"""


def is_multiline_text(text, lineend='\n'):
    return len(text.strip().split(lineend)) > 1


def is_python3():
    return sys.version_info[0] > 2


def load_json(path, as_namespace=False):
    """
    - Load Json configuration file.
    - supports UTF-8 only, due to no way to support mixed encodings
    - most usecases involve either utf-8 or mixed encodings
    - windows users must fix their region and localization setup via control panel
    """
    if is_python3():
        with open(path, 'r', encoding=TXT_CODEC, errors='backslashreplace', newline=None) as f:
            text = f.read()
    else:
        with open(path, 'rU') as f:
            text = f.read()
    # Add object_pairs_hook=collections.OrderedDict hook for py3.5 and lower.
    return json.loads(text, object_pairs_hook=collections.OrderedDict) if not as_namespace else json.loads(text, object_hook=lambda d: SimpleNamespace(**d))


def save_json(path, config):
    """
    Use io.open(), aka open() with py3 to produce a file object that encodes
    Unicode as you write, then use json.dump() to write to that file.
    Validate keys to avoid JSON and program out-of-sync.
    """
    dict_config = vars(config) if isinstance(config, types.SimpleNamespace) else config
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    with open(path, 'w', encoding=TXT_CODEC) as f:
        return json.dump(dict_config, f, ensure_ascii=False, indent=4)


class Tracer:
    """
    - custom module-ignore rules
    - trace calls and returns
    - exclude first, then include
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
    max_workers = 10 if iobound else multiprocessing.cpu_count()-1
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


def substitute_keywords_in_file(file, str_map, useliteral=False):
    with open(file) as f:
        original = f.read()
        updated = substitute_keywords(original, str_map, useliteral)
    with open(file, 'w') as f:
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


def alert(title, content, action='Close'):
    if platform.system() == 'Windows':
        cmd = ['mshta', f'vbscript:Execute("msgbox ""{content}"", 0,""{title}"":{action}")']
        os.system(' '.join(cmd))
        return cmd
    if platform.system() == 'Darwin':
        cmd = ['osascript', '-e', f'display alert "{title}" message "{content}"']
    else:
        cmd = ['echo', f'{title}: {content}: {action}']
    subprocess.run(cmd)
    return cmd


def convert_to_wine_path(path, drive=None):
    """
    - path is a macOS-style POSIX full path, e.g.
      - ~/path/to/file
      - /path/to
    - on windows, ~/path/to/file is
    """
    full_path = osp.expanduser(path)
    assert osp.isabs(full_path), f'expected absolute paths, got: {full_path}'
    home_folder = os.environ['USERPROFILE'] if platform.system() == 'Windows' else os.environ['HOME']
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
        home_folder = '~/' if platform.system() == 'Windows' else os.environ['HOME']
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
    plat = platform.system() if platform.system() in cmd_map else '*'
    cmd = cmd_map[plat]['hardKill'] if forcekill else cmd_map[plat]['softKill']
    if plat == '*':
        proc = run_cmd(["pgrep", "-x", name], check=False)
        if proc.returncode != 0:
            return return_codes['procNotFound']
        proc = run_cmd(cmd, check=False)
        if 'not permitted' in (err_log := proc.stderr.decode(LOCALE_CODEC).lower()):
            return return_codes['permissionDenied']
        if err_log:
            return return_codes['unknownError']
        return return_codes['success']
    # Windows: wmic cmd can kill admin-level process
    cmd = cmd_map[plat]['hardKill'] if forcekill else cmd_map[plat]['softKill']
    proc = run_cmd(cmd, check=False)
    if 'not found' in (err_log := proc.stderr.decode(LOCALE_CODEC).lower()) or 'no instance' in proc.stdout.decode(LOCALE_CODEC).lower():
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
    """
    if langs:
        cur_langs = langs
    else:
        cur_locale, encoding = locale.getlocale()
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


class RerunLock:
    """Lock process from reentering when seeing lock file on disk."""

    def __init__(self, name, folder=None, logger=None):
        os.makedirs(folder, exist_ok=True)
        filename = f'lock_{name}.json' if name else 'lock_{}.json'.format(next(tempfile._get_candidate_names()))
        self.lockFile = osp.join(folder, filename) if folder else osp.join(get_platform_tmp_dir(), filename)
        self.logger = logger or glogger
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
            ] if platform.system() == 'Windows' else [
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
                signal.signal(sig, self.handle_signal)

    def lock(self):
        if self.is_locked():
            self.logger.warning('Locked by pid: {}. Will stay locked until it ends.'.format(os.getpid()))
            return False
        save_json(self.lockFile, {'pid': os.getpid()})
        # CAUTION: race condition: saving needs a sec, it's up to application to await lockfile
        return True

    def unlock(self):
        try:
            os.remove(self.lockFile)
        except FileNotFoundError:
            self.logger.warning('Already unlocked. Aborted.')
        except Exception:
            failure = traceback.format_exc()
            self.logger.error('{}\nFailed to unlock. Must delete the lock by hand: {}.'.format(failure, self.lockFile))

    def is_locked(self):
        return osp.isfile(self.lockFile)

    def handle_signal(self, sig, frame):
        msg = f'Terminated due to signal: {signal.Signals(sig).name}; Will unlock'
        self.logger.warning(msg)
        self.unlock()
        raise RuntimeError(msg)


def rerun_lock(name, folder=None, logger=glogger):
    """Decorator for reentrance locking on functions"""

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            my_lock = None
            try:
                my_lock = RerunLock(name, folder, logger)
                if not my_lock.lock():
                    return 1
                try:
                    ret = f(*args, **kwargs)
                except KeyboardInterrupt as e:
                    my_lock.unlock()
                    raise e
                my_lock.unlock()
            except Exception as e:
                my_lock.unlock()
                # leave exception to its upper handler or let the program crash
                raise e
            return ret

        return wrapper

    return decorator


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
    class PathExistsCondition:
        def __init__(self, path):
            self.path = path

        def met(self):
            return osp.exists(self.path) if until_gone else not osp.exists(self.path)

        def update(self):
            pass

    return await_while(PathExistsCondition(lockpath), timeout_ms, step_ms)


def append_to_os_paths(bindir, usesyspath=True, inmemonly=False):
    """
    On macOS, PATH update will only take effect after calling `source ~/.bash_profile` directly in shell. It won't work 
    """
    if platform.system() == 'Windows':
        import winreg
        root_key = winreg.HKEY_LOCAL_MACHINE if usesyspath else winreg.HKEY_CURRENT_USER
        sub_key = r'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment' if usesyspath else r'Environment'
        path_var = 'Path'
        separator = ';'
        with winreg.ConnectRegistry(None, root_key):
            with winreg.OpenKey(root_key, sub_key, 0, winreg.KEY_ALL_ACCESS) as key:
                env_paths, _ = winreg.QueryValueEx(key, path_var)
                matches = [path for path in env_paths.split(separator) if path.lower() == bindir.lower()]
                if matches:
                    return
                if env_paths[-1] != separator:
                    env_paths += separator
                env_paths += f'{bindir}{separator}'
                winreg.SetValueEx(key, path_var, 0, winreg.REG_EXPAND_SZ, env_paths)
    else:
        path_var = 'PATH'
        cfg_file = os.path.expanduser('~/.bash_profile') if platform.system() == 'Darwin' else os.path.expanduser('~/.bashrc')
        if bindir in os.environ[path_var]:
            return
        with open(cfg_file, 'a') as fp:
            fp.write(f'\nexport {path_var}="${path_var}:{bindir}"\n\n')
    os.environ[path_var] += os.pathsep + bindir


def prepend_to_os_paths(bindir, usesyspath=True, inmemonly=False):
    path_var = 'Path' if platform.system() == 'Windows' else 'PATH'
    if inmemonly:
        os.environ[path_var] = bindir + os.pathsep + os.environ[path_var]
        return
    if platform.system() == 'Windows':
        import winreg
        root_key = winreg.HKEY_LOCAL_MACHINE if usesyspath else winreg.HKEY_CURRENT_USER
        sub_key = r'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment' if usesyspath else r'Environment'
        with winreg.ConnectRegistry(None, root_key):
            with winreg.OpenKey(root_key, sub_key, 0, winreg.KEY_ALL_ACCESS) as key:
                env_paths, _ = winreg.QueryValueEx(key, path_var)
                matches = [path for path in env_paths.split(os.pathsep) if path.lower() == bindir.lower()]
                if matches:
                    return
                env_paths = f'{bindir}{os.pathsep}' + env_paths
                winreg.SetValueEx(key, path_var, 0, winreg.REG_EXPAND_SZ, env_paths)
    else:
        cfg_file = os.path.expanduser('~/.bash_profile') if platform.system() == 'Darwin' else os.path.expanduser('~/.bashrc')
        if bindir in os.environ[path_var]:
            return
        with open(cfg_file) as fp:
            lines = fp.readlines()
        lines = [f'\nexport {path_var}="{bindir}:${path_var}"\n\n'] + lines
        with open(cfg_file, 'w') as fp:
            fp.writelines(lines)
    os.environ[path_var] = bindir + os.pathsep + os.environ[path_var]


def remove_from_os_paths(bindir, usesyspath=True, inmemonly=False):
    path_var = 'Path' if platform.system() == 'Windows' else 'PATH'
    if inmemonly:
        os.environ[path_var] = os.environ[path_var].replace(bindir, '')
        return
    if platform.system() == 'Windows':
        import winreg
        root_key = winreg.HKEY_LOCAL_MACHINE if usesyspath else winreg.HKEY_CURRENT_USER
        sub_key = r'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment' if usesyspath else r'Environment'
        with winreg.ConnectRegistry(None, root_key):
            with winreg.OpenKey(root_key, sub_key, 0, winreg.KEY_ALL_ACCESS) as key:
                env_paths, _ = winreg.QueryValueEx(key, path_var)
                matches = [path for path in env_paths.split(os.pathsep) if path.lower() == bindir.lower()]
                if not matches:
                    return
                keepers = [path for path in env_paths.split(os.pathsep) if path.lower() != bindir.lower()]
                env_paths = os.pathsep.join(keepers)
                winreg.SetValueEx(key, path_var, 0, winreg.REG_EXPAND_SZ, env_paths)
    else:
        cfg_file = os.path.expanduser('~/.bash_profile') if platform.system() == 'Darwin' else os.path.expanduser('~/.bashrc')
        if bindir not in os.environ[path_var]:
            return
        # keyword = r'^[\s](*)export PATH="{}:$PATH"'.format(bindir)
        # escape to handle metachars
        pattern_prepend = f'export {path_var}="{bindir}:${path_var}"'
        pattern_append = f'export {path_var}="${path_var}:{bindir}"'
        with open(cfg_file) as fp:
            lines = fp.readlines()
        for li, line in enumerate(lines):
            line = line.strip()
            if line.startswith(pattern_prepend):
                lines[li] = line.replace(pattern_prepend, '')
            elif line.startswith(pattern_append):
                lines[li] = line.replace(pattern_append, '')
        with open(cfg_file, 'w') as fp:
            fp.writelines(lines)
    os.environ[path_var] = os.environ[path_var].replace(bindir, '')


def run_cmd(cmd, cwd=None, logger=None, check=True, shell=False, verbose=False, useexception=True):
    """
    Use shell==True with autotools where new shell is needed to treat the entire command option sequence as a command,
    e.g., shell=True means running sh -c ./configure CFLAGS="..."
    """
    local_debug = logger.debug if logger else print
    local_info = logger.info if logger else print
    local_error = logger.error if logger else print
    console_info = local_info if logger and verbose else local_debug
    if return_error_proc := not useexception:
        check, shell = False, True
    # show cmdline with or without exceptions
    cmd_log = f"""\
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
"""
    local_info(cmd_log)
    try:
        proc = subprocess.run(cmd, check=check, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        stdout_log = proc.stdout.decode(LOCALE_CODEC, errors='backslashreplace')
        stderr_log = proc.stderr.decode(LOCALE_CODEC, errors='backslashreplace')
        if stdout_log:
            console_info(f'stdout:\n{stdout_log}')
        if stderr_log:
            local_error(f'stderr:\n{stderr_log}')
    except subprocess.CalledProcessError as e:
        stdout_log = f'stdout:\n{e.stdout.decode(LOCALE_CODEC, errors="backslashreplace")}'
        stderr_log = f'stderr:\n{e.stderr.decode(LOCALE_CODEC, errors="backslashreplace")}'
        local_info(stdout_log)
        local_error(stderr_log)
        raise e
    except Exception as e:
        # no need to have header, exception has it all
        local_error(e)
        raise e
    return proc


def run_daemon(cmd, cwd=None, logger=None, shell=False):
    local_debug = logger.debug if logger else print
    local_info = logger.info if logger else print
    local_error = logger.error if logger else print
    local_debug(f"""run in ground:
{" ".join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
""")
    try:
        proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        # won't be able to retrieve log from background
    except subprocess.CalledProcessError as e:
        local_info(f'stdout: {e.stdout.decode(LOCALE_CODEC, errors="backslashreplace")}')
        local_error(f'stderr: {e.stderr.decode(LOCALE_CODEC, errors="backslashreplace")}')
        raise e
    except Exception as e:
        local_error(e)
        raise e
    return proc


def watch_cmd(cmd, cwd=None, logger=None, shell=False, verbose=False, useexception=True, prompt=None, timeout=None):
    """
    realtime output
    """
    def read_stream(stream, output_queue):
        for line in iter(stream.readline, b''):
            output_queue.put(line)
    local_debug = logger.debug if logger else print
    local_info = logger.info if logger else print
    local_error = logger.error if logger else print
    console_info = local_info if logger and verbose else local_debug
    if return_error_proc := not useexception:
        check, shell = False, True
    # show cmdline with or without exceptions
    cmd_log = f"""\
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
"""
    local_info(cmd_log)
    try:
        # Start the subprocess with the slave ends as its stdout and stderr
        process = subprocess.Popen(cmd, cwd=cwd, shell=shell, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_queue, stderr_queue = queue.Queue(), queue.Queue()
        # Start separate threads to read from stdout and stderr
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_queue))
        stdout_thread.start()
        stderr_thread.start()

        # Read and print stdout and stderr in real-time
        while True:
            try:
                stdout_line = stdout_queue.get_nowait().decode('utf-8')
                sys.stdout.write(stdout_line)
                sys.stdout.flush()
            except queue.Empty:
                pass
            try:
                stderr_line = stderr_queue.get_nowait().decode('utf-8')
                sys.stderr.write(stderr_line)
                sys.stderr.flush()
            except queue.Empty:
                pass
            if process.poll() is not None and stdout_queue.empty() and stderr_queue.empty():
                break
        # Wait for the threads to finish
        stdout_thread.join()
        stderr_thread.join()
        stdout, stderr = process.communicate()
    except subprocess.CalledProcessError as e:
        stdout_log = f'stdout:\n{e.stdout.decode(LOCALE_CODEC, errors="backslashreplace")}'
        stderr_log = f'stderr:\n{e.stderr.decode(LOCALE_CODEC, errors="backslashreplace")}'
        local_info(stdout_log)
        local_error(stderr_log)
        raise e
    except Exception as e:
        # no need to have header, exception has it all
        local_error(e)
        raise e
    return process.returncode, stdout, stderr


def extract_call_args(file, caller, callee):
    """
    - only support literal args
    - will throw if an arg value is a function call itself
    """

    def _get_kwarg_value_by_type(kwarg):
        if isinstance(kwarg.value, ast.Constant):
            return kwarg.value.value
        elif negative_num := isinstance(kwarg.value, ast.UnaryOp) and isinstance(kwarg.value.op, ast.USub):
            return -kwarg.value.operand.value
        elif isinstance(kwarg.value, ast.Name):
            return kwarg.value.id
        elif isinstance(kwarg.value, (ast.List, ast.Tuple)):
            return [elem.value if isinstance(elem, ast.Constant) else None for elem in kwarg.value.elts]
        elif use_type_map := isinstance(kwarg.value, ast.Attribute):
            return kwarg.value.attr
        print(f'Unsupported syntax node: {kwarg.value}. Will fallback to None.')
        return None

    def _extract_caller_def(cls, func):
        if not cls:
            return next((node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == func), None)
        class_def = next((node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == cls), None)
        return next((node for node in class_def.body if isinstance(node, ast.FunctionDef) and node.name == func), None)

    import ast
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
                'args': [arg.value for arg in call.value.args],
                'kwargs': {kw.arg: _get_kwarg_value_by_type(kw) for kw in call.value.keywords},
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
            elif is_proxy_type := isinstance(node.annotation, ast.Attribute):
                # self.myAttr: TMyType().pyType
                attr_type = node.annotation.value.func.attr
            else:
                attr_type = None
        elif is_assigned_with_const:
            attr_type = type(node.value.value).__name__
        else:
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

    import ast
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
    - only support regular assignments (var_name = literal_value)
    """
    import ast
    import inspect
    mod_name = osp.splitext(osp.basename(file))[0]
    mod = safe_import_module(mod_name, osp.dirname(file))
    parsed = ast.parse(inspect.getsource(mod))

    caller_def = next((node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == caller), None)
    if not caller_def:
        return None
    single_literal_assigns = [node for node in ast.walk(caller_def) if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Constant)]
    assigns = [{
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
    import ast
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
    comments = {}
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
    - returns indices of inserted in resulted lines
    """
    inserts = [inserts] if isinstance(inserts, str) else inserts
    selected_lines = iolines[startlineno:] if startlineno > 0 else iolines
    # find range
    startln, endln = None, None
    # always use startswith, because when we leave a cue, we want it to appear first and foremost
    startln = next((ln for ln, line in enumerate(selected_lines) if line.strip().startswith(startcue)), None)
    if startln is None:
        return startln, endln
    endln = next((ln for ln, line in enumerate(selected_lines[startln:]) if line.strip().startswith(endcue)), None)
    if endln is None:
        return startlineno + startln, None
    if removecues:
        startln -= 1
        endln += 2
    # back to all lines with offset applied
    startln += startlineno
    endln += startln
    if withindent:
        ref_startln = startln + 1 if removecues else startln
        indent = iolines[ref_startln].find(startcue)
        indent_by_spaces = 0
        for idt in range(indent):
            indent_by_spaces += 4 if iolines[ref_startln][idt] == '\t' else 1
        inserts = ['{}{}'.format(' ' * indent_by_spaces, line) for line in inserts]
    # append to current content between cues or not
    lines_to_insert = iolines[startln + 1: endln] + inserts if useappend else inserts
    if skipdups:
        lines_to_insert = list(dict.fromkeys(lines_to_insert))
    # remove lines in b/w
    has_lines_between_keywords = endln - startln > 1
    if has_lines_between_keywords:
        del iolines[startln + 1: endln]
    iolines[startln + 1: startln + 1] = lines_to_insert
    insert_start = startln + 1
    rg_inserted = [insert_start, insert_start + len(lines_to_insert) - 1]
    return rg_inserted


def substitute_lines_in_file(inserts, file, startcue, endcue, startlineno=0, removecues=False, withindent=True, useappend=False, skipdups=False):
    """inserts don't have lineends"""
    lines = load_lines(file)
    rg_inserted = substitute_lines_between_cues(inserts, lines, startcue, endcue, startlineno, removecues, withindent, useappend, skipdups)
    save_lines(file, lines)
    return rg_inserted


def wrap_lines_with_tags(lines, starttag, endtag):
    assert isinstance(lines, list)
    return [starttag] + lines + [endtag]


def convert_compound_cases(snake_text, style='pascal'):
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
        return ' '.join(split_strs)
    # if input is one-piece, then we preserve its middle chars' cases
    # to avoid str.capitalize() turning a string into Titlecase
    if len(split_strs) == 1:
        out_text = split_strs
        leading = out_text[0][0].lower() if style == 'camel' else out_text[0][0].upper()
        out_text[0] = leading + out_text[0][1:]
        return out_text[0]
    out_text = [s.capitalize() for s in split_strs]
    if style == 'camel':
        out_text[0] = out_text[0].lower()
    return ''.join(out_text)


def combine_words(words, style='snake'):
    out_text = '_'.join(words)
    if style == 'snake':
        return out_text
    if style == 'sentence':
        return ' '.join(words)
    if style == 'Sentence':
        return ' '.join(words).capitalize()
    if style == 'title':
        return ' '.join([word.capitalize() for word in words])
    return convert_compound_cases(out_text, style)


def append_lineends_to_lines(lines, style='posix'):
    lineend = '\r\n' if style == 'windows' else '\n'
    return [line + lineend for line in lines]


def zip_dir(srcdir, dstbasename=None):
    """
    zip the entire folder into a zip file under the same parent folder.
    """
    src_par, src_name = osp.split(srcdir)
    dstbn = dstbasename or src_name
    out_zip = osp.join(src_par, dstbn)
    shutil.make_archive(out_zip, format='zip', root_dir=src_par, base_dir=src_name)
    return out_zip


def unzip_dir(srcball, destpardir=None):
    """
    assume srcball has a top-level folder "product".
    unzip to destpardir/product.
    """
    ext = osp.splitext(srcball)[1]
    fmt = None
    if ext == '.zip':
        fmt = 'zip'
    elif ext == '.tar':
        fmt = 'tar'
    elif ext in ('.xz', '.xzip', '.txz'):
        fmt = 'xztar'
    elif ext in ('.gz', '.gzip', '.tgz'):
        fmt = 'gztar'
    elif ext in ('.bz', '.bzip', '.tbz'):
        fmt = 'bztar'
    else:
        raise ValueError(f'Only support zip, tar, gztar, bztar, xztar; got unknown file {ext}')
    if destpardir is None:
        destpardir = osp.dirname(srcball)
    shutil.unpack_archive(srcball, extract_dir=destpardir, format=fmt)


def duplicate_dir(srcdir, dstdir):
    def _dup_dir_posix(sdir, ddir):
        if not sdir.endswith('/'):
            sdir += '/'
        cmd = ['rsync', '-a', '--delete', sdir, ddir.strip('/')]
        run_cmd(cmd, '/')

    def _dup_dir_windows(sdir, ddir):
        cmd = ['xcopy', '/I', '/E', '/Q', '/Y', sdir, f'{ddir}\\']
        run_cmd(cmd, sdir)

    if platform.system() == 'Windows':
        _dup_dir_windows(srcdir, dstdir)
        return
    _dup_dir_posix(srcdir, dstdir)


def compare_textfiles(file1, file2, showdiff=False, contextonly=True, ignoredlinenos=None, logger=None):
    with open(file1) as fp1, open(file2) as fp2:
        lines1 = fp1.readlines()
        lines2 = fp2.readlines()
        if showdiff:
            diff_func = difflib.context_diff if contextonly else difflib.Differ().compare
            diff = diff_func(lines1, lines2)
            lazy_logging(f'***\n{file1} vs.\n{file2}\n***')
            lazy_logging(''.join(diff), logger)
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


def lazy_logging(msg, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)


def copy_file(src, dst, isdstdir=False, keepmeta=False):
    par_dir = dst if isdstdir else osp.dirname(dst)
    os.makedirs(par_dir, exist_ok=True)
    copyfunc = shutil.copy2 if keepmeta else shutil.copy
    try:
        copyfunc(src, dst)
    except shutil.SameFileError:
        glogger.warning(f'source and destination are identical. will SKIP: {osp.abspath(src)} -> {osp.abspath(dst)}.')


def move_file(src, dst, isdstdir=False):
    if isdstdir:
        par_dir = dst
    else:
        par_dir = osp.split(dst)[0]
    os.makedirs(par_dir, exist_ok=True)
    try:
        shutil.move(src, dst)
    except shutil.SameFileError:
        glogger.warning(f'source and destination are identical. will SKIP: {osp.abspath(src)} -> {osp.abspath(dst)}.')


def compare_dirs(dir1, dir2, ignoreddirpatterns=(), ignoredfilepatterns=(), showdiff=True):
    def _collect_folders_files(my_dir):
        n_truncates = len(my_dir)
        my_dir_contents = {
            'dirs': [],
            'files': [],
        }
        for root, folders, files in os.walk(my_dir):
            for folder in folders:
                folder_matching_pattern = next((pat for pat in ignoreddirpatterns if pat in folder), None)
                if folder_matching_pattern:
                    continue
                my_dir_contents['dirs'].append(osp.join(root, folder)[n_truncates + 1:])
            for file in files:
                file_matching_pattern = next((pat for pat in ignoredfilepatterns if fnmatch.fnmatch(file, pat)), None)
                if file_matching_pattern:
                    continue
                my_dir_contents['files'].append(osp.join(root, file)[n_truncates + 1:])

        my_dir_contents['dirs'] = sorted(my_dir_contents['dirs'])
        my_dir_contents['files'] = sorted(my_dir_contents['files'])
        return my_dir_contents

    dir1_contents = _collect_folders_files(dir1)
    dir2_contents = _collect_folders_files(dir2)
    if showdiff:
        import filecmp
        dc = filecmp.dircmp(dir1, dir2, ignore=list(ignoreddirpatterns))
        dc.report_full_closure()
        pp.pprint(f'dir1: {dir1_contents}')
        pp.pprint(f'dir2: {dir2_contents}')
    assert dir1_contents['dirs'] == dir2_contents['dirs'], 'folders different:\n{}\n\nvs.\n\n{}'.format(pp.pformat(dir1_contents['dirs'], indent=2), pp.pformat(dir2_contents['dirs'], indent=2))
    assert dir1_contents['files'] == dir2_contents['files'], 'files different:\n{}\n\nvs.\n\n{}'.format(pp.pformat(dir1_contents['files'], indent=2), pp.pformat(dir2_contents['files'], indent=2))


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
                if isinstance(o, classes):
                    return o.__dict__
                else:
                    return json.JSONEncoder.encode(self, o)

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
        except KeyError as e:
            glogger.debug(f'reload module failed: {e}')
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
        return par_dir
    dirs = [par_dir]
    # focus on pardir(i.e., depth 1), then backtrace 1 at a time from depth-1 to depth
    for dp in range(depth - 1):
        dirs.append(osp.abspath(osp.join(par_dir, osp.normpath('../' * (dp + 1)))))
    return dirs


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
    if is_posix := platform.system() != 'Windows':
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
    if platform.system() != 'Windows':
        drive, relpath = osp.splitdrive(path)
        drive = '/' if relpath.startswith('/') else ''
        return drive, relpath[1:] if drive else relpath
    drive, relpath = osp.splitdrive(path)
    return drive.lower(), relpath


def open_in_browser(path, window='tab', islocal=True):
    """
    - path must be absolute
    - widows path must be converted to posix
    """
    import webbrowser as wb
    import urllib.parse
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


def open_in_editor(path):
    cmds = {
        'Windows': 'explorer',
        'Darwin': 'open',
        'Linux': 'xdg-open',  # ubuntu
    }
    # explorer.exe only supports \
    # start.exe supports / and \, but is not an app cmd but open a prompt
    path = normalize_paths([path])[0]
    cmd = [cmds[platform.system()], path]
    check = platform.system() != 'Windows'
    run_cmd(cmd, check=check)


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
    if not cur_numeric_suffixes:
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


def load_lines(path, rmlineend=False):
    with open(path) as fp:
        lines = fp.readlines()
        if rmlineend:
            lines = [line.rstrip('\n').rstrip('\r') for line in lines]
    return lines


def save_lines(path, lines, toappend=False, addlineend=False, style='posix'):
    if isinstance(lines, str):
        lines = [lines]
    lines_to_write = copy.deepcopy(lines)
    mode = 'a' if toappend else 'w'
    if addlineend:
        line_end = '\n' if style == 'posix' else '\r\n'
        lines_to_write = [line + line_end for line in lines]
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    with open(path, mode) as fp:
        fp.writelines(lines_to_write)
    return lines_to_write


def load_text(path):
    with open(path) as fp:
        text = fp.read()
    return text


def save_text(path, text: str, toappend=False):
    mode = 'a' if toappend else 'w'
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    with open(path, mode) as fp:
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
    if (plat := platform.system()) in supported_plats:
        return
    raise NotImplementedError(f'Expected to run on {supported_plats}, but got {plat}')


def touch(file, withmtime=True):
    par_dir = osp.dirname(file)
    os.makedirs(par_dir, exist_ok=True)
    with open(file, 'a'):
        if withmtime:
            os.utime(file, None)


def lazy_load_listfile(single_or_listfile: str, ext='.list'):
    """
    - we don't force return type-hint to be -> list for reusing args.path str
    - assume list can be text of any nature, i.e., not just paths
    """
    if is_single_item := osp.splitext(single_or_listfile)[1] != ext:
        # we don't care whether it exists or not
        return [single_or_listfile]
    if not osp.isfile(single_or_listfile):
        raise FileNotFoundError(f'Missing list file: {single_or_listfile}')
    return load_lines(single_or_listfile, rmlineend=True)


def normalize_path(path, mode='native'):
    if mode == 'native':
        return path.replace('/', '\\') if platform.system() == 'Windows' else path.replace('\\', '/')
    if mode == 'posix':
        return path.replace('\\', '/')
    if mode == 'win':
        return path.replace('/', '\\')
    raise NotImplementedError(f'Unsupported path noralization mode: {mode}')


def normalize_paths(paths, mode='native'):
    """
    - modes:
      - auto: use platform pathsep
      - posix: use /
      - win: use \\
    """
    return [normalize_path(p, mode) for p in paths]


def lazy_load_filepaths(single_or_listfile: str, ext='.list', root=''):
    """
    - we don't force return type-hint to be -> list for reusing args.path str
    - listfile can have \\ or /, so can root and litfile path
    - we must normalize for file paths
    """
    # if not file path, then user must give root for relative paths
    root = root or os.getcwd()
    # prepare for path normalization: must input posix paths for windows
    root = normalize_path(root, mode='posix')
    abs_list_file = single_or_listfile
    if not osp.isabs(single_or_listfile):
        abs_list_file = normalize_path(single_or_listfile, mode='posix')
        abs_list_file = osp.abspath(f'{root}/{abs_list_file}')
    if is_single_file := osp.splitext(abs_list_file)[1] != ext:
        # we don't care whether it exists or not
        return [single_or_listfile]
    if not osp.isfile(abs_list_file):
        raise FileNotFoundError(f'Missing list file: {abs_list_file}')
    # native win-paths remain the same;
    # posix-format win-paths are converted to native
    paths = [osp.normpath(path) for path in load_lines(abs_list_file, rmlineend=True)]
    return [path if osp.isabs(path) else osp.abspath(f'{root}/{path}') for path in paths]


def read_link(link_path):
    """
    cross-platform symlink/shortcut resolver
    - Windows .lnk can be a command, thus can contain source-path and arguments
    """
    if platform.system() != 'Windows':
        return os.readlink(link_path)
    if osp.islink(link_path):
        return os.readlink(link_path)
    # get_target implementation by hannes, https://gist.github.com/Winand/997ed38269e899eb561991a0c663fa49
    ps_command = \
        "$WSShell = New-Object -ComObject Wscript.Shell;" \
        "$Shortcut = $WSShell.CreateShortcut(\"" + str(link_path) + "\"); " \
                                                                    "Write-Host $Shortcut.TargetPath ';' $shortcut.Arguments "
    output = subprocess.run(["powershell.exe", ps_command], capture_output=True)
    raw = output.stdout.decode('utf-8')
    src_path, args = [x.strip() for x in raw.split(';', 1)]
    return src_path


def is_link(path):
    """
    on windows
    - osp.islink(path) always returns False
    - os.readlink(path) throws when link itself does not exist
    - osp.isdir(path) returns True only when linked source is an existing dir
    - os.readlink(file) raises OSError WinError 4390
    - os.readlink(file.lnk) raises OSError WinError 4390
    - osp.isfile(file.lnk) returns True
    on mac
    - osp.islink(path) returns True when link exists
    - osp.isdir(path) / osp.exists(path) returns True only when linked source is an existing dir
    """
    if platform.system() != 'Windows':
        return osp.islink(path)
    if osp.islink(path):  # posix symlink
        return True
    src = read_link(path)
    return src and src != path


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
    try:
        details = dir(obj)
    except TypeError:
        details = []
    return {'type': type_name, 'attrs': attrs, 'repr': repr(obj), 'details': details}


def _test():
    pass


if __name__ == '__main__':
    _test()
