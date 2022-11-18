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
import copy
import difflib
import fnmatch
import functools
import gettext
import glob
import hashlib
import importlib
import json
import locale
import logging
import logging.config
import multiprocessing
import operator
import os
import os.path as osp
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
from types import SimpleNamespace


#
# Globals
#
_script_dir = osp.abspath(osp.dirname(__file__))
TXT_CODEC = 'utf-8'  # Importable.
MAIN_CFG_FILENAME = 'app.json'
DEFAULT_CFG_FILENAME = 'default.json'


class ChildPromptProxy(threading.Thread):
    """
    When calling a subprocess that prompts for user input, transfer interaction to parent process to avoid indefinite blocking.
    Thread which reads byte-by-byte from the input stream and writes it to the
    standard out.
    Example:
        p = subprocess.Popen(cmd_that_prompts, stdout=subprocess.PIPE)
        r = LivePrinter(p.stdout)
        r.start()
        p.wait()
    """
    def __init__(self, stream):
        self.stream = stream
        self.log = bytearray()
        super().__init__()

    def run(self):
        while True:
            # read one byte from the stream
            buf = self.stream.read(1)

            # break if end of file reached
            if len(buf) == 0:
                break

            # save output to internal log
            self.log.extend(buf)

            # write and flush to main standard output
            sys.stdout.buffer.write(buf)
            sys.stdout.flush()


class SingletonDecorator:
    """
    Decorator to build Singleton class, single-inheritance only.
    Usage:
        class MyClass: ...
        myobj = SingletonDecorator(MyClass, args, kwargs)
    """
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


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
    Create per-file logger and output to shared log file.
    - Otherwise use default config: save to /project_root/project_name.log.
    - 'filename' in config is a filename; must prepend folder path to it.
    :logdir: directory the log file is saved into.
    :name: basename of the log file,
    :cfgfile: config file in the format of dictConfig.
    :return: logger object.
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
    # breakpoint()
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

    logpath = logpath or osp.abspath(f'{osp.dirname(srcpath)}/app.log')

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


def load_json(path):
    """
    Load Json configuration file.
    :param path: path to the config file
    :return: config as a dict
    """
    if is_python3():
        with open(path, 'r', encoding=TXT_CODEC, errors='backslashreplace', newline=None) as f:
            text = f.read()
    else:
        with open(path, 'rU') as f:
            text = f.read()
    # Add object_pairs_hook=collections.OrderedDict hook for py3.5 and lower.
    return json.loads(text, object_pairs_hook=collections.OrderedDict)


def load_json_obj(path):
    """
    Load Json configuration file.
    :param path: path to the config file
    :return: config as an object
    """
    if is_python3():
        with open(path, 'r', encoding=TXT_CODEC, errors='backslashreplace', newline=None) as f:
            text = f.read()
    else:
        with open(path, 'rU') as f:
            text = f.read()
    # Add object_pairs_hook=collections.OrderedDict hook for py3.5 and lower.
    return json.loads(text, object_hook=lambda d: SimpleNamespace(**d))


def save_json(path, config):
    """
    Use io.open(), aka open() with py3 to produce a file object that encodes
    Unicode as you write, then use json.dump() to write to that file.
    Validate keys to avoid JSON and program out-of-sync.
    """
    par_dir = osp.split(path)[0]
    os.makedirs(par_dir, exist_ok=True)
    if is_python3():
        with open(path, 'w', encoding=TXT_CODEC) as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    else:
        with open(path, 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)


def trace_calls_and_returns(frame, event, arg):
    """
    track hook for function calls. Usage:
    sys.settrace(trace_calls)
    """
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from printing
        return
    line_no = frame.f_lineno
    filename = co.co_filename
    if event == 'call':
        print(f'* Call to {func_name} on line {line_no} of {filename}')
        return trace_calls_and_returns
    elif event == 'return':
        print(f'* {func_name} => {arg}')
    return


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


class ParallelWorker:
    def __init__(self, *args, **kwargs):
        pass

    def main(self):
        pass


def init_concurrency(ntasks, nworkers=None, useio=False):
    """
    Suggest concurrency approach based on tasks and number of processes.
    - Use Processes when processes are few or having I/O tasks.
    - Use Pool for many processes or no I/O.
    - Use sequential when tasks are
    :param ntasks: number of total tasks.
    :param nworkers: number of processes, None means to let algorithm decide.
    :param useio: are we I/O-bound?
    :return: dict of all needed parameters.
    """
    if ntasks <= 1:
        return types.SimpleNamespace(type='Sequential', workerCount=1)

    # max out cores but don't take all them all
    if not nworkers:
        nworkers = 10 if useio else multiprocessing.cpu_count() - 1
    concurrency_type = 'Thread' if useio else 'Process'
    return types.SimpleNamespace(type='Thread', workerCount=nworkers)


def ranged_worker(worker, rg, shared, lock):
    results = [worker(shared['Tasks'][t]) for t in range(rg[0], rg[1])]
    # pprint(results)
    with lock:
        tmp = shared['Results']
        for r, result in enumerate(results):
            tmp.append(result)
        shared['Results'] = tmp
    # pprint('tmp: {}'.format(tmp))


def execute_concurrency(worker, shared, lock, algorithm):
    """
    Execute tasks and return results, based on algorithm.
    - worker is unit sequential worker, using single arg.
    - worker returns result as (task['Index'], value).
    - shared is a manager().dict().
    - shared has keys: Title, Tasks.
    - shared['Tasks']: tuple of args for each task worker instance
    - shared['Tasks'][i] has keys: Title, Index, Args, Result
        - Title: info for progress report
        - Index: order of tasks, None for unordered
        - Args: worker input args
        - Result: worker returned results in order
    """
    global glogger
    # TODO: measure timeout for .join()
    if algorithm['Type'] == 'Sequential':
        results = []
        for t, task in enumerate(shared['Tasks']):
            glogger.debug('Execute {} in order: {} of {}: {}'.format(shared['Title'], t + 1, len(shared['Tasks']), task['Title']))
            results.append(worker(task))
        return [result[1] for result in results]
    elif algorithm['Type'] == 'Process':
        glogger.debug('Execute {} in pool of {} processes ...'.format(shared['Title'], algorithm['Count']))
        #
        # Known Issue:
        # - https://bugs.python.org/issue9400
        # - Python multiprocessing.Pool is buggy at join()
        # Reference:
        # - https://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
        #
        results = []
        try:
            with multiprocessing.Pool(processes=algorithm['Count']) as pool:
                results = pool.map(worker, shared['Tasks'])
                pool.close()
                pool.join()
        except Exception:
            traceback.print_exc()
        # Results are always sorted in pool.
        return [result[1] for result in results]
    elif algorithm['Type'] == 'Thread':
        import concurrent.futures
        glogger.debug('Execute {} in pool of {} threads ...'.format(shared['Title'], algorithm['Count']))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=algorithm['Count'])
        results = executor.map(worker, shared['Tasks'])
        return [result[1] for result in results]
    raise ValueError(format_error_message('Found undefined concurrency algorithm.', expected='One of: {}, {}, {}'.format('Sequential', 'Pool', 'Process'), got=algorithm['Type'], suggestions=('Check if this API is up to date', 'retry me'), action='Aborted'))


def profile_runs(funcname, modulefile, nruns=5):
    module_name = osp.splitext(osp.basename(modulefile))[0]
    stats_dir = osp.abspath(f'{osp.dirname(modulefile)}/stats')
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


def save_plist(cfg_file, my_map):
    with open(cfg_file, 'rb') as fp:
        plist = plistlib.load(fp, fmt=plistlib.FMT_XML)
    plist.update(my_map)
    with open(cfg_file, 'wb') as fp:
        plistlib.dump(plist, fp)


def substitute_keywords_in_file(file, str_map, useliteral=False):
    with open(file) as f:
        original = f.read()
        if not useliteral:
            updated = original % str_map
        else:
            updated = original
            for src, dest in str_map.items():
                updated = updated.replace(src, dest)
    with open(file, 'w') as f:
        f.write(updated)


def substitute_keywords(text, str_map, useliteral=False):
    if not useliteral:
        return text % str_map
    updated = text
    for src, dest in str_map.items():
        updated = updated.replace(src, dest)
    return updated


def is_uuid(text, version=4):
    import uuid
    try:
        uuid_obj = uuid.UUID(text, version=version)
    except ValueError:
        return False
    return True


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
        return
    if platform.system() == 'Darwin':
        cmd = ['osascript', '-e', f'display alert "{title}" message "{content}"']
    else:
        cmd = ['echo', f'{title}: {content}: {action}']
    subprocess.run(cmd)


def convert_to_wine_path(path, drive=None):
    full_path = osp.abspath(osp.expanduser(path))
    home_folder = os.environ['HOME']
    if leading_homefolder := full_path.startswith(home_folder):
        mapped_drive = drive or 'Y:'
        full_path = full_path.removeprefix(home_folder)
    else:
        mapped_drive = drive or 'Z:'
    full_path = full_path.replace('/', '\\')
    return mapped_drive + full_path


def convert_from_wine_path(path):
    path = path.strip()
    if path.startswith('Z:') or path.startswith('z:'):
        return path[2:].replace('\\', '/') if len(path) > 2 else '/'
    elif path.startswith('Y:') or path.startswith('y:'):
        return osp.join(os.environ['HOME'], path[2:].replace('\\', '/').strip('/'))
    return path


def kill_process_by_name(name):
    cmd = ['taskkill', '/IM', name, '/F'] if platform.system() == 'Windows' else ['pkill', name]
    subprocess.run(cmd)


def init_translator(localedir, domain='all', langs=None):
    """
    - select locale and set up translator based on system language
    - the leading language in langs, if any, is selected to override current locale
    """
    if langs:
        cur_langs = langs
    else:
        cur_locale, encoding = locale.getdefaultlocale()
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
        excluded = [excluded] if type(excluded).__name__ == 'int' else excluded
        content1 = [cont for c, cont in enumerate(content1) if c not in excluded]
        content2 = [cont for c, cont in enumerate(content2) if c not in excluded]
    return content1 == content2


class RerunLock:
    """Lock process from reentering when seeing lock file on disk."""
    def __init__(self, name, folder=None, logger=glogger):
        os.makedirs(folder, exist_ok=True)
        filename = f'lock_{name}.json' if name else 'lock_{}.json'.format(next(tempfile._get_candidate_names()))
        self.lockFile = osp.join(folder, filename) if folder else osp.join(get_platform_tmp_dir(), filename)
        self.logger = logger
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
                signal.SIGCONT,
                signal.SIGHUP,
                # signal.SIGKILL,
                signal.SIGPIPE,
            ]
            for sig in common_sigs + plat_sigs:
                signal.signal(sig, self.handle_signal)

    def lock(self):
        if not self.is_locked():
            save_json(self.lockFile, {'pid': os.getpid()})
            return True
        else:
            self.logger.warning('Locked by pid: {}. Will stay locked until it ends.'.format(os.getpid()))
            return False

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


def run_cmd(cmd, cwd=None, logger=None, check=True, shell=False, verbose=False):
    """
    Use shell==True with autotools where new shell is needed to treat the entire command option sequence as a command,
    e.g., shell=True means running sh -c ./configure CFLAGS="..."
    """
    local_debug = logger.debug if logger else print
    local_info = logger.info if logger else print
    local_error = logger.error if logger else print
    console_info = local_info if logger and verbose else local_debug
    # show cmdline with or without exceptions
    cmd_log = f"""\
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
"""
    local_info(cmd_log)
    try:
        proc = subprocess.run(cmd, check=check, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        stdout_log = proc.stdout.decode(TXT_CODEC, errors='backslashreplace')
        stderr_log = proc.stderr.decode(TXT_CODEC, errors='backslashreplace')
        if stdout_log:
            console_info(f'stdout:\n{stdout_log}')
        if stderr_log:
            local_error(f'stderr:\n{stderr_log}')
    except subprocess.CalledProcessError as e:
        stdout_log = f'stdout:\n{e.stdout.decode(TXT_CODEC, errors="backslashreplace")}'
        stderr_log = f'stderr:\n{e.stderr.decode(TXT_CODEC, errors="backslashreplace")}'
        local_info(stdout_log)
        local_error(stderr_log)
        raise e
    except Exception as e:
        # no need to have header, exception has it all
        local_error(e)
        raise e
    return proc


def run_daemon(cmd, cwd=None, logger=None, check=True, shell=False):
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
        local_info(f'stdout: {e.stdout.decode(TXT_CODEC, errors="backslashreplace")}')
        local_error(f'stderr: {e.stderr.decode(TXT_CODEC, errors="backslashreplace")}')
        raise e
    except Exception as e:
        local_error(e)
        raise e
    return proc


def run_prompt(cmd, cwd=None, logger=None, check=True, shell=False, verbose=False):
    """
    Use shell==True with autotools where new shell is needed to treat the entire command option sequence as a command,
    e.g., shell=True means running sh -c ./configure CFLAGS="..."
    """
    local_debug = logger.debug if logger else print
    local_info = logger.info if logger else print
    local_error = logger.error if logger else print
    console_info = local_info if logger and verbose else local_debug
    # show cmdline with or without exceptions
    cmd_log = f"""\
{' '.join(cmd)}
cwd: {osp.abspath(cwd) if cwd else os.getcwd()}
"""
    local_info(cmd_log)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        stdout_proxy = ChildPromptProxy(proc.stdout)
        stderr_proxy = ChildPromptProxy(proc.stderr)
        stdout_proxy.start()
        stderr_proxy.start()
        proc.wait()
        stdout_log = stdout_proxy.log.decode(TXT_CODEC, errors='backslashreplace')
        stderr_log = stderr_proxy.log.decode(TXT_CODEC, errors='backslashreplace')
        if stdout_log:
            console_info(f'stdout:\n{stdout_log}')
        if stderr_log:
            local_error(f'stderr:\n{stderr_log}')
    except subprocess.CalledProcessError as e:
        stdout_log = f'stdout:\n{e.stdout.decode(TXT_CODEC, errors="backslashreplace")}'
        stderr_log = f'stderr:\n{e.stderr.decode(TXT_CODEC, errors="backslashreplace")}'
        local_info(stdout_log)
        local_error(stderr_log)
        raise e
    except Exception as e:
        # no need to have header, exception has it all
        local_error(e)
        raise e
    return proc


def extract_call_args(file, caller, callee):
    """
    - only support literal args
    - will throw if an arg value is a function call itself
    """
    def get_kwarg_value_by_type(kwarg):
        if isinstance(kwarg.value, ast.Constant):
            return kwarg.value.value
        elif negative_num := isinstance(kwarg.value, ast.UnaryOp) and isinstance(kwarg.value.op, ast.USub):
            return -kwarg.value.operand.value
        elif isinstance(kwarg.value, ast.Name):
            return kwarg.value.id
        elif isinstance(kwarg.value, (ast.List, ast.Tuple)):
            return [elem.value if isinstance(elem, ast.Constant) else None for elem in kwarg.value.elts]
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
                'kwargs': {kw.arg: get_kwarg_value_by_type(kw) for kw in call.value.keywords},
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
    - all assignments must be about attributes, no local variable are allowed
    - attributes can use type-annotated assignemts (taa)
    - types of attributes without taa can be inferred from constant values
    """
    def _get_attr_by_type(node):
        is_type_annotated = isinstance(node, ast.AnnAssign)
        is_assigned_with_const = isinstance(node.value, ast.Constant)
        is_assigned_with_seq = isinstance(node.value, (ast.List, ast.Tuple))
        is_typed_coll = is_type_annotated and 'slice' in node.annotation._fields
        if is_typed_coll:
            coll_type = node.annotation.value.id
            elem_type = node.annotation.slice.id if coll_type == 'list' else node.annotation.slice.dims[0].id
            attr_type = f'{coll_type}[{elem_type}]'
        elif is_type_annotated:
            attr_type = node.annotation.id
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
    names = [node.attr for node in ast.walk(ctor) if isinstance(node, ast.Attribute)]
    assigns = [node for node in ast.walk(ctor) if isinstance(node, (ast.AnnAssign, ast.Assign))]
    dtypes, values, linenos, end_linenos = [], [], [], []
    for node in assigns:
        atype, avalue = _get_attr_by_type(node)
        dtypes.append(atype)
        values.append(avalue)
        linenos.append(node.lineno)
        end_linenos.append(node.end_lineno)
    attributes = [{'name': n, 'type': t, 'default': v, 'lineno': l, 'end_lineno': e} for n, t, v, l, e in zip(names, dtypes, values, linenos, end_linenos)]
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
    import importlib
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
        return startlineno+startln, None
    if removecues:
        startln -= 1
        endln += 2
    # back to all lines with offset applied
    startln += startlineno
    endln += startln
    if withindent:
        ref_startln = startln+1 if removecues else startln
        indent = iolines[ref_startln].find(startcue)
        indent_by_spaces = 0
        for idt in range(indent):
            indent_by_spaces += 4 if iolines[ref_startln][idt] == '\t' else 1
        inserts = ['{}{}'.format(' ' * indent_by_spaces, line) for line in inserts]
    # append to current content between cues or not
    lines_to_insert = iolines[startln+1: endln] + inserts if useappend else inserts
    if skipdups:
        lines_to_insert = list(dict.fromkeys(lines_to_insert))
    # remove lines in b/w
    has_lines_between_keywords = endln - startln > 1
    if has_lines_between_keywords:
        del iolines[startln+1: endln]
    iolines[startln+1: startln+1] = lines_to_insert
    insert_start = startln+1
    rg_inserted = [insert_start, insert_start+len(lines_to_insert)-1]
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
        return ' '.join([part[0].title()+part[1:] if part else part.title() for part in split_strs])
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


def lazy_logging(msg, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)


def copy_file(src, dst, isdstdir=False, keepmeta=False):
    if isdstdir:
        par_dir = dst
    else:
        par_dir = osp.split(dst)[0]
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
                my_dir_contents['dirs'].append(osp.join(root, folder)[n_truncates+1:])
            for file in files:
                file_matching_pattern = next((pat for pat in ignoredfilepatterns if fnmatch.fnmatch(file, pat)), None)
                if file_matching_pattern:
                    continue
                my_dir_contents['files'].append(osp.join(root, file)[n_truncates+1:])
        
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


def pack_obj(obj, topic=None, envelope=('<KK-ENV>', '</KK-ENV>'), classes=()):
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
        classes = tuple([SimpleNamespace]+list(classes))
        msg_str = json.dumps(msg, cls=CustomJsonEncoder)
    else:
        msg_str = json.dumps(msg, default=lambda o: o.__dict__)

    return f'{envelope[0]}{msg_str}{envelope[1]}'


def lazy_extend_sys_path(paths):
    sys.path = list(dict.fromkeys(sys.path+paths))


def lazy_prepend_sys_path(paths):
    sys.path = list(dict.fromkeys(paths+sys.path))


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


def get_parent_dirs(file, subs=(), depth=1):
    script_dir = osp.abspath(osp.dirname(file))
    par_seq = osp.normpath('../'*depth)
    root = osp.abspath(osp.join(script_dir, par_seq)) if depth > 0 else script_dir
    return script_dir, root, *[osp.join(root, sub) for sub in subs]


def get_ancestor_dirs(file, depth=1):
    """
    given structure: X > Y > Z > file,
    return folder sequence: Z, Y, X
    """
    par_dir = osp.abspath(osp.dirname(file))
    if depth < 2:
        return par_dir
    dirs = [par_dir]
    # focus on pardir(i.e., depth 1), then backtrace 1 at a time from depth-1 to depth
    for dp in range(depth-1):
        dirs.append(osp.abspath(osp.join(par_dir, osp.normpath('../'*(dp+1)))))
    return dirs


def get_child_dirs(root, subs=()):
    return (osp.join(root, sub) for sub in subs)


def read_lines(file, striplineend=False, posix=True):
    with open(file) as fp:
        lines = fp.readlines()
    if striplineend:
        line_end = '\n' if posix else '\r\n'
        for f, line in enumerate(lines):
            lines[f] = line.strip(line_end)
    return lines


def open_in_browser(path, window='tab', islocal=True):
    import webbrowser as wb
    url = f'file://{path}' if islocal else path
    api = {
        'current': wb.open,
        'tab': wb.open_new_tab,
        'window': wb.open_new,
    }
    api[window](url)


def open_in_editor(path):
    cmds = {
        'Windows': 'explorer',
        'Darwin': 'open',
        'Linux': 'xdg-open',  # ubuntu
    }
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


def init_repo(srcfile, appdepth=2, repodepth=3, organization='mycompany', logname=None, verbose=False, uselocale=False):
    """
    assuming a project has a folder structure, create structure and facilities around it
    - structure example: app > subs (src, test, temp, locale, ...), where app is app root
    - structure example: repo > app > subs (src, test, temp, locale, ...), where repo is app-suite root
    - by default, assume srcfile is under repo > app > src
    - set flag uselocale to use gettext localization, by using _T() function around non-fstrings
    - set verbose to all show log levels in console
    - set inter-app sharable tmp folder to platform_cache > organization > app
    """
    assert appdepth <= repodepth
    common = types.SimpleNamespace()
    common.ancestorDirs = get_ancestor_dirs(srcfile, depth=repodepth)
    # CAUTION:
    # - do not include repo to sys path here
    # - always use lazy_extend and lazy_remove
    # just have fixed initial folders to meet most needs in core and tests
    common.locDir, common.srcDir, common.tmpDir, common.testDir = get_child_dirs(app_root := common.ancestorDirs[appdepth - 1], subs=('locale', 'src', 'temp', 'test'))
    common.pubTmpDir = osp.join(get_platform_tmp_dir(), organization, osp.basename(app_root))
    common.stem = osp.splitext(osp.basename(srcfile))[0]
    common.logger = build_default_logger(common.tmpDir, name=logname if logname else common.stem, verbose=verbose)
    if uselocale:
        common.translator = init_translator(common.locDir)
    return common


def backup_file(file, dstdir=None, suffix='.1', keepmeta=True):
    """
    save numeric backup in dstdir or same dir
    - preserve metadata
    - always overwrite non-numeric backup
    """
    bak_dir = dstdir if dstdir else osp.dirname(file)
    bak = osp.join(bak_dir, osp.basename(file)+suffix)
    num = suffix[1:]
    if not num.isnumeric():
        copy_file(file, bak, keepmeta=keepmeta)
        return
    while osp.isfile(bak):
        stem = osp.splitext(bak)[0]
        num = int(osp.splitext(bak)[1][1:]) + 1
        bak = stem + f'.{num}'
    copy_file(file, bak, keepmeta=keepmeta)


def recover_file(file, bakdir=None, suffix=None, keepmeta=True):
    """
    recover file from numeric backup in bakdir or same dir
    """
    bak_dir = bakdir if bakdir else osp.dirname(file)
    assert osp.isdir(bak_dir)
    bn = osp.basename(file)
    files = glob.glob(osp.join(bak_dir, f'{bn}.*'))
    if not files:
        raise FileNotFoundError(f'No backup found for {file} under {bak_dir}')
    if suffix:
        bak = osp.join(bak_dir, bn+suffix)
        copy_file(bak, file, keepmeta=keepmeta)
        return
    latest = max([int(num_sfx) for file in files if (num_sfx := osp.splitext(file)[1][1:]).isnumeric()])
    bak = osp.join(bak_dir, f'{bn}.{latest}')
    copy_file(bak, file, keepmeta=keepmeta)


def deprecate_log(replacewith=None):
    replacement = replacewith if replacewith else 'a documented replacement'
    return f'This is deprecated; use {replacement} instead'


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
        lines_to_write = [line+line_end for line in lines]
    with open(path, mode) as fp:
        fp.writelines(lines_to_write)
    return lines_to_write


def load_text(path):
    with open(path) as fp:
        text = fp.read()
    return text


def save_text(path, text, toappend=False):
    assert isinstance(text, str)
    mode = 'a' if toappend else 'w'
    with open(path, mode) as fp:
        fp.write(text)


def remove_duplication(mylist):
    return list(dict.fromkeys(mylist))


def install_by_macports(pkg, ver=None, lazybin=None):
    """
    Homebrew has the top priority.
    Macports only overrides in memory on demand.
    """
    os_paths = os.environ['PATH']
    prepend_to_os_paths('/opt/local/sbin', inmemonly=True)
    prepend_to_os_paths('/opt/local/bin', inmemonly=True)
    if lazybin and (exe := shutil.which(lazybin)):
        print(f'Found binary: {exe}, and skipped installing package: {pkg}')
        return
    run_cmd(['sudo', 'port', 'install', pkg])
    os.environ['PATH'] = os_paths


def uninstall_by_macports(pkg, ver=None):
    """
    Homebrew has the top priority.
    Macports only overrides in memory on demand.
    """
    os_paths = os.environ['PATH']
    prepend_to_os_paths('/opt/local/sbin', inmemonly=True)
    prepend_to_os_paths('/opt/local/bin', inmemonly=True)
    run_cmd(['sudo', 'port', 'uninstall', pkg])
    os.environ['PATH'] = os_paths


def install_by_homebrew(pkg, ver=None, lazybin=None):
    if lazybin and (exe := shutil.which(lazybin)):
        print(f'Found binary: {exe}, and skipped installing package: {pkg}')
        return
    run_cmd(['brew', 'install', pkg])


def uninstall_by_homebrew(pkg, ver=None):
    run_cmd(['brew', 'remove', pkg])


class Autotools:
    def __init__(self, pkgroot, logger=None):
        self.pkgRoot = pkgroot
        self.logger = logger
        self.make = shutil.which('make')

    def init(self):
        install_by_homebrew('autoconf', lazybin='/usr/local/bin/autoreconf')
        install_by_homebrew('automake', lazybin='/usr/local/bin/automake')

    def autogen(self):
        run_cmd(['./autogen.sh'], cwd=self.pkgRoot, logger=self.logger)

    def configure(self, opts=(), flags=()):
        """
        Must run under shell mode
        - CFLAGS="...", CXXFLAGS... are not options but environment variables
        - So must spawn subprocess under shell mode: Treat the ./configure sequence as a whole command
        """
        compiler_flags = ['env'] + list(flags) if flags else []
        run_cmd([' '.join(compiler_flags + ['./configure'] + list(opts))], shell=True, cwd=self.pkgRoot, logger=self.logger)

    def make_install(self, njobs=8):
        run_cmd([self.make, f'-j{njobs}'], cwd=self.pkgRoot, logger=self.logger)
        run_cmd([self.make, 'install'], cwd=self.pkgRoot, logger=self.logger)

    def make_clean(self):
        run_cmd([self.make, 'clean'], cwd=self.pkgRoot, logger=self.logger)


class CMake:
    def __init__(self, cmakelistsdir, builddir, logger=None):
        self.cmakelistsDir = cmakelistsdir
        self.buildDir = builddir
        self.logger = logger
        self.cmake = shutil.which('cmake')

    def configure(self, prefix=None, config='Debug', builddll=True, opts=()):
        cmd = [self.cmake,
               '-LAH',  # see all cmake options
               f'-DCMAKE_BUILD_TYPE={config}',
               f'-DBUILD_SHARED_LIBS={"ON" if builddll else "OFF"}',
               ]
        if prefix:
            cmd += [f'-DCMAKE_INSTALL_PREFIX={prefix}']
        cmd += list(opts)
        cmd += [self.cmakelistsDir]
        run_cmd(cmd, cwd=self.buildDir)

    def build(self, gnumake=True):
        job_flag = ['-j8'] if gnumake else []
        cmd = [self.cmake, '--build', '.', '--target', 'install'] + job_flag
        run_cmd(cmd, cwd=self.buildDir)

    def clean(self):
        cmd = [self.cmake, '--build', '.', '--target', 'clean']
        run_cmd(cmd, cwd=self.buildDir)


def validate_platform(plat):
    if platform.system() != plat:
        raise NotImplementedError(f'Runs on {plat} only.')


def build_ico(master, ico, size='256x256'):
    cmd = ['magick', master, '-background', 'none', '-resize', size, '-density', size, ico]
    run_cmd(cmd)


def build_iconset(master, iconset):
    validate_platform('Darwin')
    iconset_dir = osp.join(f'/{get_platform_tmp_dir()}/icons/icon.iconset')
    os.makedirs(iconset_dir, exist_ok=True)
    sizes = ('16', '16@2x', '32', '32@2x', '128', '128@2x', '256', '256@2x', '512')
    for sz in sizes:
        comps = sz.split('@')
        sz = comps[0]
        actual_size = str(2*int(sz)) if (has_sep := len(comps) > 1) else sz
        suffix = f'{sz}x{sz}@2x' if has_sep else f'{sz}x{sz}'
        run_cmd(['sips', '-z', actual_size, actual_size, master, '--out', osp.abspath(f'{iconset_dir}/icon_{suffix}.png')])
    copy_file(master, osp.abspath(f'{iconset_dir}/icon_512x512@2x.png'))
    run_cmd(['iconutil', '-c', 'icns', iconset_dir, '-o', iconset])
    shutil.rmtree(iconset_dir, ignore_errors=True)


def fix_dylib_dependencies(target, rootprefix='@executable_path'):
    """
    prepare .dylibs for macOS app deployment
    - pre-condition: all the fixing must happen at the dst locations, i.e., requiring pre-deployment
    - this is because the fixing requires src libs to exist and maintains old embedded paths
    - recursively find all dependencies of a target binary executable or lib
    - replace build-time prefix with user-supplied string for all dependencies
    - replace dylib's own id with user-provided prefix
    """
    class DynLinked:
        """
        a pre-deployed binary file having dependencies to fix for runtime execution
        """
        analyzer = None
        fixer = None
        execPath = None  # physical runtime exec folder
        syslibPrefixes = ['/System/Library', '/usr/']

        def __init__(self, buildproduct, prefix='@executable_path', parent=None):
            assert osp.isabs(buildproduct), f'Not an absolute path: {buildproduct}, parent: {parent}'
            self.buildProduct = buildproduct
            self.distributable = None
            self.deps = None
            # CAUTION
            # - prefix is to be embedded into binaries
            # - input prefix must be relative to @executable_path, e.g., @executable_path/libs
            # - input may already start with a build-time base
            runtime_base_tag = '@executable_path'
            # prefix must start with '@executable_path'
            self.prefix = prefix.rstrip(os.sep) if prefix.startswith(runtime_base_tag) else osp.join(runtime_base_tag, prefix.strip(os.sep))
            # for reference only, we trace deps tree top-down from the outside
            self.ref = parent
            if is_root := not self.ref:
                # root binary is our target, no need to decorate
                self.distributable = buildproduct
                DynLinked.execPath = osp.dirname(self.distributable)
                glogger.info(f'Root aka. target binary: {self.distributable}')
            else:
                assert osp.isdir(DynLinked.execPath), 'Root binary has not been processed'
                # this path may not exist yet
                # @executable_path/relative/to/my.dylib
                dest_dir = osp.join(DynLinked.execPath, relpath_to_exepath := osp.relpath(prefix, runtime_base_tag).strip(os.sep))
                self.distributable = osp.join(dest_dir, bin_filename := osp.split(self.buildProduct)[1])

        def extract_direct_deps(self):
            """
            - get 1st-order dependencies using otool -L
            """
            # all deps of a leaf binary are system libs and so their deps are empty
            if sys_lib := not osp.isfile(self.buildProduct):
                return []
            # CAUTION:
            # - install_name_tool requires source binaries to exist at their embedded src paths
            # - so we must copy all the src bins to prefix location (dst) first to preserve embedded src paths
            if not osp.isfile(self.distributable):
                copy_file(self.buildProduct, self.distributable, isdstdir=False)
            proc = run_cmd([DynLinked.analyzer, '-L', self.distributable])
            list_head_lineno = 1
            deps = [line.strip() for line in self._extract_deps(proc.stdout)[list_head_lineno:]]
            # remove suffixes from dependency line
            deps = list(map(lambda d: d[:d.find(' (compatibility version')], deps))
            deps = list(map(lambda d: d[:start] if (start := d.find(' (architecture')) >= 0 else d, deps))
            # bypass:
            # - sys deps: they are not shipped and should be ignored
            # - self: it requires a different cmd to fix, dylib only
            to_bypass = list(filter(lambda d: any([d.startswith(path) for path in DynLinked.syslibPrefixes + [self.buildProduct]]), deps))
            deps = list(set(deps).difference(to_bypass))
            if not self.deps:
                self.deps = deps
            else:
                glogger.info('Deps exist. Skipped to avoid pollution from already-fixed @executable_path prefix')
            glogger.info(f"""Dependencies found for {self.buildProduct}:
System and self (bypassed):
{pp.pformat(to_bypass, indent=2)}
Application (fixable) :
{pp.pformat(deps, indent=2)}""")
            return deps  # always return latest analytical result

        def fix_deps(self):
            """
            recursively fix deps' paths with global prefix:
            - install-dependent dylib search path
            """
            assert osp.isfile(self.distributable)
            for dep in self.deps:
                assert dep.endswith('.dylib')
                distributable = osp.join(self.prefix, osp.split(dep)[1])
                run_cmd([DynLinked.fixer, '-change', dep, distributable, dep_ref := self.distributable])

        def fix_self(self):
            """
            - The first line of otool output is the binary itself.
            - It requires a unique way of fixing
            """
            # copy src libs to dst in advance to avoid polluting src libs while iterating
            # - translate prefix if it starts with @executable_path
            assert osp.isfile(self.distributable)
            distributable = osp.join(self.prefix, fn := osp.split(self.buildProduct)[1])
            # rename parent dir
            run_cmd([DynLinked.fixer, '-id', distributable, self.distributable])

        def report(self):
            if sys_lib := not osp.isfile(self.distributable):
                return
            proc = run_cmd([DynLinked.analyzer, '-L', self.distributable])
            glogger.info(f'fixed for {self.distributable}:\n{proc.stdout.decode(TXT_CODEC)}')

        def _extract_deps(self, otoolout):
            return otoolout.decode(TXT_CODEC).split('\n\t')

    def _collect_deps_tree(my_bin, io_allbins, prefix):
        """
        recursively find input binary's dependencies
        :param my_bin: input binary as the root
        :param io_allbins: dict to save all binaries and their deps for fixing them next
        :return:
        """
        if not isinstance(my_bin, DynLinked):
            raise TypeError('Input binary must be a DynLinked type')
        if exit_at_leaf := not my_bin.extract_direct_deps():
            return False
        parent = copy.deepcopy(my_bin)
        io_allbins[parent.buildProduct] = parent
        for child in parent.deps:
            if cyclic_refs := child == parent.buildProduct:
                continue
            if already_fixed := child.startswith(prefix) or child.startswith('@executable_path'):
                glogger.debug(f'Already fixed by other exec: {child}, ref: {parent.buildProduct}; Skipped')
                continue
            glogger.debug(f'Found child: {child} of {parent.buildProduct}')
            cbin = DynLinked(child, prefix=prefix, parent=parent)
            _collect_deps_tree(cbin, io_allbins, prefix)
        return True
    validate_platform('Darwin')
    if not osp.isfile(target):
        raise FileNotFoundError(f'Missing target: {target}')
    target = osp.abspath(target)
    if osp.isabs(rootprefix):
        raise ValueError(f'Expected prefix to be relative path or @executable_path, got absolute: {rootprefix}')
    DynLinked.fixer = shutil.which('install_name_tool')
    DynLinked.analyzer = shutil.which('otool')
    if not DynLinked.fixer or not DynLinked.analyzer:
        raise FileNotFoundError(f'Missing install_name_tool or otool. Install Xcode and toolchain first')
    all_bins = {}
    root = DynLinked(target, prefix=rootprefix)
    if _collect_deps_tree(root, all_bins, prefix=rootprefix):
        glogger.info('dependency recursively parsed')
    for key in all_bins:
        all_bins[key].fix_deps()
    # CAUTION: must fix all deps before fixing self
    for key in all_bins:
        all_bins[key].fix_self()
    for key in all_bins:
        all_bins[key].report()


def codesign(binary, identity=None, overwrite=True):
    """
    identity looks like: e.g., "Apple Development: me@email.com (IDIDIDIDID)"
    """
    validate_platform('Darwin')
    run_cmd(['security', 'find-identity', '-v', '-p', 'codesigning'])
    if overwrite:
        run_cmd(['codesign', '--remove-signature', binary])
    # -v with feedback
    run_cmd(['codesign', '-s', identity, '-v', binary])
    # gatekeeper validation
    run_cmd(['spctl', '-a', '-t', 'exec', '-vv', binary])


def build_dmg(masterdir, resdir='', dmg='', name=''):
    """
    build a DMG installer based on fixed layout, and lazy-include resources
    - masterdir: content folder holding the app and optionally sub-folder "help"
    - resdir: asset folder for building dmg only, not part of app content, usually holds background image (bg.png, made by Keynote 4:3, size 1024x768) and volume iconset (app.icns)
    - dmg: path to output dmg
    - name: app name, default to use basename of masterdir
    """
    validate_platform('Darwin')
    app = types.SimpleNamespace(name='', helpDir='', license='', bundle='')
    res = types.SimpleNamespace(background='', icon='', mountVolume='')
    app.name = name or osp.basename(masterdir)
    app.bundle = f'{app.name}.app'
    app.helpDir = osp.join(masterdir, 'help')
    if osp.isdir(app.helpDir):
        app.license = osp.join(app.helpDir, 'LICENSE.txt')
    res.mountVolume = f'{app.name}_installer'
    if osp.isdir(resdir):
        res.background = osp.join(resdir, 'bg.png')
        res.icon = osp.join(resdir, 'app.icns')
        assert osp.isfile(res.background)
        assert osp.isfile(res.icon)
    out_dmg = dmg or osp.join(osp.dirname(masterdir), f'install_{app.name}.dmg'.lower())
    dmg_tool = shutil.which('create-dmg')
    if not dmg_tool:
        raise FileNotFoundError('Missing create-dmg; install and retry')
    cmd = [dmg_tool,
           '--volname', res.mountVolume,
           '--volicon', res.icon,
           '--icon-size', '90',
           '--hide-extension', app.bundle, ]
    if osp.isfile(app.license):
        cmd += ['--eula', app.license]
    window_rect = types.SimpleNamespace(x='100', y='0', w='500', h='300')
    icon_pos = types.SimpleNamespace(x='150', y='110')
    appdroplink_pos = types.SimpleNamespace(x='350', y='110')
    help_pos = types.SimpleNamespace(x='', y='')
    if osp.isdir(app.helpDir):
        window_rect.w, window_rect.h = '550', '500'
        icon_pos.x, icon_pos.y = '150', '150'
        help_pos.x, help_pos.y = '150', '300'
        appdroplink_pos.x, appdroplink_pos.y = '400', '150'
    if osp.isfile(res.background):
        glogger.warning('Found background image; size must be 1024x768 (Keynote 4:3)')
        cmd += ['--background', res.background]
        window_rect.w, window_rect.h = '1024', '820'
        icon_pos.x, icon_pos.y = '362', '378'
        help_pos.x, help_pos.y = '362', '500'
        appdroplink_pos.x, appdroplink_pos.y = '650', '378'
    # must add help after adjusting for background
    if osp.isdir(app.helpDir):
        cmd += ['--add-file', 'Help', app.helpDir, help_pos.x, help_pos.y]
    cmd += ['--window-pos', window_rect.x, window_rect.y,
            '--window-size', window_rect.w, window_rect.h,
            '--icon', app.bundle, icon_pos.x, icon_pos.y,
            '--app-drop-link', appdroplink_pos.x, appdroplink_pos.y, ]
    cmd += [out_dmg, masterdir]
    if tmp_dmgs := glob.glob('rw.*.dmg'):
        for td in tmp_dmgs:
            os.remove(td)
    if osp.isfile(out_dmg):
        os.remove(out_dmg)
    run_cmd(cmd)


def build_xcodeproj(proj, scheme, config='Debug', sdk='macosx'):
    cmd = ['xcodebuild', '-project', proj, '-scheme', scheme, '-sdk', sdk, '-configuration', config, 'build']
    run_cmd(cmd)


def clean_xcodeproj(proj, scheme, config='Debug', sdk='macosx'):
    cmd = ['xcodebuild', '-project', proj, '-scheme', scheme, '-sdk', sdk, '-configuration', config, 'clean']
    run_cmd(cmd)


def touch(file, withmtime=True):
    with open(file, 'a'):
        if withmtime:
            os.utime(file, None)


def lazy_load_listfile(single_or_listfile: str, ext='.list'):
    """
    we don't force return type-hint to be -> list for reusing args.path str
    """
    if is_listfile := fnmatch.fnmatch(single_or_listfile, f'*{ext}'):
        if not osp.isfile(single_or_listfile):
            raise FileNotFoundError(f'Missing list file ({ext}): {single_or_listfile}')
        return load_lines(single_or_listfile, rmlineend=True)
    single_item = single_or_listfile
    return [single_item]


def is_link(path):
    """
    on windows
    - osp.islink(path) always returns False
    - os.readlink(path) throws when link itself does not exist
    - osp.isdir(path) / osp.exists(path) returns True only when linked source is an existing dir
    on mac
    - osp.islink(path) returns True when link exists
    - osp.isdir(path) / osp.exists(path) returns True only when linked source is an existing dir
    """
    if platform.system() == 'Windows':
        try:
            lnk = os.readlink(path)
            return True
        except FileNotFoundError:
            return False
    return osp.islink(path)


def _test():
    pass


if __name__ == '__main__':
    _test()
