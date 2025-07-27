"""
tests that don't need external data
"""
import copy
import datetime
import getpass
import glob
import json
import math
import platform
import shutil
import signal
import sys
import os
import os.path as osp
import subprocess
import tempfile
import threading
import time
import types
import unittest.mock as um
import uuid

# 3rd party
import pytest

# project
_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, repo_root := osp.abspath(f'{_script_dir}/..'))
import xypyutil as util

_case_dir = _script_dir
_src_dir = osp.abspath(osp.dirname(_case_dir))
_org_dir = osp.join(_case_dir, '_org')
_gen_dir = osp.join(_case_dir, '_gen')
_ref_dir = osp.join(_case_dir, '_ref')
_skip_slow_tests = osp.isfile(osp.join(_case_dir, 'skip_slow_tests.cfg.txt'))
_skip_reason = 'tests requires long network or file i/o are temporarily skipped during tdd'


def test_singletion_decorator():
    class MyClass:
        def __init__(self, n, s):
            self.n = n
            self.s = s

    singleton_class = util.SingletonDecorator(MyClass, 100, 'hello')
    obj1 = singleton_class(100, 'hello')
    obj2 = singleton_class(200, 'world')
    assert obj1 == obj2


def test_log_filters():
    info_lpf = util.LowPassLogFilter(20)
    debug_log = types.SimpleNamespace(levelno=10)
    info_log = types.SimpleNamespace(levelno=20)
    warning_log = types.SimpleNamespace(levelno=30)
    error_log = types.SimpleNamespace(levelno=40)
    assert info_lpf.filter(debug_log)
    assert not info_lpf.filter(warning_log)
    info_hpf = util.HighPassLogFilter(20)
    assert not info_hpf.filter(debug_log)
    assert info_hpf.filter(warning_log)
    info_warning_bpf = util.BandPassLogFilter((20, 30))
    assert not info_warning_bpf.filter(debug_log)
    assert not info_warning_bpf.filter(error_log)
    assert info_warning_bpf.filter(info_log)
    assert info_warning_bpf.filter(warning_log)


def test_get_platform_home_dir():
    if util.PLATFORM == 'Windows':
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}')
        assert util.get_platform_home_dir() == expected
    elif util.PLATFORM == 'Darwin':
        expected = osp.abspath(f'/Users/{getpass.getuser()}')
        assert util.get_platform_home_dir() == expected
    elif util.PLATFORM == 'Linux':
        expected = osp.abspath(f'/home/{getpass.getuser()}')
        assert util.get_platform_home_dir() == expected
    else:
        with pytest.raises(NotImplementedError):
            util.get_platform_home_dir()


def test_get_platform_appdata_dir():
    if util.PLATFORM == 'Windows':
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}\\AppData\\Roaming')
        assert util.get_platform_appdata_dir() == expected
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}\\AppData\\Local')
        assert util.get_platform_appdata_dir(winroam=False) == expected
    elif util.PLATFORM == 'Darwin':
        expected = osp.abspath(f'/Users/{getpass.getuser()}/Library/Application Support')
        assert util.get_platform_appdata_dir() == expected
    else:
        with pytest.raises(NotImplementedError):
            util.get_platform_appdata_dir()


def test_get_platform_tmp_dir():
    if util.PLATFORM == 'Windows':
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}\\AppData\\Local\\Temp')
        assert util.get_platform_tmp_dir() == expected
    elif util.PLATFORM == 'Darwin':
        expected = osp.abspath(f'/Users/{getpass.getuser()}/Library/Caches')
        assert util.get_platform_tmp_dir() == expected
    elif util.PLATFORM == 'Linux':
        expected = '/tmp'
        assert util.get_platform_tmp_dir() == expected
    else:
        with pytest.raises(NotImplementedError):
            util.get_platform_tmp_dir()


def test_get_posix_user_cfgfile():
    if util.PLATFORM == 'Windows':
        assert True
        return
    sh = os.environ['SHELL']
    os.environ['SHELL'] = '/bin/bash'
    assert util.get_posix_shell_cfgfile() == osp.join(os.environ['HOME'], '.bash_profile')
    os.environ['SHELL'] = '/bin/zsh'
    assert util.get_posix_shell_cfgfile() == osp.join(os.environ['HOME'], '.zshrc')
    os.environ['SHELL'] = sh


def test_build_default_logger():
    log_dir = osp.dirname(__file__)
    logger = util.build_default_logger(log_dir, name='my')
    logger.debug('hello logger')
    log_file = osp.join(log_dir, 'my.log')
    assert osp.isfile(log_file)
    for hdl in logger.handlers:
        hdl.close()
    os.remove(log_file)


def test_catch_unknown_exception():
    util.catch_unknown_exception(RuntimeError, 'exception info', None)


def test_format_brief():
    got = util.format_brief(
        title='title',
        bullets=['item 1', 'item 2'],
    )
    expected = """\
title:
- item 1
- item 2"""
    assert got == expected
    got = util.format_brief(
        bullets=['item 1', 'item 2'],
    )
    expected = """\
- item 1
- item 2"""
    assert got == expected
    got = util.format_brief(
        title='title',
    )
    expected = 'title'
    assert got == expected


def test_format_log():
    expected = """\
situation:
Detail:
- fact 1
- fact 2
Advice:
- advice 1
- advice 2
Done-for-you:
- done 1
"""
    got = util.format_log(
        situation='situation',
        detail=['fact 1', 'fact 2'],
        advice=['advice 1', 'advice 2'],
        reso=['done 1'],
    )
    assert got == expected
    expected = """\
situation:
Detail:
  Expected:
  - expected 1
  - expected 2
  Got:
  - got 1
  - got 2
Advice:
- advice 1
- advice 2
Done-for-you:
- done 1
"""
    got = util.format_log(
        situation='situation',
        detail=util.format_error(['expected 1', 'expected 2'], ['got 1', 'got 2']),
        advice=['advice 1', 'advice 2'],
        reso=['done 1'],
    )
    assert got == expected


def test_throw():
    with pytest.raises(ValueError) as excinfo:
        util.throw(ValueError, 'hello', 'world')
        assert excinfo.value.args == ('hello', 'world')


def test_is_python3():
    assert util.is_python3()


def test_load_json():
    utf8_file = osp.join(_org_dir, 'load_utf8.json')
    loaded = util.load_json(utf8_file)
    assert loaded == {
        "english": "Bye",
        "简体中文": "再见",
        "繁體中文": "再會",
        "日本語": "さよなら"
    }
    loaded = util.load_json(utf8_file, as_namespace=True)
    assert loaded == types.SimpleNamespace(
        english="Bye",
        简体中文="再见",
        繁體中文="再會",
        日本語="さよなら"
    )


def test_save_json():
    out_file = osp.join(_gen_dir, 'save_utf8.json')
    config = {
        "english": "Bye",
        "简体中文": "再见",
        "繁體中文": "再會",
        "日本語": "さよなら"
    }
    util.save_json(out_file, config)
    assert osp.isfile(out_file)
    config = types.SimpleNamespace(**config)
    util.save_json(out_file, config)
    assert osp.isfile(out_file)
    shutil.rmtree(_gen_dir, ignore_errors=True)


def test_tracer():
    """
    - test tracer directly would quit test
    - start tracer directly under pytest triggers "missing excinfo attribute" when referencing frame object
    - must use a different file
    """
    src = osp.join(_org_dir, 'test_trace_this.py')
    cmd = ['poetry', 'run', 'python', src]
    proc = util.run_cmd(cmd, cwd=osp.dirname(__file__))
    lines = proc.stdout.decode(util.TXT_CODEC).splitlines()
    assert lines == [
        "Call: __main__.hello(n=100, s='world', f='0.99') - def hello(n, s, f):",
        "Call: __main__.hello => hello, 100, world, 0.99 - return x",
    ]


def test_get_md5_checksum():
    missing_file = 'missing'
    assert util.get_md5_checksum(missing_file) is None
    valid_file = osp.abspath(f'{_script_dir}/../LICENSE')
    assert osp.isfile(valid_file)
    # line-ends count
    expected = '5d326be91ee12591b87f17b6f4000efe' if util.PLATFORM == 'Windows' else '7a3beb0af03d4afff89f8a69c70a87c0'
    assert util.get_md5_checksum(valid_file) == expected


def test_logcall():
    @util.logcall('trace', logger=util.build_default_logger(logdir := osp.join(util.get_platform_tmp_dir(), '_util'), name='test_logcall'))
    def myfunc(n, s, f=1.0):
        x = f'hello, {n}, {s}, {f}'
        return x

    key_lines = {
        "Enter: 'myfunc' <= args=(100, 'hello'), kwargs={'f': 0.99}: trace",
        "Exit: 'myfunc' => hello, 100, hello, 0.99",
    }
    myfunc(100, 'hello', f=0.99)
    log_file = osp.join(logdir, 'test_logcall.log')
    log = util.load_lines(log_file, rmlineend=True)
    assert key_lines.issubset(log)


def map_worker(enum):
    e, elem = enum[0], enum[1]
    return elem * 2


def test_is_toplevel_function():
    def inner_func(enum):
        e, elem = enum[0], enum[1]
        return elem * 2

    assert util.is_toplevel_function(map_worker)
    assert not util.is_toplevel_function(inner_func)


def test_concur_map():
    def inner_func(enum):
        e, elem = enum[0], enum[1]
        return elem * 2

    n = 20
    data = [i for i in range(n)]
    logger = util.build_default_logger(
        logdir=osp.abspath(f'{_gen_dir}'),
        name='test_concur_map',
        verbose=True)
    res = util.concur_map(inner_func, data, worker_count=5, iobound=True, logger=logger)
    assert res == [i * 2 for i in range(n)]
    res = util.concur_map(map_worker, data, worker_count=5, iobound=False, logger=logger)
    assert res == [i * 2 for i in range(n)]
    shutil.rmtree(_gen_dir, ignore_errors=True)


def test_profile_runs():
    profile_mod = osp.join(_org_dir, 'profile_this.py')
    funcname = 'run_profile_target'
    stats = util.profile_runs(funcname, profile_mod, outdir=_gen_dir)
    assert stats.total_calls <= 714
    shutil.rmtree(_gen_dir, ignore_errors=True)


def test_load_save_plist():
    my_plist = osp.join(_org_dir, 'my.plist')
    loaded = util.load_plist(my_plist)
    assert loaded['FirstName'] == 'John' and loaded['LastName'] == 'Doe'
    bin_plist = osp.join(_org_dir, 'my.bin.plist')
    loaded = util.load_plist(bin_plist, True)
    assert loaded['FirstName'] == 'John' and loaded['LastName'] == 'Doe'
    out_file = osp.join(_gen_dir, 'my.plist')
    util.save_plist(out_file, loaded)
    loaded = util.load_plist(out_file)
    assert loaded['FirstName'] == 'John' and loaded['LastName'] == 'Doe'
    util.save_plist(out_file, loaded, True)
    loaded = util.load_plist(out_file, True)
    assert loaded['FirstName'] == 'John' and loaded['LastName'] == 'Doe'
    shutil.rmtree(_gen_dir, ignore_errors=True)


def test_substitute_keywords_in_file():
    fn = 'subs_this.txt'
    org_file = osp.join(_org_dir, fn)
    in_file = osp.join(_gen_dir, fn)
    util.copy_file(org_file, in_file)
    ref_file = osp.join(_ref_dir, 'subs_this.ref.txt')
    str_map = {
        'var': 'foo',
    }
    util.substitute_keywords_in_file(in_file, str_map)
    assert util.compare_textfiles(in_file, ref_file, showdiff=True)
    str_map = {
        'var': 'foo',
        '%%': '$$'
    }
    util.copy_file(org_file, in_file)
    ref_file = osp.join(_ref_dir, 'subs_this_literal.ref.txt')
    util.substitute_keywords_in_file(in_file, str_map, useliteral=True)
    assert util.compare_textfiles(in_file, ref_file, showdiff=True)
    shutil.rmtree(_gen_dir, ignore_errors=True)


def test_substitute_keywords():
    str_map = {
        'var': 'foo',
    }
    text = """
变量 : %(var)s
Escape : %%
Variable in text: %(var)siable
"""
    assert util.substitute_keywords(text, str_map) == """
变量 : foo
Escape : %
Variable in text: fooiable
"""
    str_map = {
        'var': 'foo',
        '%%': '$$'
    }
    text = """
变量 : %(var)s
Escape : %%
Variable in text: %(var)siable
"""
    assert util.substitute_keywords(text, str_map, useliteral=True) == """
变量 : %(foo)s
Escape : $$
Variable in text: %(foo)siable
"""


def test_is_uuid():
    # any version
    valid = 'c9bf9e57-1685-4c89-bafb-ff5af830be8a'
    assert util.is_uuid(valid)
    invalid = 'c9bf9e58'
    assert not util.is_uuid(invalid)
    assert util.is_uuid(valid, version=4)
    assert not util.is_uuid(valid, version=1)


def test_get_uuid_version():
    valid = '{C9BF9E57-1685-4C89-BAFB-FF5AF830BE8A}'
    assert util.get_uuid_version(valid) == 4
    valid = '{C9BF9E57-1685-1C89-BAFB-FF5AF830BE8A}'
    assert util.get_uuid_version(valid) == 1
    invalid = 'c9bf9e58'
    assert util.get_uuid_version(invalid) is None


def test_create_guid():
    uid = util.create_guid()
    guid = util.create_guid()
    assert isinstance(guid, str)
    assert len(guid) == 38
    assert guid[0] == '{'
    assert guid[-1] == '}'
    assert guid[9] == '-'
    assert guid[14] == '-'
    assert guid[19] == '-'
    assert guid[24] == '-'
    assert guid[37] == '}'
    assert guid[1:9].isalnum()
    assert guid[10:14].isalnum()
    assert guid[15:19].isalnum()
    assert guid[20:24].isalnum()
    assert guid[25:37].isalnum()
    assert guid.upper() == guid
    assert util.get_uuid_version(uid) == 4
    uid = util.create_guid(1)
    assert util.get_uuid_version(uid) == 1
    uid = util.create_guid(5, 'my_namespace')
    assert util.get_uuid_version(uid) == 5
    assert util.create_guid(5) is None
    assert util.create_guid(3) is None
    assert util.create_guid(2) is None
    assert util.create_guid(999) is None


def test_get_guid_from_uuid():
    uid = uuid.UUID('{e6a6dd92-5b96-4a09-9cc4-d44153b900a4}')
    assert util.get_guid_from_uuid(uid) == '{E6A6DD92-5B96-4A09-9CC4-D44153B900A4}'


def test_get_clipboard_content():
    content = 'hello world!'
    import tkinter as tk
    root = tk.Tk()
    # keep the window from showing
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(content)
    root.quit()
    assert util.get_clipboard_content() == content


# def test_alert_on_windows(monkeypatch):
#     if util.PLATFORM == 'Windows':
#         assert True
#         return
#     monkeypatch.setattr(platform, 'system', lambda: 'Windows')
#     monkeypatch.setattr('os.system', lambda cmd: None)
#     assert util.alert('Content', 'Title', 'Finish') == ['mshta', 'vbscript:Execute("msgbox ""Content"", 0,""Title"":Finish")']


# def test_alert_on_darwin(monkeypatch):
#     if util.PLATFORM == 'Darwin':
#         assert True
#         return
#     monkeypatch.setattr(platform, 'system', lambda: 'Darwin')
#     monkeypatch.setattr(subprocess, 'run', lambda cmd: None)
#     assert util.alert('Content') == ['osascript', '-e', 'display alert "Debug" message "Content"']


# def test_alert_on_other_platform(monkeypatch):
#     monkeypatch.setattr(platform, 'system', lambda: 'Other')
#     monkeypatch.setattr(subprocess, 'run', lambda cmd: None)
#     assert util.alert('Content', 'Title', 'Action') == ['echo', 'Title: Content: Action']


def test_convert_to_wine_path():
    path = '/path/to/my/file'
    assert util.convert_to_wine_path(path) == 'Z:\\path\\to\\my\\file'
    drive = 'X:'
    assert util.convert_to_wine_path(path, drive) == 'X:\\path\\to\\my\\file'
    path = '~/my/file'
    assert util.convert_to_wine_path(path) == 'Y:\\my\\file'
    drive = 'H:'
    assert util.convert_to_wine_path(path, drive) == 'H:\\my\\file'


def test_convert_from_wine_path():
    path = ' Z:\\path\\to\\my\\file   '
    assert util.convert_from_wine_path(path) == '/path/to/my/file'
    path = 'Y:\\my\\file'
    expected = '~/my/file' if util.PLATFORM == 'Windows' else f'{os.environ.get("HOME")}/my/file'
    assert util.convert_from_wine_path(path) == expected
    path = 'X:\\my\\file'
    assert util.convert_from_wine_path(path) == path


def test_kill_process_by_name_windows():
    if util.PLATFORM != 'Windows':
        assert True
        return
    long_proc = osp.join(_org_dir, proc_name := 'kkpyutil_proc')
    cmd = [long_proc]
    util.run_daemon(cmd)
    ret = util.kill_process_by_name(proc_name + '.exe')
    assert ret == 3
    ret = util.kill_process_by_name(proc_name + '.exe', True)
    assert ret == 0
    ret = util.kill_process_by_name('csrss.exe', True)
    assert ret == 2


def test_kill_process_by_name_macos():
    if util.PLATFORM != 'Darwin':
        assert True
        return
    long_proc = osp.join(_org_dir, 'kill_this.sh')
    util.run_daemon([long_proc])
    # let daemon start
    time.sleep(0.1)
    ret = util.kill_process_by_name('sleep')
    assert ret == 0
    long_proc = osp.join(_org_dir, 'kill_this.sh')
    util.run_daemon([long_proc])
    ret = util.kill_process_by_name('sleep', True)
    assert ret == 0
    ret = util.kill_process_by_name('missing_proc')
    assert ret == 1
    ret = util.kill_process_by_name('missing_proc', True)
    assert ret == 1
    ret = util.kill_process_by_name('mdworker', True)
    assert ret == 2


def test_init_translator():
    """
    locale folder must contain .mo files
    """
    locale_dir = osp.join(_org_dir, 'locale')
    trans = util.init_translator(locale_dir)
    assert isinstance(trans, types.MethodType)
    trans = util.init_translator(locale_dir, langs=['en', 'zh'])
    assert isinstance(trans, types.MethodType)
    # hit exception with empty folder
    locale_dir = osp.join(_gen_dir, 'locale')
    trans = util.init_translator(locale_dir, langs=['en', 'zh'])
    assert issubclass(trans, str)
    util.safe_remove(_gen_dir)


def test_match_files_except_lines():
    file1 = osp.abspath(f'{_org_dir}/match_files/ours.txt')
    file2 = osp.abspath(f'{_org_dir}/match_files/theirs.txt')
    assert util.match_files_except_lines(file1, file2, excluded=[2, 3])


def test_rerunlock_class(monkeypatch):
    def _mock_os_remove(*args, **kwargs):
        raise Exception("Mocked exception")

    lock_file = osp.join(util.get_platform_tmp_dir(), '_util', 'lock_test.0.lock.json')
    util.safe_remove(lock_file)
    run_lock = util.RerunLock(name='test')
    run_lock.lock()
    assert glob.glob(osp.join(util.get_platform_tmp_dir(), '_util', 'lock_test.*.lock.json'))
    run_lock.unlock()
    assert not osp.isfile(lock_file)
    # reenter
    util.touch(lock_file)
    assert not run_lock.lock()
    util.safe_remove(lock_file)
    # unlock fallthrough
    assert not run_lock.unlock()
    # unknown exceptions
    monkeypatch.setattr("os.remove", _mock_os_remove)
    assert not run_lock.unlock()
    # handle signal
    # Create a mock instance of the RerunLock class
    run_lock.logger = um.Mock()  # Mock the logger
    # Create a mock frame object for testing
    mock_frame = um.Mock()
    # Call the handle_signal method with a mocked signal and frame
    with pytest.raises(RuntimeError) as exc_info:
        run_lock.handle_signal(signal.SIGINT, mock_frame)
    assert str(exc_info.value) == f"Terminated due to signal: {signal.Signals(signal.SIGINT).name}; Will unlock"


def test_rerun_lock(monkeypatch):
    @util.rerun_lock('test', _gen_dir)
    def _worker():
        assert osp.isfile(glob.glob(osp.abspath(f'{_gen_dir}/lock_test*json'))[0])
        util.touch(osp.join(_gen_dir, 'entered'))
        print('entered')

    @util.rerun_lock('test', _gen_dir)
    def _mock_ctrl_c():
        raise KeyboardInterrupt

    @util.rerun_lock('test', _gen_dir)
    def _mock_misc_exception():
        raise Exception

    init = osp.join(_org_dir, 'exclusive.py')
    reenter = osp.join(_org_dir, 'reenter.py')
    lockfile = osp.join(util.get_platform_tmp_dir(), '_util', f'lock_test_rerun_lock.json')
    save = osp.join(util.get_platform_tmp_dir(), '_util', f'run_exclusive_1.json')
    for file in (save, lockfile):
        util.safe_remove(file)
    cmd = ['poetry', 'run', 'python', init, '3']
    proc1 = util.run_daemon(cmd, cwd=_org_dir)
    time.sleep(1)
    # assert osp.isfile(lockfile)
    # run a second instance before the first finishes (bg)
    cmd2 = ['poetry', 'run', 'python', reenter, '1']
    proc2 = util.run_cmd(cmd2, cwd=_org_dir, useexception=True)
    assert not osp.isfile(save)
    assert 'is locked by processes' in proc2.stderr.decode(util.TXT_CODEC)
    proc1.communicate()
    assert not osp.isfile(lockfile)
    for file in (save, lockfile):
        util.safe_remove(file)
    # decorator
    for lock_file in glob.glob(osp.abspath(f'{_gen_dir}/lock_test*json')):
        util.safe_remove(lock_file)
    _worker()
    assert not glob.glob(osp.abspath(f'{_gen_dir}/lock_test*json'))
    util.safe_remove(_gen_dir)
    # lock it in advance
    lock_file = osp.join(_gen_dir, f'lock_test.{os.getpid()}.lock.json')
    util.touch(lock_file)
    _worker()
    assert not osp.isfile(osp.join(_gen_dir, 'entered'))
    # ctrl+c term
    util.safe_remove(lock_file)
    try:
        _mock_ctrl_c()
    except KeyboardInterrupt:
        assert not osp.isfile(lock_file)
    util.safe_remove(_gen_dir)
    # other exception term
    util.safe_remove(lock_file)
    try:
        _mock_misc_exception()
    except Exception:
        assert not osp.isfile(lock_file)
    util.safe_remove(_gen_dir)


def test_await_while():
    class Condition:
        def __init__(self, *args, **kwargs):
            self.timerMs = 0

        def met(self):
            return self.timerMs > 3000

        def update(self):
            self.timerMs += 10

    cond = Condition()
    assert not cond.met()
    assert util.await_while(cond, 3100, 10)
    assert cond.met()


def test_await_lockfile_until_appear():
    def _delayed_lock():
        time.sleep(2)
        os.makedirs(lock_dir, exist_ok=True)

    # until gone
    lock_dir = _gen_dir
    util.safe_remove(lock_dir)
    th = threading.Thread(target=_delayed_lock)
    th.start()
    util.await_lockfile(lock_dir, until_gone=False)
    th.join()
    assert osp.isdir(lock_dir)
    util.safe_remove(lock_dir)


@pytest.mark.skipif(util.PLATFORM == 'Windows', reason='hangs during batch-testing on Windows, although passes in all other cases')
def test_await_lockfile_until_gone():
    def _delayed_unlock():
        time.sleep(2)
        util.safe_remove(lock_dir)

    # until gone
    lock_dir = _gen_dir
    os.makedirs(lock_dir, exist_ok=True)
    th = threading.Thread(target=_delayed_unlock)
    th.start()
    util.await_lockfile(lock_dir)
    th.join()
    assert not osp.isdir(lock_dir)


def test_append_to_os_paths():
    path_var = 'Path' if util.PLATFORM == 'Windows' else 'PATH'
    bin_dir = 'my_bin'
    os.environ[path_var].replace(bin_dir, '')
    # in-mem only
    util.append_to_os_paths(bin_dir, inmemonly=True)
    assert os.environ[path_var].endswith(f'{os.pathsep}{bin_dir}')
    util.remove_from_os_paths(bin_dir, inmemonly=True)
    # persistent
    if util.PLATFORM == 'Darwin':
        cfg_file = os.path.expanduser('~/.bash_profile' if os.getenv('SHELL') == '/bin/bash' else '~/.zshrc')
        util.append_to_os_paths(bin_dir, inmemonly=False)
        lines = util.load_lines(cfg_file, rmlineend=True)
        assert lines[-2].endswith(f'{os.pathsep}{bin_dir}"')
        util.remove_from_os_paths(bin_dir)
        return
    # TODO: support windows after imp. low-level regedit wrapper


def test_prepend_to_os_paths():
    path_var = 'Path' if util.PLATFORM == 'Windows' else 'PATH'
    bin_dir = 'my_bin'
    os.environ[path_var].replace(bin_dir, '')
    # in-mem only
    util.prepend_to_os_paths(bin_dir, inmemonly=True)
    assert os.environ[path_var].startswith(bin_dir)
    util.remove_from_os_paths(bin_dir, inmemonly=True)
    # persistent
    if util.PLATFORM == 'Darwin':
        cfg_file = os.path.expanduser('~/.bash_profile' if os.getenv('SHELL') == '/bin/bash' else '~/.zshrc')
        util.prepend_to_os_paths(bin_dir, inmemonly=False)
        lines = util.load_lines(cfg_file, rmlineend=True)
        assert lines[-2].startswith(f'export PATH="{bin_dir}{os.pathsep}')
        util.remove_from_os_paths(bin_dir)
    # TODO: support windows after imp. low-level regedit wrapper


def test_run_cmd():
    """
    only test rare code paths
    """
    # child cmd exception
    py = 'python' if util.PLATFORM == 'Windows' else 'python3'
    cmd = [py, osp.join(_org_dir, 'my_cmd.py'), 'suberr']
    with pytest.raises(subprocess.CalledProcessError):
        util.run_cmd(cmd)
    proc = util.run_cmd(cmd, useexception=False)
    assert proc.returncode == 1
    # generic exception
    cmd = ['missing']
    with pytest.raises(Exception) as e:
        util.run_cmd(cmd, useexception=True)
    # no exception
    proc = util.run_cmd(cmd, useexception=False)
    assert proc.returncode == 2
    err_log = proc.stderr.decode(util.LOCALE_CODEC)
    assert 'missing' in err_log or '[WinError 2]' in err_log
    cmd = [py, osp.join(_org_dir, 'my_cmd.py'), 100]
    proc = util.run_cmd(cmd, useexception=False)
    assert proc.returncode == 0, 'expected util.run_cmd() to have converted number to str'


def test_run_daemon():
    ls = 'dir' if util.PLATFORM == 'Windows' else 'ls'
    use_shell = util.PLATFORM == 'Windows'
    proc = util.run_daemon([ls], shell=use_shell, useexception=False)
    proc.communicate()
    assert proc.returncode == 0
    # no exception
    cmd = ['missing']
    proc = util.run_daemon(cmd, useexception=False)
    assert proc.returncode == 2
    err_log = proc.stderr.decode(util.LOCALE_CODEC)
    assert 'missing' in err_log or '[WinError 2]' in err_log
    # child cmd exception
    # generic exception
    cmd = ['poetry', 'run', 'python', '-c', 'raise Exception("failed")']
    cmd = ['missing']
    with pytest.raises(Exception) as e:
        util.run_daemon(cmd, useexception=True)


def test_watch_cmd():
    # Prepare logger if necessary (here we use Python's built-in logging)
    import logging
    logger = logging.getLogger("watch_cmd_test")
    logging.basicConfig(level=logging.INFO)
    # The command to run the test_output.py script
    py = 'python' if util.PLATFORM == 'Windows' else 'python3'
    cmd = [py, osp.join(_org_dir, 'child_proc_prints.py'), 'default']
    # Run watch_cmd and observe the real-time output
    proc = util.watch_cmd(cmd, cwd=osp.abspath(f'{_case_dir}/..'), logger=logger, verbose=True)
    assert proc.returncode == 0
    assert proc.stdout.decode(util.LOCALE_CODEC) == f"Starting...{os.linesep}stdout: Count 1{os.linesep}stdout: Count 2{os.linesep}Ending...{os.linesep}"
    assert proc.stderr.decode(util.LOCALE_CODEC) == ''
    cmd = [py, osp.join(_org_dir, 'child_proc_prints.py'), 'suberr']
    proc = util.watch_cmd(cmd, cwd=osp.abspath(f'{_case_dir}/..'), logger=logger, useexception=False)
    assert proc.returncode == 1
    assert 'CalledProcessError' in proc.stderr.decode(util.LOCALE_CODEC)
    cmd = ['missing']
    with pytest.raises(FileNotFoundError):
        util.watch_cmd(cmd, useexception=True)
    proc = util.watch_cmd(cmd, useexception=False)
    assert proc.returncode == 2
    err_log = proc.stderr.decode(util.LOCALE_CODEC)
    assert 'missing' in err_log or '[WinError 2]' in err_log


def test_extract_call_args():
    src_file = osp.join(_org_dir, 'ast_test.py')
    # missing caller
    func_calls, method_calls = util.extract_call_args(src_file, 'missing', 'my_func')
    assert not func_calls
    assert not method_calls
    # missing callee
    func_calls, method_calls = util.extract_call_args(src_file, 'main', 'missing')
    assert not func_calls
    assert not method_calls
    # called by function
    func_calls, method_calls = util.extract_call_args(src_file, 'main', 'my_func')
    # breakpoint()
    assert func_calls == [{'args': [100, 0.5], 'kwargs': {'attr': 'Caller', 'cls': 'Caller', 'lst': [3, 4], 's': 'bar', 'unsupported': None, 'dtype': 'list[str]'}, 'lineno': 15, 'end_lineno': 15}]
    assert not method_calls
    func_calls, method_calls = util.extract_call_args(src_file, 'main', 'my_method')
    assert not func_calls
    assert method_calls == [{'args': [99, 0.99], 'kwargs': {'s': 'BAR'}, 'lineno': 17, 'end_lineno': 17}]
    # called by method
    func_calls, method_calls = util.extract_call_args(src_file, 'Caller.caller_method', 'my_func')
    assert func_calls == [{'args': [-100, 0.5], 'kwargs': {'s': 'bar'}, 'lineno': 25, 'end_lineno': 27}]
    assert not method_calls
    func_calls, method_calls = util.extract_call_args(src_file, 'Caller.caller_method', 'my_method')
    assert not func_calls
    assert method_calls == [{'args': [99, 0.99], 'kwargs': {'s': 'BAR'}, 'lineno': 29, 'end_lineno': 29}]
    # ast.Name


def test_extract_class_attributes():
    src_file = osp.join(_org_dir, 'ast_test.py')
    attrs = util.extract_class_attributes(src_file, 'MyClass')
    assert attrs == [
        {'default': 99, 'end_lineno': 42, 'lineno': 42, 'name': 'i', 'type': 'int'},
        {'default': 0.9, 'end_lineno': 44, 'lineno': 44, 'name': 'f', 'type': 'float'},
        {'default': 'FOO', 'end_lineno': 47, 'lineno': 47, 'name': 's', 'type': 'str'},
        {'default': None, 'end_lineno': 48, 'lineno': 48, 'name': 'caller', 'type': 'Caller'},
        {'default': [0, 1], 'end_lineno': 49, 'lineno': 49, 'name': 'lstInt', 'type': 'list[int]'},
        {'default': (0.8, 0.9), 'end_lineno': 50, 'lineno': 50, 'name': 'lstFloat', 'type': 'tuple[float]'},
        {'default': 'proxy', 'end_lineno': 51, 'lineno': 51, 'name': 'proxy', 'type': 'tStrProxy'},
        {'default': None, 'end_lineno': 53, 'lineno': 53, 'name': 'noneTyped', 'type': None},
        {'default': None, 'end_lineno': 55, 'lineno': 55, 'name': 'unsupported', 'type': None},
    ]
    # class not found
    attrs = util.extract_class_attributes(src_file, 'missing')
    assert not attrs
    attrs = util.extract_class_attributes(src_file, 'NoCtor')
    assert not attrs


def test_extract_local_var_assignments():
    src_file = osp.join(_org_dir, 'ast_test.py')
    assigns = util.extract_local_var_assignments(src_file, 'local_assign', 's')
    assert assigns == [
        {'end_lineno': 71, 'lineno': 71, 'name': 's', 'value': 'foo'},
        {'end_lineno': 72, 'lineno': 72, 'name': 's', 'value': 'bar'},
    ]
    assert not util.extract_local_var_assignments(src_file, 'missing', 's')
    assert not util.extract_local_var_assignments(src_file, 'local_assign', 'missing')


def test_extract_imported_modules():
    src_file = osp.join(_org_dir, 'ast_test.py')
    assert util.extract_imported_modules(src_file) == [
        'os',
        'os.path',
        'subprocess',
        'sys',
    ]


def test_extract_sourcecode_comments():
    src_file = osp.join(_org_dir, 'ast_test.py')
    assert util.extract_sourcecode_comments(src_file) == {
        '(41, 8)': '# integer',
        '(43, 8)': '# float',
        '(45, 8)': '# string',
        '(46, 8)': '# variable',
        '(52, 8)': '# None is not ast.Name',
        '(54, 8)': '# without annotation or default',
        '(66, 4)': '# comment',
        '(75, 0)': '# comment line 1',
        '(76, 0)': '# comment line 2',
        '(78, 4)': '# comment line 3',
        '(79, 12)': '# inline comment',
    }


def test_open_in_browser():
    if util.PLATFORM == 'Windows':
        path = 'C:\\Windows\\System32\\drivers\\etc\\hosts'
        assert util.open_in_browser(path, islocal=True) == path
        assert util.open_in_browser(path, islocal=False) == 'C:\\Windows\\System32\\drivers\\etc\\hosts'
        path = osp.join(_org_dir, 'open_in_browser.html')
        norm_path = path.replace('\\', '/')
        assert util.open_in_browser(path, islocal=True) == f'file:///{norm_path}'
    else:
        path = '/etc/hosts'
        assert util.open_in_browser(path, islocal=True) == 'file:///etc/hosts'
        assert util.open_in_browser(path, islocal=False) == '/etc/hosts'
        path = '/path/to/filename with spaces'
        assert util.open_in_browser(path, islocal=True) == 'file:///path/to/filename%20with%20spaces'


# @pytest.mark.skipif(_skip_slow_tests, reason=_skip_reason)
def test_open_in_editor():
    path = 'C:\\Windows\\System32\\drivers\\etc\\hosts' if util.PLATFORM == 'Windows' else '/etc/hosts'
    util.open_in_editor(path, foreground=False)
    if util.PLATFORM == 'Windows':
        try:
            util.kill_process_by_name('notepad.exe')
        except Exception:
            pass
    # folder
    util.open_in_editor(_org_dir, foreground=True)


def test_flatten_nested_lists():
    nested = [[1, 2], [3, 4], [5, 6, 7, 8], [9]]
    flat = util.flatten_nested_lists(nested)
    assert flat == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_show_results():
    # full input
    succeeded = True
    detail = """\
- detail 1
- detail 2
- detail 3"""
    advice = """\
- advice 1
- advice 2
- advice 3"""
    dryrun = False
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
*** SUCCEEDED ***

Detail:
- detail 1
- detail 2
- detail 3

Next:
- advice 1
- advice 2
- advice 3"""

    # no detail
    detail = None
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
*** SUCCEEDED ***

Detail:
- (N/A)

Next:
- advice 1
- advice 2
- advice 3"""

    # no advice
    advice = None
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
*** SUCCEEDED ***

Detail:
- (N/A)

Next:
- (N/A)"""

    # full input: failed
    succeeded = False
    detail = """\
- detail 1
- detail 2
- detail 3"""
    advice = """\
- advice 1
- advice 2
- advice 3"""
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
* FAILED *

Detail:
- detail 1
- detail 2
- detail 3

Advice:
- advice 1
- advice 2
- advice 3"""

    # no detail
    detail = ''
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
* FAILED *

Detail:
- (N/A)

Advice:
- advice 1
- advice 2
- advice 3"""

    # no advice
    advice = ''
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
* FAILED *

Detail:
- (N/A)

Advice:
- (N/A)"""

    # dryrun
    dryrun = True
    report = util.show_results(succeeded, detail, advice, dryrun)
    assert report == """
** DRYRUN **

Detail:
- (N/A)

Advice:
- (N/A)"""


def test_init_repo():
    src_file = osp.join('repo', 'app', 'src', 'my.code')
    app = util.init_repo(src_file, organization='kk', logname='mylogtitle', uselocale=True)
    assert app.ancestorDirs == [
        osp.abspath(osp.join('repo', 'app', 'src')),
        osp.abspath(osp.join('repo', 'app')),
        osp.abspath(osp.join('repo')),
    ]
    assert app.locDir == osp.abspath(osp.join('repo', 'app', 'locale'))
    assert app.srcDir == osp.abspath(osp.join('repo', 'app', 'src'))
    assert app.tmpDir == osp.abspath(osp.join('repo', 'app', 'temp'))
    assert app.testDir == osp.abspath(osp.join('repo', 'app', 'test'))
    assert app.pubTmpDir == osp.join(util.get_platform_tmp_dir(), 'kk', 'app')
    assert app.logger.name == 'mylogtitle'
    util.safe_remove(app.ancestorDirs[2])


def test_backup_file():
    file = osp.join(_gen_dir, 'my.file')
    util.touch(file)
    # non-numeric suffix
    bak = util.backup_file(file, suffix='.bak')
    assert osp.isfile(bak)
    assert bak == osp.join(_gen_dir, 'my.file.bak')
    # numeric suffix
    bak = util.backup_file(file, suffix='.3')
    assert osp.isfile(bak)
    assert bak == osp.join(_gen_dir, 'my.file.3')
    bak = util.backup_file(file, suffix='.1')
    assert osp.isfile(bak)
    assert bak == osp.join(_gen_dir, 'my.file.4')
    util.safe_remove(_gen_dir)


def test_recover_file():
    file = osp.join(_gen_dir, 'my.file')
    assert not osp.isfile(file)
    bak = osp.join(_gen_dir, 'my.file.bak')
    util.touch(bak)
    bak = util.recover_file(file, suffix='.bak')
    assert osp.isfile(file)
    assert bak == osp.join(_gen_dir, 'my.file.bak')
    util.safe_remove(_gen_dir)
    os.makedirs(_gen_dir, exist_ok=True)
    assert util.recover_file(file, suffix='.bak') is None
    util.safe_remove(_gen_dir)
    # numerical suffix
    src_baks = [osp.abspath(f'{_org_dir}/recover_from_these/{file}') for file in (
        'my.file.1',
        'my.file.2',
    )]
    os.makedirs(_gen_dir, exist_ok=True)
    for src in src_baks:
        util.copy_file(src, _gen_dir, isdstdir=True)
    to_recover = osp.join(_gen_dir, 'my.file')
    assert util.recover_file(to_recover) == osp.join(_gen_dir, 'my.file.2')
    assert util.load_lines(to_recover, rmlineend=True) == ['my.file.2']
    util.safe_remove(_gen_dir)
    # numerical suffix: failed
    os.makedirs(_gen_dir, exist_ok=True)
    util.safe_remove(_gen_dir)
    src_bak = osp.abspath(f'{_org_dir}/recover_from_these/my.file.unkownsuffix')
    util.copy_file(src_bak, _gen_dir, isdstdir=True)
    assert util.recover_file(to_recover) is None


def test_deprecate_log():
    msg = util.deprecate('old', 'new')
    assert msg == 'old is deprecated; use new instead'


def test_save_load_lines():
    lines = [
        'first line',
        'second line',
        'third line',
    ]
    file = osp.join(_gen_dir, 'lines.txt')
    util.save_lines(file, lines, addlineend=True)
    lines = util.load_lines(file, rmlineend=True)
    assert lines == [
        'first line',
        'second line',
        'third line',
    ]
    lines = util.load_lines(file)
    assert lines == [
        'first line\n',
        'second line\n',
        'third line\n',
    ]
    util.save_lines(file, lines)
    lines = util.load_lines(file, rmlineend=True)
    assert lines == [
        'first line',
        'second line',
        'third line',
    ]
    util.save_lines(file, lines, toappend=True)
    lines = util.load_lines(file, rmlineend=True)
    assert lines == [
        'first line',
        'second line',
        'third line',
        'first linesecond linethird line',
    ]
    util.save_lines(file, 'line 1', addlineend=True)
    assert util.load_lines(file, rmlineend=True) == ['line 1']
    util.safe_remove(_gen_dir)


def test_save_load_text():
    text = '\n'.join([
        'first line',
        'second line',
        'third line',
    ])
    file = osp.join(_gen_dir, 'text.txt')
    util.save_text(file, text)
    loaded = util.load_text(file)
    assert loaded == text
    util.save_text(file, text, toappend=True)
    loaded = util.load_text(file)
    assert loaded == text + text
    util.safe_remove(_gen_dir)


def test_find_duplication():
    list_with_dups = [1, 2, 3, 2, 4, 1, 5]
    assert util.find_duplication(list_with_dups) == {
        1: [0, 5],
        2: [1, 3],
    }


def test_remove_duplication():
    list_with_dups = [1, 2, 3, 2, 5, 3]
    assert (util.remove_duplication(list_with_dups)) == [1, 2, 3, 5]
    list_with_dups = [1, 5.0, 'xyz', 5.0, 5, 'xyz']
    assert (util.remove_duplication(list_with_dups)) == [1, 5.0, 'xyz']


def test_find_runs():
    list_with_runs = [1, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 10, 10]
    assert util.find_runs(list_with_runs) == [
        [1, 2, 3],
        [7, 8, 9, 10],
        [11, 12],
    ]


@pytest.mark.skipif(_skip_slow_tests, reason=_skip_reason)
def test_install_uninstall_by_macports(monkeypatch):
    if util.PLATFORM != 'Darwin':
        assert True
        return
    with pytest.raises(FileNotFoundError):
        util.install_by_macports('cowsay')
    with pytest.raises(FileNotFoundError):
        util.uninstall_by_macports('cowsay')


@pytest.mark.skipif(_skip_slow_tests, reason=_skip_reason)
def test_install_uninstall_by_homebrew(monkeypatch):
    if util.PLATFORM != 'Darwin':
        assert True
        return
    pkg, lazybin = 'fortune', 'fortune'
    util.uninstall_by_homebrew(pkg, lazybin)
    util.install_by_homebrew(pkg)
    util.install_by_homebrew('fortune', lazybin='fortune')


def test_validate_platform():
    supported = ['os1', 'os2']
    with pytest.raises(NotImplementedError):
        util.validate_platform(supported)
    supported = util.PLATFORM
    util.validate_platform(supported)


def test_touch():
    file = osp.join(_gen_dir, 'my.file')
    util.touch(file)
    assert osp.isfile(file)
    util.safe_remove(_gen_dir)


def test_lazy_load_listfile():
    lst = ['a', 'b', 'c']
    assert util.lazy_load_listfile(lst) == lst
    single_file = osp.join(_gen_dir, 'my.file')
    assert util.lazy_load_listfile(single_file) == [single_file]
    list_file = osp.join(_org_dir, 'my.list')
    assert util.lazy_load_listfile(list_file) == [
        'first line',
        'second line',
        'third line',
    ]
    list_file = [osp.join(_org_dir, 'my.list')]
    assert util.lazy_load_listfile(list_file) == [
        'first line',
        'second line',
        'third line',
    ]
    list_file = osp.join(_org_dir, 'missing.list')
    with pytest.raises(FileNotFoundError):
        util.lazy_load_listfile(list_file)


def test_normalize_paths():
    paths = [
        'c:\\path/to/file',
        '/path/to/file'
    ]
    assert util.normalize_paths(paths, mode='win') == [
        'c:\\path\\to\\file',
        '\\path\\to\\file',
    ]
    assert util.normalize_paths(paths, mode='posix') == [
        'c:/path/to/file',
        '/path/to/file',
    ]
    if util.PLATFORM == 'Windows':
        assert util.normalize_paths(paths, mode='native') == [
            'c:\\path\\to\\file',
            '\\path\\to\\file',
        ]
    else:
        assert util.normalize_paths(paths, mode='native') == [
            'c:/path/to/file',
            '/path/to/file',
        ]
    with pytest.raises(NotImplementedError):
        util.normalize_paths(paths, mode='invalid')


def test_lazy_load_filepaths():
    if util.PLATFORM == 'Windows':
        single_path = 'c:\\path\\to\\file'
        assert util.lazy_load_filepaths(single_path) == single_path
        paths = ['c:\\path\\to\\file1', 'c:\\path\\to\\file2', 'c:\\path\\to\\file3']
        assert util.lazy_load_filepaths(single_path) == paths
        list_file = osp.join(_org_dir, 'files_abs_win.list')
        assert util.lazy_load_filepaths(list_file) == [
            'c:\\path\\to\\file1',
            'D:\\path\\to\\file2',
        ]
        list_file = osp.join(_org_dir, 'files_rel_win.list')
        assert util.lazy_load_filepaths(list_file, root='c:\\root') == [
            'c:\\root\\rel\\to\\file1',
            'c:\\root\\rel\\to\\file2',
        ]
    else:
        single_path = '/path/to/file'
        assert util.lazy_load_filepaths(single_path) == ['/path/to/file']
        single_path = ['/path/to/file1', '/path/to/file2', '/path/to/file3']
        assert util.lazy_load_filepaths(single_path) == single_path
        list_file = osp.join(_org_dir, 'files_abs_posix.list')
        assert util.lazy_load_filepaths(list_file) == [
            '/path/to/file1',
            '/path/to/file2',
        ]
        list_file = [osp.join(_org_dir, 'files_abs_posix.list')]
        assert util.lazy_load_filepaths(list_file) == [
            '/path/to/file1',
            '/path/to/file2',
        ]
        list_file = osp.join(_org_dir, 'files_rel_posix.list')
        assert util.lazy_load_filepaths(list_file, root='/root') == [
            '/root/rel/to/file1',
            '/root/rel/to/file2',
        ]
    with pytest.raises(FileNotFoundError):
        util.lazy_load_filepaths('missing.list')


def test_read_link():
    if util.PLATFORM == 'Windows':
        lnk = osp.join(_org_dir, 'lines.txt.symlink')
        assert util.read_link(lnk) == lnk
        lnk = osp.join(_org_dir, 'lines.txt.lnk')
        assert util.read_link(lnk) == 'D:\\kakyo\\_dev\\kkpyutil\\test\\_org\\lines.txt'
    else:
        lnk = osp.join(_org_dir, 'lines.txt.symlink')
        assert util.read_link(lnk) == '/Users/kakyo/Desktop/_dev/kkpyutil/test/_org/lines.txt'
        lnk = osp.join(_org_dir, 'lines.txt.lnk')
        assert util.read_link(lnk) == lnk


def test_is_link():
    file = osp.join(_org_dir, 'lines.txt')
    if util.PLATFORM == 'Windows':
        shortcut = osp.join(_org_dir, 'lines.txt.lnk')
        assert util.is_link(shortcut)
    else:
        symlink = osp.join(_org_dir, 'lines.txt.symlink')
        assert util.is_link(symlink)
    assert not util.is_link(file)


def test_raise_error():
    errcls = NotImplementedError
    detail = '- This is a test error'
    advice = '- Fix it'
    with pytest.raises(NotImplementedError) as diagnostics:
        util.raise_error(errcls, detail, advice)
    assert str(diagnostics.value) == """\
Detail:
- This is a test error

Advice:
- Fix it"""


def test_sanitize_text_as_path():
    path_part = 'tab: 天哪*?"<\\/\x00\x1F'
    assert util.sanitize_text_as_path(path_part) == 'tab_ 天哪________'


def test_remove_file():
    missing = osp.join(_gen_dir, 'missing.file')
    util.remove_file(missing)


def test_find_first_line_in_range():
    lines = """
keyword: other stuff
...... ...... ...... ...... ......
...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword') == 1
    lines = """
other stuff: keyword
...... ...... ...... ...... ......
...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword', algo='endswith') == 1
    lines = """
other stuff: keyword: other stuff
...... ...... ...... ...... ......
...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword', algo='contains') == 1
    lines = """
...... ...... ...... ...... ......
...... ...... ...... ...... ......
keyword: other stuff
...... ...... ...... ...... ......
...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword', linerange=(3,)) == 3
    lines = """0
1...... ...... ...... ...... ......
2...... ...... ...... ...... ......
keyword: other stuff
4...... ...... ...... ...... ......
5...... ...... ...... ...... ......
6...... ...... ...... ...... ......
keyword: other stuff
...... ...... ...... ...... ......
...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword', linerange=(4,)) == 7

    lines = """
...... ...... ...... ...... ......
...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword') is None
    lines = """0
1...... ...... ...... ...... ......
2...... ...... ...... ...... ......
keyword: other stuff
4...... ...... ...... ...... ......
5...... ...... ...... ...... ......
6...... ...... ...... ...... ......
keyword: other stuff
8...... ...... ...... ...... ......
9...... ...... ...... ...... ......
""".split('\n')
    assert util.find_first_line_in_range(lines, 'keyword', linerange=(8,)) is None

    lines = """0
keyword: other stuff
2...... ...... ...... ...... ......
3...... ...... ...... ...... ......
""".split('\n')
    with pytest.raises(AssertionError):
        util.find_first_line_in_range(lines, 'keyword', linerange=(2, 0))

    lines = """
keyword: other stuff
...... ...... ...... ...... ......
...... ...... ...... ...... ......
"""
    with pytest.raises(TypeError):
        util.find_first_line_in_range(lines, 'keyword')


def test_substitute_lines_between_cues():
    org_lines = """\
# some text below are wrapped in tags
    <START
    this will be replaced with:
    - line 1
    - line 2
    END>
<START2
    this will be replaced with:
    - line 1
    - line 2
END2>
""".splitlines()
    backup_lines = copy.deepcopy(org_lines)
    inserts = """\
- line 1
- line 2
""".splitlines()
    # inserted line range is the result range, not original line range that got replaced
    line_range = util.substitute_lines_between_cues(inserts, org_lines, '<START', 'END>', withindent=True)
    assert line_range == [2, 3]
    assert '\n'.join(org_lines) == """\
# some text below are wrapped in tags
    <START
    - line 1
    - line 2
    END>
<START2
    this will be replaced with:
    - line 1
    - line 2
END2>"""
    line_range = util.substitute_lines_between_cues(inserts, org_lines, '<START2', 'END2>', startlineno=line_range[1], withindent=True)
    assert line_range == [6, 7]
    assert '\n'.join(org_lines) == """\
# some text below are wrapped in tags
    <START
    - line 1
    - line 2
    END>
<START2
- line 1
- line 2
END2>"""
    org_lines = copy.deepcopy(backup_lines)
    assert util.substitute_lines_between_cues(inserts, org_lines, '<MISSING', 'END>', withindent=True) == (None, None)
    org_lines = copy.deepcopy(backup_lines)
    assert util.substitute_lines_between_cues(inserts, org_lines, '<START', 'MISSING>', withindent=True) == (2, None)
    # remove cues
    org_lines = copy.deepcopy(backup_lines)
    assert util.substitute_lines_between_cues(inserts, org_lines, '<START', 'END>', withindent=True, removecues=True) == [1, 2]
    # skip dups in lines to insert
    inserts = """\
- line 1
- line 1
""".splitlines()
    org_lines = copy.deepcopy(backup_lines)
    assert util.substitute_lines_between_cues(inserts, org_lines, '<START', 'END>', withindent=True, removecues=False, skipdups=True) == [2, 2]


def test_substitute_lines_in_file():
    org_file = osp.join(_org_dir, fn := 'subs_keywords.txt')
    util.copy_file(org_file, _gen_dir, isdstdir=True)
    dst_file = osp.join(_gen_dir, fn)
    inserts = """\
- line 1
- line 2
""".splitlines()
    assert util.substitute_lines_in_file(inserts, dst_file, '<START', 'END>', withindent=True, removecues=False) == [2, 3]


def test_wrap_lines_with_tags():
    wrapped = util.wrap_lines_with_tags(['    wrap me'], '<START', 'END>', withindent=True)
    # breakpoint()
    assert wrapped == """\
    <START
    wrap me
    END>
""".splitlines()


def test_convert_compound_cases():
    with pytest.raises(AssertionError):
        util.convert_compound_cases('hello_world', style='missing')
    with pytest.raises(KeyError):
        util.convert_compound_cases('hello_world', instyle=',,,')
    assert util.convert_compound_cases('hello_world', style='oneword') == 'helloworld'
    assert util.convert_compound_cases('hello_world', style='ONEWORD') == 'HELLOWORLD'
    assert util.convert_compound_cases('hello_world', style='SNAKE') == 'HELLO_WORLD'
    assert util.convert_compound_cases('hello_world', style='kebab') == 'hello-world'
    assert util.convert_compound_cases('hello_world', style='title') == 'Hello World'
    assert util.convert_compound_cases('hello_world', style='phrase') == 'hello world'
    assert util.convert_compound_cases('hello_world', style='camel') == 'helloWorld'
    assert util.convert_compound_cases('hello_world', style='pascal') == 'HelloWorld'

    assert util.convert_compound_cases('helloWorld1', style='oneword') == 'helloworld1'
    assert util.convert_compound_cases('helloWorld1', style='ONEWORD') == 'HELLOWORLD1'
    assert util.convert_compound_cases('helloWorld1', style='SNAKE') == 'HELLO_WORLD1'
    assert util.convert_compound_cases('helloWorld1', style='kebab') == 'hello-world1'
    assert util.convert_compound_cases('helloWorld1', style='title') == 'Hello World1'
    assert util.convert_compound_cases('helloWorld1', style='phrase') == 'hello world1'
    assert util.convert_compound_cases('helloWorld1', style='camel') == 'helloWorld1'
    assert util.convert_compound_cases('helloWorld1', style='pascal') == 'HelloWorld1'

    assert util.convert_compound_cases('Hello1World1', style='oneword') == 'hello1world1'
    assert util.convert_compound_cases('Hello1World1', style='ONEWORD') == 'HELLO1WORLD1'
    assert util.convert_compound_cases('Hello1World1', style='SNAKE') == 'HELLO1_WORLD1'
    assert util.convert_compound_cases('Hello1World1', style='kebab') == 'hello1-world1'
    assert util.convert_compound_cases('Hello1World1', style='title') == 'Hello1 World1'
    assert util.convert_compound_cases('Hello1World1', style='phrase') == 'hello1 world1'
    assert util.convert_compound_cases('Hello1World1', style='camel') == 'hello1World1'
    assert util.convert_compound_cases('Hello1World1', style='pascal') == 'Hello1World1'

    assert util.convert_compound_cases('Hello1-World1', style='oneword') == 'hello1world1'
    assert util.convert_compound_cases('Hello1-World1', style='ONEWORD') == 'HELLO1WORLD1'
    assert util.convert_compound_cases('Hello1-World1', style='SNAKE') == 'HELLO1_WORLD1'
    assert util.convert_compound_cases('Hello1-World1', style='kebab') == 'Hello1-World1'
    assert util.convert_compound_cases('Hello1-World1', style='title') == 'Hello1 World1'
    assert util.convert_compound_cases('Hello1-World1', style='phrase') == 'hello1 world1'
    assert util.convert_compound_cases('Hello1-World1', style='camel') == 'hello1World1'
    assert util.convert_compound_cases('Hello1-World1', style='pascal') == 'Hello1World1'

    assert util.convert_compound_cases('helloWorld', style='pascal', instyle='camel') == 'HelloWorld'
    assert util.convert_compound_cases('HelloWorld', style='SNAKE', instyle='pascal') == 'HELLO_WORLD'
    assert util.convert_compound_cases('Hello World', style='camel', instyle='title') == 'helloWorld'
    assert util.convert_compound_cases('hello world', style='camel', instyle='phrase') == 'helloWorld'
    assert util.convert_compound_cases('hello_world', style='camel', instyle='snake') == 'helloWorld'
    assert util.convert_compound_cases('hello-world', style='camel', instyle='kebab') == 'helloWorld'
    assert util.convert_compound_cases('hello-world', style='kebab', instyle='kebab') == 'hello-world'
    assert util.convert_compound_cases('helloworld', style='camel', instyle='snake') == 'helloworld'
    assert util.convert_compound_cases('Helloworld', style='pascal', instyle='snake') == 'Helloworld'

    # ambiguity in detection and fallbacks
    assert util.convert_compound_cases('page1', style='snake') == 'page1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='SNAKE') == 'PAGE1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='camel') == 'page1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='kebab') == 'page1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='phrase') == 'page1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='oneword') == 'page1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='ONEWORD') == 'PAGE1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='pascal') == 'Page1', 'taken as camel'
    assert util.convert_compound_cases('page1', style='title') == 'Page1', 'taken as camel'
    assert util.convert_compound_cases('p1_xyz', style='camel') == 'p1Xyz', 'snakecase with mixed alphanumeric sections should be recognized'


def test_append_lineends_to_lines():
    assert util.append_lineends_to_lines(
        ['line 1', 'line 2'],
        'windows',
    ) == [
               'line 1\r\n',
               'line 2\r\n',
           ]
    assert util.append_lineends_to_lines(
        ['line 1', 'line 2'],
        'win',
    ) == [
               'line 1\r\n',
               'line 2\r\n',
           ]
    assert util.append_lineends_to_lines(
        ['line 1', 'line 2'],
    ) == [
               'line 1\n',
               'line 2\n',
           ]


def test_zip_unzip_dir():
    src_dir = osp.join(_org_dir, target := 'zip_this')
    util.safe_remove(_gen_dir)
    os.makedirs(_gen_dir, exist_ok=True)
    shutil.copytree(src_dir, dst_dir := osp.join(_gen_dir, target), dirs_exist_ok=True)
    expected_zip = osp.join(_org_dir, f'{target}.zip')
    assert util.zip_dir(src_dir) == expected_zip
    util.safe_remove(_gen_dir)
    assert util.unzip_dir(expected_zip, _gen_dir) == osp.join(_gen_dir, target)
    same_level = osp.join(_org_dir, target)
    assert util.unzip_dir(expected_zip) == same_level
    util.safe_remove(_gen_dir)
    util.safe_remove(expected_zip)


def test_compare_textfiles():
    file1 = osp.join(_org_dir, 'compare_these', 'file1.txt')
    file2 = osp.join(_org_dir, 'compare_these', 'file2.txt')
    assert util.compare_textfiles(file1, file2, showdiff=True, ignoredlinenos=[1])


def test_pack_obj():
    # namespace
    obj = types.SimpleNamespace(n=1, s='hello', f=9.99, l=[1, 2, 3])
    topic = 'pkg'
    packed = util.pack_obj(obj, topic)
    assert packed == '<KK-ENV>{"payload": {"n": 1, "s": "hello", "f": 9.99, "l": [1, 2, 3]}, "topic": "pkg"}</KK-ENV>'
    # custom tags
    envelope = ('<MyEnv>', '</MyEnv>')
    packed = util.pack_obj(obj, topic, envelope=envelope)
    assert packed == '<MyEnv>{"payload": {"n": 1, "s": "hello", "f": 9.99, "l": [1, 2, 3]}, "topic": "pkg"}</MyEnv>'
    # default topic
    packed = util.pack_obj(obj)
    assert packed == '<KK-ENV>{"payload": {"n": 1, "s": "hello", "f": 9.99, "l": [1, 2, 3]}, "topic": "SimpleNamespace"}</KK-ENV>'

    # custom class
    class MyClass:
        def __init__(self, *args, **kwargs):
            self.n: int = 1
            self.s: str = 'hello'
            self.f: float = 9.99
            self.l: list[int] = [1, 2, 3]

        def main(self):
            pass

    obj = MyClass()
    packed = util.pack_obj(obj, classes=(MyClass,))
    assert packed == '<KK-ENV>{"payload": {"n": 1, "s": "hello", "f": 9.99, "l": [1, 2, 3]}, "topic": "MyClass"}</KK-ENV>'


def test_lazy_extend_sys_path():
    # single path
    util.lazy_extend_sys_path([_org_dir])
    assert _org_dir in sys.path
    # duplicate
    util.lazy_extend_sys_path([_org_dir])
    assert sys.path.count(_org_dir) == 1


def test_lazy_prepend_sys_path():
    # single path
    util.lazy_prepend_sys_path([_org_dir])
    assert _org_dir == sys.path[0]
    # duplicate
    util.lazy_prepend_sys_path([_org_dir])
    assert sys.path.count(_org_dir) == 1


def test_lazy_remove_from_sys_path():
    # single path
    util.lazy_extend_sys_path([_org_dir])
    assert _org_dir in sys.path
    util.lazy_remove_from_sys_path([_org_dir])
    assert _org_dir not in sys.path
    # missing
    util.lazy_remove_from_sys_path(['missing'])
    assert sys.path.count('missing') == 0


def test_safe_import_module():
    at = util.safe_import_module('ast_test', _org_dir, reload=True)
    assert at is not None


def test_get_parent_dirs():
    assert util.get_parent_dirs(_org_dir, subs=['sub']) == (_case_dir, _src_dir, osp.join(_src_dir, 'sub'))


def test_get_ancestor_dirs():
    file = osp.join(_org_dir, 'exclusive.py')
    dirs = util.get_ancestor_dirs(file, depth=3)
    assert dirs == [_org_dir, _case_dir, _src_dir]
    dirs = util.get_ancestor_dirs(_org_dir, depth=2)
    assert dirs == [_case_dir, _src_dir]
    dirs = util.get_ancestor_dirs(_org_dir, depth=1)
    assert dirs == [_case_dir]


def test_get_child_dirs():
    if util.PLATFORM == 'Windows':
        root = r'c:\path\to\root'
        subs = ('ci', 'src', 'test')
        assert util.get_child_dirs(root, subs) == [
            r'c:\path\to\root\ci',
            r'c:\path\to\root\src',
            r'c:\path\to\root\test',
        ]
    else:
        root = '/path/to/root'
        subs = ('ci', 'src', 'test')
        assert util.get_child_dirs(root, subs) == [
            '/path/to/root/ci',
            '/path/to/root/src',
            '/path/to/root/test',
        ]


def test_get_drivewise_commondirs():
    # single path
    if is_posix := util.PLATFORM != 'Windows':
        abs_paths = ['/path/to/dir1/file1']
        assert util.get_drivewise_commondirs(abs_paths) == {'/': '/path/to/dir1'}
        rel_paths = ['path/to/dir1/file1']
        assert util.get_drivewise_commondirs(rel_paths) == {'': 'path/to/dir1'}
    else:
        abs_paths = ['C:\\path\\to\\dir1\\file1.ext']
        assert util.get_drivewise_commondirs(abs_paths) == {'c:': 'c:\\path\\to\\dir1'}
        rel_paths = ['path\\to\\dir1\\file1']
        assert util.get_drivewise_commondirs(rel_paths) == {'': 'path\\to\\dir1'}
    # many paths
    if is_posix := util.PLATFORM != 'Windows':
        abs_paths = ['/path/to/dir1/file1', '/path/to/dir2/', '/path/to/dir3/dir4/file2']
        assert util.get_drivewise_commondirs(abs_paths) == {'/': '/path/to'}
        rel_paths = ['path/to/dir1/file1', 'path/to/dir2/', 'path/to/dir3/dir4/file2']
        assert util.get_drivewise_commondirs(rel_paths) == {'': 'path/to'}
        # case-sensitive
        rel_paths = ['path/TO/dir1/file1', 'path/to/dir2/', 'path/to/dir3/dir4/file2']
        assert util.get_drivewise_commondirs(rel_paths) == {'': 'path'}
    else:
        abs_paths = [
            'C:\\path\\to\\dir1\\file1.ext',
            'c:\\path\\to\\dir2\\',
            'd:\\path\\to\\dir3\\dir4\\file2.ext',
            'D:\\path\\to\\dir5\\dir6\\file3.ext',
            'e:\\path\\to\\file8.ext',
            '\\\\Network\\share\\path\\to\\dir7\\file4.ext',
            '\\\\network\\share\\path\\to\\dir7\\dir8\\file5.ext',
            'path\\to\\dir9\\file6.ext',
            'path\\to\\dir9\\file7.ext',
        ]
        assert util.get_drivewise_commondirs(abs_paths) == {
            'c:': 'c:\\path\\to',
            'd:': 'd:\\path\\to',
            'e:': 'e:\\path\\to',
            '\\\\network\\share': '\\\\network\\share\\path\\to\\dir7',
            '': 'path\\to\\dir9'
        }
        # case-insensitive
        rel_paths = [
            'path\\to\\dir1\\file1.ext',
            '\\Path\\to\\dir2\\',
            'path\\To\\dir1\\dir4\\file2.ext'
        ]
        assert util.get_drivewise_commondirs(rel_paths) == {'': 'path\\to'}


def test_split_platform_drive():
    if util.PLATFORM == 'Windows':
        path = osp.normpath('C:/path/to/dir1/file1')
        assert util.split_platform_drive(path) == ('c:', osp.normpath('/path/to/dir1/file1'))
        path = osp.normpath('path/to/dir1/file1')
        assert util.split_platform_drive(path) == ('', osp.normpath('path/to/dir1/file1'))
    else:
        path = '/path/to/dir1/file1'
        assert util.split_platform_drive(path) == ('/', 'path/to/dir1/file1')
        path = 'path/to/dir1/file1'
        assert util.split_platform_drive(path) == ('', 'path/to/dir1/file1')


def test_is_float_text():
    assert util.is_float_text('1.0')
    assert util.is_float_text('1.0e-3')
    assert util.is_float_text('1.0e+3')
    assert not util.is_float_text('100')
    assert not util.is_float_text('hello')
    # uuid
    assert not util.is_float_text('e6a6dd92-5b96-4a09-9cc4-d44153b900a4')


def test_compare_dsv_lines():
    line1 = 'length b c'
    line2 = 'length b c'
    assert util.compare_dsv_lines(line1, line2)
    line1 = 'length b c'
    line2 = '  length b c   '
    assert not util.compare_dsv_lines(line1, line2, striptext=False)
    assert util.compare_dsv_lines(line1, line2, striptext=True)
    line1 = 'length, b, c'
    line2 = '  length,b,c   '
    assert util.compare_dsv_lines(line1, line2, delim=',', striptext=True)
    line1 = 'length 1.23458 e6a6dd92-5b96-4a09-9cc4-d44153b900a4'
    line2 = 'length 1.23459 e3016d69-cb30-4eb1-9f93-bb28621aba28'
    assert not util.compare_dsv_lines(line1, line2), 'float mismatch, uuid mismatch'
    assert util.compare_dsv_lines(line1, line2, float_rel_tol=1e-5, float_abs_tol=1e-5, randomidok=True)
    line1 = 'length 35 e6a6dd92-5b96-4a09-9cc4-d44153b900a4'
    line2 = 'length 1.23459 e3016d69-cb30-4eb1-9f93-bb28621aba28'
    assert not util.compare_dsv_lines(line1, line2), 'float type mismatch'
    line1 = 'length 1.23458 10f64942-5500-11ee-8902-6298388980f0'
    line2 = 'length 1.23458 89116a26-9e15-4bec-bc1e-74003277cf83'
    assert not util.compare_dsv_lines(line1, line2), 'uuid type mismatch'
    line1 = 'length 1.23458 89116a26-9e15-4bec-bc1e-74003277cf83'
    line2 = 'length 1.23458 89116a26-9e15-4bec-bc1e-74003277cf83'
    assert util.compare_dsv_lines(line1, line2), 'uuids match'
    line1 = 'length 1.23458 10f64942-5500-11ee-8902-6298388980f0'
    line2 = 'lengthy 1.23458 89116a26-9e15-4bec-bc1e-74003277cf83'
    assert not util.compare_dsv_lines(line1, line2), 'string mismatch'


def test_copy_file():
    # bypass SameFileError
    src_file = osp.join(_org_dir, 'lines.txt')
    util.copy_file(src_file, src_file)


def test_move_file():
    util.safe_remove(_gen_dir)
    src_file = util.touch(src := osp.join(_gen_dir, 'to_move.file'))
    dst_dir = osp.join(_gen_dir, 'to_move')
    util.move_file(src_file, dst_dir, isdstdir=True)
    assert osp.isfile(dst := osp.join(dst_dir, 'to_move.file'))
    # dst exists
    src_file = util.touch(src)
    util.move_file(src_file, dst)
    src_file = util.touch(src)
    util.move_file(src_file, osp.dirname(dst), isdstdir=True)
    # no SameFileError
    src_file = util.touch(src)
    util.move_file(src_file, src_file)
    util.safe_remove(_gen_dir)


def test_compare_dirs():
    src_dir = osp.join(_org_dir, 'compare_these', 'dir1')
    dst_dir = osp.join(_org_dir, 'compare_these', 'dir1_clone')
    util.compare_dirs(src_dir, dst_dir)
    dst_dir = osp.join(_org_dir, 'compare_these', 'dir2_dir_diff')
    assert not util.compare_dirs(src_dir, dst_dir)
    dst_dir = osp.join(_org_dir, 'compare_these', 'dir3_file_diff')
    assert not util.compare_dirs(src_dir, dst_dir)
    dst_dir = osp.join(_org_dir, 'compare_these', 'dir4_dir_fuzzy')
    os.makedirs(osp.join(_org_dir, 'compare_these', 'dir4_dir_fuzzy', 'sub_suffix_ignored'), exist_ok=True)
    assert util.compare_dirs(src_dir, dst_dir, ignoreddirpatterns=['sub'])
    dst_dir = osp.join(_org_dir, 'compare_these', 'dir5_file_fuzzy')
    assert util.compare_dirs(src_dir, dst_dir, ignoredfilepatterns=['*.fuzzy'])


def test_safe_remove():
    file = osp.join(_gen_dir, 'to_remove.file')
    util.touch(file)
    assert osp.isfile(file)
    util.safe_remove(file)
    assert not osp.isfile(file)
    with tempfile.TemporaryDirectory() as d:
        assert osp.isdir(d)
        util.safe_remove(d)
        assert not osp.isdir(d)
    util.safe_remove(_gen_dir)


def test_is_non_ascii_text():
    invalid = "我是 I"
    assert util.is_non_ascii_text(invalid)
    valid = "i'm me"
    assert not util.is_non_ascii_text(valid)


def test_inspect_obj():
    class MyClass:
        def __init__(self, *args, **kwargs):
            self.n: int = 1
            self.s: str = 'hello'
            self.f: float = 9.99
            self.l: list[int] = [1, 2, 3]

        def main(self):
            pass

    obj = MyClass()
    obj_info = util.inspect_obj(obj)
    assert obj_info['type'] == 'MyClass'
    assert obj_info['attrs'] == {'n': 1, 's': 'hello', 'f': 9.99, 'l': [1, 2, 3]}
    assert '<test_default.test_inspect_obj.<locals>.MyClass object' in obj_info['repr']
    assert obj_info['details'] == ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__',
                                   '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'f', 'l', 'main', 'n', 's']
    num = 100
    assert util.inspect_obj(num) == {'type': 'int', 'attrs': {}, 'repr': '100',
                                     'details': ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__',
                                                 '__getattribute__', '__getnewargs__', '__getstate__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__',
                                                 '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__',
                                                 '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__',
                                                 '__xor__', 'as_integer_ratio', 'bit_count', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']}


def test_cache():
    src_file = osp.join(_gen_dir, 'data.json')
    retriever = util.load_json
    util.save_json(src_file, {'a': 1, 'b': 2})
    cache = util.Cache(src_file, retriever)
    retrieved = cache.retrieve()
    assert retrieved['a'] == 1 and retrieved['b'] == 2
    # cache hit
    retrieved = cache.retrieve()
    assert retrieved['a'] == 1 and retrieved['b'] == 2
    util.save_json(src_file, {'a': 1, 'b': 200})
    retrieved = cache.retrieve()
    assert retrieved['a'] == 1 and retrieved['b'] == 200
    cache = util.Cache(src_file, retriever, cache_type='test')
    assert cache.cacheFile.split('.')[-2:] == ['test', 'json']
    cache = util.Cache(src_file, retriever, algo='mtime')
    assert cache._compare_hash()
    assert not cache._compare_hash()
    time.sleep(0.1)
    t = time.time()
    os.utime(src_file, (t, t))
    assert cache._compare_hash()
    util.safe_remove(_gen_dir)
    assert cache._compute_hash_as_modified_time() is None


def test_mem_caching():
    @util.mem_caching(maxsize=None)
    def load(src):
        time.sleep(1)
        return util.load_json(src)

    src1 = src_file = osp.join(_gen_dir, 'data1.json')
    src2 = src_file = osp.join(_gen_dir, 'data2.json')
    util.save_json(src1, {'a': 1, 'b': 2})
    util.save_json(src2, {'a': 100, 'b': 200})
    assert load(src1) == {'a': 1, 'b': 2}
    assert load(src2) == {'a': 100, 'b': 200}
    assert load(src1) == {'a': 1, 'b': 2}
    assert load(src2) == {'a': 100, 'b': 200}
    util.safe_remove(_gen_dir)


def test_find_invalid_path_chars():
    invalid = util.find_invalid_path_chars('hello \\*wor#ld@')
    if util.PLATFORM != 'Windows':
        assert invalid == {'hello \\*wor#ld@': [(6, '\\'), (7, '*')]}
    else:
        assert invalid == {'*wor#ld@': [(0, '*')]}


def test_extract_path_stem():
    path = r'c:\path\to\file.ext' if util.PLATFORM == 'Windows' else r'/path/to/file.ext'
    assert util.extract_path_stem(path) == 'file'


def test_extract_docstring():
    src_file = osp.join(_org_dir, 'with_docstring.py')
    assert util.extract_docstring(src_file) == ("""\
source-level docstring
- line 1
- line 2
- line 3""", 1, 4)
    src_file = osp.join(_org_dir, 'without_docstring.py')
    assert util.extract_docstring(src_file) == (None, None, None)
    src_file = osp.join(_org_dir, 'without_docstring2.py')
    assert util.extract_docstring(src_file) == (None, None, None)
    src_file = osp.join(_org_dir, 'without_docstring3.py')
    assert util.extract_docstring(src_file) == (None, None, None)
    src_file = osp.join(_org_dir, 'without_docstring4.py')
    assert util.extract_docstring(src_file) == (None, None, None)
    src_file = osp.join(_org_dir, 'with_docstring.lua')
    assert util.extract_docstring(src_file, envelope='---') == ("""\
--line 1
--line 2""", 1, 2)
    with pytest.raises(SyntaxError):
        util.extract_docstring(src_file)
    assert util.extract_docstring(src_file, envelope='"""') == (None, None, None)


def test_inject_docstring():
    src_file = osp.join(_org_dir, 'without_docstring.py')
    dst_file = osp.join(_gen_dir, 'without_docstring.py')
    util.copy_file(src_file, dst_file)
    assert util.inject_docstring(dst_file, 'hello doc') == util.append_lineends_to_lines("""\
\"\"\"
hello doc
\"\"\"
# just comments without docstring

if __name__ == '__main__':
    \"\"\"
    at the module source level
    \"\"\"
    pass
""".splitlines())
    util.safe_remove(_gen_dir)


def test_save_load_csv():
    dsv = osp.join(_org_dir, 'dsv_comma.txt')
    assert util.load_dsv(dsv) == [
        ['header1', 'header2', 'header3'],
        ['co11', 'co12', 'co13'],
        ['co21', 'co22', 'co23'],
    ]
    dsv = osp.join(_org_dir, 'dsv_whitespace.txt')
    assert util.load_dsv(dsv, delimiter=' ') == [
        ['header1', 'header with space2', 'header3'],
        ['co11', 'co12', 'co13 with space'],
        ['co21', 'co22', 'co23'],
    ]
    util.safe_remove(_gen_dir)
    dst = osp.join(_gen_dir, )
    rows = [
        ['header1', 'header2', 'header3'],
        ['co11', 'co12', 'co13'],
        ['co21', 'co22', 'co23'],
    ]
    util.save_dsv(dst, rows)
    assert util.load_dsv(dst) == rows
    util.safe_remove(_gen_dir)


def test_say():
    os.makedirs(_gen_dir, exist_ok=True)
    out = osp.join(_gen_dir, 'hello.wav')
    assert osp.isfile(util.say('hello', outfile=out))
    util.safe_remove(_gen_dir)


def test_http_get(monkeypatch):
    # Define the URL and the mock response content
    url = "http://example.com"
    mock_content = "Hello, World!"

    # Create a mock response object with a read method
    class MockResponse:
        def read(self):
            return mock_content.encode()

        # Add __enter__ and __exit__ methods to allow use as a context manager
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    # Use monkeypatch to replace urllib.request.urlopen with a lambda function
    # that returns an instance of our MockResponse
    monkeypatch.setattr('urllib.request.urlopen', lambda _: MockResponse())

    # Call the http_get function and check the returned value
    result = util.http_get(url)
    assert result == mock_content


def test_http_post(monkeypatch):
    # Define the URL, the input data, and the mock response content
    in_url = "http://example.com"
    input_data = {'key': 'value'}
    mock_content = "Response content"
    mock_status = 200

    # Create a mock response object with read method, status, and url attributes
    class MockResponse:
        def read(self):
            return mock_content.encode()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        status = mock_status
        url = in_url  # Changed variable name to avoid shadowing

    # Use monkeypatch to replace urllib.request.urlopen
    def mock_urlopen(request, data):
        # Check that the headers are set correctly
        assert request.headers['Content-type'] == 'application/json'
        # Check that the data is correctly encoded
        assert json.loads(data.decode()) == input_data
        # Return the mock response object
        return MockResponse()

    monkeypatch.setattr('urllib.request.urlopen', mock_urlopen)

    # Call the http_post function and check the returned value
    resp = util.http_post(in_url, input_data)
    assert resp.status_code == mock_status
    assert resp.url == in_url  # The expected URL is the one defined at the beginning
    assert resp.content.decode() == mock_content


def test_lazy_download():
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/320px-Tsunami_by_hokusai_19th_century.jpg'
    file = osp.join(_gen_dir, 'tsunami.jpg')
    assert osp.isfile(util.lazy_download(file, url))
    util.safe_remove(_gen_dir)


def test_get_environment():
    assert {'os', 'pyExe', 'pyVersion', 'pyPath', 'osPath'} == set(util.get_environment().keys())


def test_safe_decode_bytes_encode_text():
    text = 'hello'
    encoded = util.safe_encode_text(text, encoding=util.TXT_CODEC)
    assert util.safe_decode_bytes(encoded, encoding=util.TXT_CODEC) == text
    text = '你好'
    encoded = util.safe_encode_text(text, encoding='gbk')
    assert util.safe_decode_bytes(encoded, encoding='gbk') == text


def test_report_duration():
    assert util.report_duration(
        start=datetime.datetime(2023, 10, 10, 0, 0, 0, 0),
        end=datetime.datetime(2023, 10, 13, 1, 10, 10, 10*1000),
    ) == f"3.1:10:10.10"


def test_safe_get_element():
    coll = [1, 2, 3, 4, 5]
    assert util.safe_get_element(coll, 0) == 1
    assert util.safe_get_element(coll, 100) is None
    assert util.safe_get_element(coll, 100, 'hello') == 'hello'


def test_safe_index():
    coll = [1, 2, 3, 4, 5]
    assert util.safe_index(coll, 1) == 0
    assert util.safe_index(coll, 100) is None
    assert util.safe_index(coll, 100, 'hello') == 'hello'


def test_load_ini():
    config = util.load_ini(osp.join(_org_dir, 'simple.ini'))
    assert config.get('bug_tracker', 'url') == 'http://localhost:8080/bugs/'


def test_indent():
    assert util.indent('hello', 2) == '  hello'
    assert util.indent(['hello', 'world']) == ['    hello', '    world']


def test_is_number():
    test_strings = ["123", "-123", "6.", "123.45", "abc", "12.12.12", "-", "-6.", "-6.5", ""]
    expected = [True, True, True, True, False, False, False, True, True, False]
    for s, ts in enumerate(test_strings):
        assert util.is_number_text(ts) == expected[s]


def test_find_log_path():
    assert util.find_log_path(util.glogger) == osp.abspath(f'{util.get_platform_tmp_dir()}/_util/util.log')


def test_collect_file_tree():
    root = osp.join(_org_dir, 'duplicate_this')
    assert [osp.relpath(path, root) for path in util.collect_file_tree(root)] == [osp.normpath(path) for path in [
        'sub/my.file',
        'my.file'
    ]]


def test_merge_namespaces():
    to_ns = types.SimpleNamespace(
        mine=100,
        overwritten=[1, 2, 3],
    )
    from_ns = types.SimpleNamespace(
        theirs='hello world',
        overwritten=[100, 200, 300],
    )
    assert util.merge_namespaces(to_ns, from_ns, True) == types.SimpleNamespace(
        mine=100,
        overwritten=[100, 200, 300],
    )
    assert util.merge_namespaces(to_ns, from_ns) == types.SimpleNamespace(
        mine=100,
        theirs='hello world',
        overwritten=[100, 200, 300],
    )


def test_is_pid_running():
    assert util.is_pid_running(os.getpid())
    assert not util.is_pid_running(99999)


@util.thread_timeout(1)
def _do_it_until_thread_timeout(sec):
    time.sleep(sec)


@util.process_timeout(1)
def _do_it_until_proc_timeout():
    cmd = ['poetry', 'run', 'python', osp.join(_org_dir, 'timeout_proc.py')]
    util.run_cmd(cmd, cwd=_org_dir)


@util.thread_timeout(1, True)
def _do_it_until_thread_timeout_bypassed(sec):
    time.sleep(sec)


@util.process_timeout(1, True)
def _do_it_until_proc_timeout_bypassed():
    cmd = ['poetry', 'run', 'python', osp.join(_org_dir, 'timeout_proc.py')]
    util.run_cmd(cmd, cwd=_org_dir)


def test_timeout():
    with pytest.raises(TimeoutError):
        _do_it_until_thread_timeout(2)
    with pytest.raises(TimeoutError):
        _do_it_until_proc_timeout()
    _do_it_until_thread_timeout_bypassed(2)
    _do_it_until_proc_timeout_bypassed()


def test_remove_unsupported_dict_keys():
    my_dict = {'a': 1, 'b': 2, 'c': 3}
    assert util.remove_unsupported_dict_keys(my_dict, {'c'}) == {'c': 3}


def test_format_xml():
    import xml.etree.ElementTree as ET

    # Create an XML element
    root = ET.Element('root')
    child1 = ET.SubElement(root, 'child')
    child2 = ET.SubElement(root, 'child')
    child1.text = 'Hello'
    child2.text = 'World'

    # Expected formatted XML string
    expected_output = '''<?xml version="1.0" encoding="utf-8"?>
<root>
    <child>Hello</child>
    <child>World</child>
</root>'''

    # Call the function
    formatted_xml = util.format_xml(root)

    # Assert the output is as expected
    assert formatted_xml == expected_output, f"Expected:\n{expected_output}\nBut got:\n{formatted_xml}"


def test_create_parameter():
    param = util.create_parameter('integer', default='10')
    assert param['name'] == 'integer'
    assert param['default'] == 10
    assert param['type'] == 'int'
    assert param['range'] == [-4294967295, 4294967295]
    assert param['step'] == 1
    param = util.create_parameter('integer', default='10', val_range=[10, None])
    assert param['name'] == 'integer'
    assert param['default'] == 10
    assert param['type'] == 'int'
    assert param['range'] == [10, 4294967295]
    assert param['step'] == 1
    param = util.create_parameter('integer', default='10', val_range=[0, 10], step=10)
    assert param['name'] == 'integer'
    assert param['default'] == 10
    assert param['type'] == 'int'
    assert param['range'] == [0, 10]
    assert param['step'] == 10

    param = util.create_parameter('float', default='1.0', val_range=[10, None])
    assert param['name'] == 'float'
    assert math.isclose(param['default'], 1.0)
    assert param['type'] == 'float'
    assert param['range'] == [10, float('inf')]
    assert math.isclose(param['step'], 0.1)
    assert param['precision'] == 2
    param = util.create_parameter('float', default='1.0', step=0.01, precision=3)
    assert param['name'] == 'float'
    assert math.isclose(param['default'], 1.0)
    assert param['type'] == 'float'
    assert param['range'] == [float('-inf'), float('inf')]
    assert math.isclose(param['step'], 0.01)
    assert param['precision'] == 3

    param = util.create_parameter('single-selection', default='opt1', val_range=('opt1', 'opt2', 'opt3'))
    assert param['name'] == 'single-selection'
    assert param['default'] == 'opt1'
    assert param['type'] == 'option'
    assert param['range'] == ('opt1', 'opt2', 'opt3')
    param = util.create_parameter('multi-selection', default='opt1 opt2', val_range=('opt1', 'opt2', 'opt3'))
    assert param['name'] == 'multi-selection'
    assert param['default'] == ('opt1', 'opt2')
    assert param['type'] == 'option'
    assert param['range'] == ('opt1', 'opt2', 'opt3')
    param = util.create_parameter('multi-selection', default='opt1, opt2', val_range=('opt1', 'opt2', 'opt3'), delim=',')
    assert param['name'] == 'multi-selection'
    assert param['default'] == ('opt1', 'opt2')
    assert param['type'] == 'option'
    assert param['range'] == ('opt1', 'opt2', 'opt3')

    param = util.create_parameter('text', default='Hello')
    assert param['name'] == 'text'
    assert param['default'] == 'Hello'
    assert param['type'] == 'str'

    # Create a parameter with no default value
    param = util.create_parameter('boolean', default='false')
    assert param['name'] == 'boolean'
    assert not param['default']
    assert param['type'] == 'bool'


def test_json_to_text():
    data = {'a': 1, 'b': "中文", 'c': 3}
    assert util.json_to_text(data) == ("""{"a": 1, "b": "中文", "c": 3}""", None)
    assert util.json_to_text(data, pretty=True) == ("""{
    "a": 1,
    "b": "中文",
    "c": 3
}""", None)
    data = {
        "name": "Leóna",
        "age": 3,
        "favorite_food": "🍕",
        "languages": ["English", "中文", "Français"],
        "unsupported_type": {1, 2, 3}  # Sets are not JSON serializable
    }
    # Try to serialize the dict with a set
    text, exc = util.json_to_text(data)
    assert text is None
    assert isinstance(exc, TypeError)


def test_json_from_text():
    text = """{"a": 1, "b": "中文", "c": 3}"""
    assert util.json_from_text(text) == ({'a': 1, 'b': '中文', 'c': 3}, None)
    # test exception
    text = """{"a": 1, "b": "中文", "c": 3"""
    obj, exc = util.json_from_text(text)
    assert obj is None
    assert isinstance(exc, json.decoder.JSONDecodeError)


def test_offline_json():
    oj = util.OfflineJSON(osp.join(_gen_dir, 'test.json'))
    oj.save({'a': 1, 'b': 2})
    assert oj.load() == {'a': 1, 'b': 2}
    data = oj.merge({'a': 1, 'b': 200, 'c': '中文'})
    assert data == {'a': 1, 'b': 200, 'c': '中文'}
    util.safe_remove(_gen_dir)


def test_borg_singleton():
    class BorgNoShareWithParent(util.BorgSingleton):
        _shared_borg_state = {}

        def __init__(self):
            super().__init__()
            self.a = 1

    b1 = BorgNoShareWithParent()
    b2 = BorgNoShareWithParent()
    assert b1 is not b2
    assert b1.a == 1
    assert b2.a == 1
    b1.a = 100
    assert b2.a == 100
    b2.a = 200
    assert b1.a == 200
    b0 = util.BorgSingleton()
    b0.a = 999
    assert b0 is not b1
    assert b0.a == 999
    assert b1.a == 200


def test_classic_singleton():
    class Singleton(util.ClassicSingleton):
        def __init__(self):
            self.a = 1

    class NonSingleton:
        def __init__(self):
            self.a = 1

    class NonSingletonChild(NonSingleton):
        def __init__(self):
            super().__init__()
            self.b = False

    b1 = Singleton.instance()
    assert b1.a == 1
    b1.a = 100
    b2 = Singleton.instance()
    assert b1 is b2
    assert b2.a == 100
    b2.a = 200
    assert b1.a == 200
    b0 = util.ClassicSingleton.instance()
    b0.a = 999
    assert b0 is not b1
    assert b0.a == 999, 'parent gets injected with new child attribute'
    assert b1.a == 200, 'child attribute should not affect or be affected by parent'
    c1 = NonSingleton()
    c2 = NonSingleton()
    assert c1 is not c2
    c2.a = 100
    assert c1.a != c2.a
    c3 = NonSingletonChild()
    c4 = NonSingletonChild()
    assert c3 is not c4
    c4.a = 999
    assert c3.a != c4.a
    assert c4.a != c2.a
    assert not c4.b
    with pytest.raises(AttributeError):
        print(c1.b)
    c1.b = True
    print(c1.b)


def test_meta_singleton():
    class Singleton(metaclass=util.MetaSingleton):
        def __init__(self):
            self.a = 1

    class Child(Singleton):
        def __init__(self):
            super().__init__()
            self.b = 2

    b1 = Singleton()
    assert b1.a == 1
    b1.a = 100
    b2 = Singleton()
    assert b1 is b2
    assert b2.a == 100
    b2.a = 200
    assert b1.a == 200
    assert b1.a == 200
    c1 = Child()
    c1.a = 2
    assert b1.a == 200
    c2 = Child()
    assert c1 is c2
    assert c2.a == 2
    assert c1 is not b1


def test_exceptable_thread():
    def _do_it():
        time.sleep(1)
        raise ValueError('test')

    et = util.ExceptableThread(target=_do_it)
    et.start()
    with pytest.raises(ValueError):
        et.join()


def test_format_callstack():
    callstack = util.format_callstack()
    assert 'test_default.py' in callstack
    assert 'inspect.stack()' in callstack
