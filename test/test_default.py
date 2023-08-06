"""
tests that don't need external data
"""
import getpass
import platform
import shutil
import sys
import os
import os.path as osp
import subprocess
from unittest.mock import patch


# 3rd party
import types
import uuid
import pytest
# project
_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, repo_root := osp.abspath(f'{_script_dir}/..'))
import kkpyutil as util


_case_dir = osp.dirname(__file__)
_org_dir = osp.join(_case_dir, '_org')
_gen_dir = osp.join(_case_dir, '_gen')
_ref_dir = osp.join(_case_dir, '_ref')


def test_childpromptproxy():
    log = 'hello world'
    parent = subprocess.Popen(['python3', '-c', f'print("{log}")'], stdout=subprocess.PIPE)
    proxy = util.ChildPromptProxy(parent.stdout)
    proxy.start()
    parent.wait()
    assert proxy.log.decode() == f'{log}\n'


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
    plat = platform.system()
    if plat == 'Windows':
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}')
        assert util.get_platform_home_dir() == expected
    elif plat == 'Darwin':
        expected = osp.abspath(f'/Users/{getpass.getuser()}')
        assert util.get_platform_home_dir() == expected
    elif plat == 'Linux':
        expected = osp.abspath(f'/home/{getpass.getuser()}')
        assert util.get_platform_home_dir() == expected
    else:
        with pytest.raises(NotImplementedError):
            util.get_platform_home_dir()


def test_get_platform_appdata_dir():
    plat = platform.system()
    if plat == 'Windows':
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}\\AppData\\Roaming')
        assert util.get_platform_appdata_dir() == expected
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}\\AppData\\Local')
        assert util.get_platform_appdata_dir(winroam=False) == expected
    elif plat == 'Darwin':
        expected = osp.abspath(f'/Users/{getpass.getuser()}/Library/Application Support')
        assert util.get_platform_appdata_dir() == expected
    else:
        with pytest.raises(NotImplementedError):
            util.get_platform_appdata_dir()


def test_get_platform_tmp_dir():
    plat = platform.system()
    if plat == 'Windows':
        expected = osp.abspath(f'C:\\Users\\{getpass.getuser()}\\AppData\\Local\\Temp')
        assert util.get_platform_tmp_dir() == expected
    elif plat == 'Darwin':
        expected = osp.abspath(f'/Users/{getpass.getuser()}/Library/Caches')
        assert util.get_platform_tmp_dir() == expected
    elif plat == 'Linux':
        expected = '/tmp'
        assert util.get_platform_tmp_dir() == expected
    else:
        with pytest.raises(NotImplementedError):
            util.get_platform_tmp_dir()


def test_build_default_logger():
    log_dir = osp.dirname(__file__)
    logger = util.build_default_logger(log_dir, name='my')
    logger.debug('hello logger')
    log_file = osp.join(log_dir, 'my.log')
    assert osp.isfile(log_file)
    os.remove(log_file)


def test_catch_unknown_exception():
    util.catch_unknown_exception(RuntimeError, 'exception info', None)


def test_build_logger():
    src_file = __file__
    logger = util.build_logger(src_file)
    logger.debug('hello source logger')
    log_file = osp.join(osp.dirname(src_file), 'test_default.log')
    assert osp.isfile(log_file)
    os.remove(log_file)


def test_format_error_message():
    got = util.format_error_message(
        situation='task result is wrong',
        expected=100,
        got=-100,
        advice='did you forget to take its absolute value?',
        reaction='aborted',
    )
    expected = """\
task result is wrong:
- Expected: 100
- Got: -100
- Advice: did you forget to take its absolute value?
- Reaction: aborted"""
    assert got == expected


def test_is_multiline_text():
    text = 'single line'
    assert not util.is_multiline_text(text)
    text = """line 1
line 2
line 3"""
    assert util.is_multiline_text(text)


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
    src = osp.join(_org_dir, 'trace_this.py')
    cmd = ['poetry', 'run', 'python', src]
    proc = util.run_cmd(cmd, cwd=osp.dirname(__file__))
    assert proc.stdout.decode(util.TXT_CODEC) == """\
Call: __main__.hello(n=100, s='world', f='0.99') - def hello(n, s, f):
Call: __main__.hello => hello, 100, world, 0.99 - return x
"""


def test_get_md5_checksum():
    missing_file = 'missing'
    assert util.get_md5_checksum(missing_file) is None
    valid_file = osp.abspath(f'{_script_dir}/../LICENSE')
    # line-ends count
    assert util.get_md5_checksum(valid_file) == '5d326be91ee12591b87f17b6f4000efe' if platform.system() == 'Windows' else '7a3beb0af03d4afff89f8a69c70a87c0'


def test_logcall():
    src = osp.join(_org_dir, 'log_this.py')
    cmd = ['poetry', 'run', 'python', src]
    proc = util.run_cmd(cmd, cwd=osp.dirname(__file__))
    stdout_lines = set(proc.stdout.decode(util.TXT_CODEC).splitlines())
    key_lines = {
        "Enter: 'myfunc' <= args=(100, 'hello'), kwargs={'f': 0.99}: trace",
        "Exit: 'myfunc' => hello, 100, hello, 0.99",
    }
    assert key_lines.issubset(stdout_lines)


def map_worker(enum):
    e, elem = enum[0], enum[1]
    return elem*2


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
        name='util',
        verbose=True)
    res = util.concur_map(inner_func, data, worker_count=5, iobound=True, logger=logger)
    assert res == [i*2 for i in range(n)]
    res = util.concur_map(map_worker, data, worker_count=5, iobound=False, logger=logger)
    assert res == [i * 2 for i in range(n)]
    shutil.rmtree(_gen_dir, ignore_errors=True)


def test_profile_runs():
    profile_mod = osp.join(_org_dir, 'profile_this.py')
    funcname = 'run_profile_target'
    stats = util.profile_runs(funcname, profile_mod, outdir=_gen_dir)
    assert stats.total_calls == 575
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


def test_alert_on_windows(monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Windows')
    monkeypatch.setattr('os.system', lambda cmd: None)
    assert util.alert('Title', 'Content', 'Finish') == ['mshta', 'vbscript:Execute("msgbox ""Content"", 0,""Title"":Finish")']


def test_alert_on_darwin(monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(subprocess, 'run', lambda cmd: None)
    assert util.alert('Title', 'Content') == ['osascript', '-e', 'display alert "Title" message "Content"']


def test_alert_on_other_platform(monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Other')
    monkeypatch.setattr(subprocess, 'run', lambda cmd: None)
    assert util.alert('Title', 'Content', 'Action') == ['echo', 'Title: Content: Action']


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
    assert util.convert_from_wine_path(path) == '~/my/file' if platform.system() == 'Windows' else f'{os.environ.get("HOME")}/my/file'
    path = 'X:\\my\\file'
    assert util.convert_from_wine_path(path) == path


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


def test_remove_duplication():
    my_list = [1, 2, 3, 2, 5, 3]
    assert (util.remove_duplication(my_list)) == [1, 2, 3, 5]
    my_list = [1, 5.0, 'xyz', 5.0, 5, 'xyz']
    assert (util.remove_duplication(my_list)) == [1, 5.0, 'xyz']


def test_validate_platform():
    supported = ['os1', 'os2']
    with pytest.raises(NotImplementedError):
        util.validate_platform(supported)
    supported = platform.system()
    util.validate_platform(supported)


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


def test_get_drivewise_commondirs():
    # single path
    if is_posix := platform.system() != 'Windows':
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
    if is_posix := platform.system() != 'Windows':
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
    if platform.system() == 'Windows':
        path = osp.normpath('C:/path/to/dir1/file1')
        assert util.split_platform_drive(path) == ('c:', osp.normpath('/path/to/dir1/file1'))
        path = osp.normpath('path/to/dir1/file1')
        assert util.split_platform_drive(path) == ('', osp.normpath('path/to/dir1/file1'))
    else:
        path = '/path/to/dir1/file1'
        assert util.split_platform_drive(path) == ('/', 'path/to/dir1/file1')
        path = 'path/to/dir1/file1'
        assert util.split_platform_drive(path) == ('', 'path/to/dir1/file1')


def test_sanitize_path_part():
    path_part = 'tab: 天哪*?"<\\/\x00\x1F'
    assert util.sanitize_text_as_path(path_part) == 'tab_ 天哪________'


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
    # another line containing different float and uuid
    line2 = 'length 1.23459 e3016d69-cb30-4eb1-9f93-bb28621aba28'
    assert not util.compare_dsv_lines(line1, line2)
    assert util.compare_dsv_lines(line1, line2, float_rel_tol=1e-5, float_abs_tol=1e-5, randomidok=True)

