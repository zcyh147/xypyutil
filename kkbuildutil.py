import copy
import glob
import os
import os.path as osp
import pprint as pp
import shutil
import types

import kkpyutil as util


class Autotools:
    def __init__(self, pkgroot, logger=None):
        self.pkgRoot = pkgroot
        self.logger = logger
        self.make = shutil.which('make')

    def init(self):
        util.install_by_homebrew('autoconf', lazybin='/usr/local/bin/autoreconf')
        util.install_by_homebrew('automake', lazybin='/usr/local/bin/automake')

    def autogen(self):
        util.run_cmd(['./autogen.sh'], cwd=self.pkgRoot, logger=self.logger)

    def configure(self, opts=(), flags=()):
        """
        Must run under shell mode
        - CFLAGS="...", CXXFLAGS... are not options but environment variables
        - So must spawn subprocess under shell mode: Treat the ./configure sequence as a whole command
        """
        compiler_flags = ['env'] + list(flags) if flags else []
        util.run_cmd([' '.join(compiler_flags + ['./configure'] + list(opts))], shell=True, cwd=self.pkgRoot, logger=self.logger)

    def make_install(self, njobs=8):
        util.run_cmd([self.make, f'-j{njobs}'], cwd=self.pkgRoot, logger=self.logger)
        util.run_cmd([self.make, 'install'], cwd=self.pkgRoot, logger=self.logger)

    def make_clean(self):
        util.run_cmd([self.make, 'clean'], cwd=self.pkgRoot, logger=self.logger)


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
        util.run_cmd(cmd, cwd=self.buildDir)

    def build(self, gnumake=True):
        job_flag = ['-j8'] if gnumake else []
        cmd = [self.cmake, '--build', '.', '--target', 'install'] + job_flag
        util.run_cmd(cmd, cwd=self.buildDir)

    def clean(self):
        cmd = [self.cmake, '--build', '.', '--target', 'clean']
        util.run_cmd(cmd, cwd=self.buildDir)


def build_ico(master, ico, size='256x256'):
    cmd = ['magick', master, '-background', 'none', '-resize', size, '-density', size, ico]
    util.run_cmd(cmd)


def build_iconset(master, iconset):
    util.validate_platform('Darwin')
    iconset_dir = osp.join(f'/{util.get_platform_tmp_dir()}/icons/icon.iconset')
    os.makedirs(iconset_dir, exist_ok=True)
    sizes = ('16', '16@2x', '32', '32@2x', '128', '128@2x', '256', '256@2x', '512')
    for sz in sizes:
        comps = sz.split('@')
        sz = comps[0]
        actual_size = str(2*int(sz)) if (has_sep := len(comps) > 1) else sz
        suffix = f'{sz}x{sz}@2x' if has_sep else f'{sz}x{sz}'
        util.run_cmd(['sips', '-z', actual_size, actual_size, master, '--out', osp.abspath(f'{iconset_dir}/icon_{suffix}.png')])
    util.copy_file(master, osp.abspath(f'{iconset_dir}/icon_512x512@2x.png'))
    util.run_cmd(['iconutil', '-c', 'icns', iconset_dir, '-o', iconset])
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

        def __init__(self, buildproduct, prefix='@executable_path', parent=None, logger=util.glogger):
            assert osp.isabs(buildproduct), f'Not an absolute path: {buildproduct}, parent: {parent}'
            self.buildProduct = buildproduct
            # CAUTION
            # - prefix is to be embedded into binaries
            # - input prefix must be relative to @executable_path, e.g., @executable_path/libs
            # - input may already start with a build-time base
            runtime_base_tag = '@executable_path'
            # prefix must start with '@executable_path'
            self.prefix = prefix.rstrip(os.sep) if prefix.startswith(runtime_base_tag) else osp.join(runtime_base_tag, prefix.strip(os.sep))
            # for reference only, we trace deps tree top-down from the outside
            self.ref = parent
            self.logger = logger
            if is_root := not self.ref:
                # root binary is our target, no need to decorate
                self.distributable = buildproduct
                DynLinked.execPath = osp.dirname(self.distributable)
                self.logger.info(f'Root aka. target binary: {self.distributable}')
            else:
                assert osp.isdir(DynLinked.execPath), 'Root binary has not been processed'
                # this path may not exist yet
                # @executable_path/relative/to/my.dylib
                dest_dir = osp.join(DynLinked.execPath, relpath_to_exepath := osp.relpath(prefix, runtime_base_tag).strip(os.sep))
                self.distributable = osp.join(dest_dir, bin_filename := osp.split(self.buildProduct)[1])
            self.distributable = None
            self.deps = None

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
                util.copy_file(self.buildProduct, self.distributable, isdstdir=False)
            proc = util.run_cmd([DynLinked.analyzer, '-L', self.distributable])
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
                self.logger.info('Deps exist. Skipped to avoid pollution from already-fixed @executable_path prefix')
            self.logger.info(f"""Dependencies found for {self.buildProduct}:
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
                util.run_cmd([DynLinked.fixer, '-change', dep, distributable, dep_ref := self.distributable])

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
            util.run_cmd([DynLinked.fixer, '-id', distributable, self.distributable])

        def report(self):
            if sys_lib := not osp.isfile(self.distributable):
                return
            proc = util.run_cmd([DynLinked.analyzer, '-L', self.distributable])
            self.logger.info(f'fixed for {self.distributable}:\n{proc.stdout.decode(util.TXT_CODEC)}')

        def _extract_deps(self, otoolout):
            return otoolout.decode(util.TXT_CODEC).split('\n\t')

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
                util.glogger.debug(f'Already fixed by other exec: {child}, ref: {parent.buildProduct}; Skipped')
                continue
            util.glogger.debug(f'Found child: {child} of {parent.buildProduct}')
            cbin = DynLinked(child, prefix=prefix, parent=parent)
            _collect_deps_tree(cbin, io_allbins, prefix)
        return True
    util.validate_platform('Darwin')
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
        util.glogger.info('dependency recursively parsed')
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
    util.validate_platform('Darwin')
    util.run_cmd(['security', 'find-identity', '-v', '-p', 'codesigning'])
    if overwrite:
        util.run_cmd(['codesign', '--remove-signature', binary])
    # -v with feedback
    util.run_cmd(['codesign', '-s', identity, '-v', binary])
    # gatekeeper validation
    util.run_cmd(['spctl', '-a', '-t', 'exec', '-vv', binary])


def build_dmg(masterdir, resdir='', dmg='', name=''):
    """
    build a DMG installer based on fixed layout, and lazy-include resources
    - masterdir: content folder holding the app and optionally sub-folder "help"
    - resdir: asset folder for building dmg only, not part of app content, usually holds background image (bg.png, made by Keynote 4:3, size 1024x768) and volume iconset (app.icns)
    - dmg: path to output dmg
    - name: app name, default to use basename of masterdir
    """
    util.validate_platform('Darwin')
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
        util.glogger.warning('Found background image; size must be 1024x768 (Keynote 4:3)')
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
    util.run_cmd(cmd)


def build_xcodeproj(proj, scheme, config='Debug', sdk='macosx'):
    cmd = ['xcodebuild', '-project', proj, '-scheme', scheme, '-sdk', sdk, '-configuration', config, 'build']
    util.run_cmd(cmd)


def clean_xcodeproj(proj, scheme, config='Debug', sdk='macosx'):
    cmd = ['xcodebuild', '-project', proj, '-scheme', scheme, '-sdk', sdk, '-configuration', config, 'clean']
    util.run_cmd(cmd)
