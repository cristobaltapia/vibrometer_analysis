import os
import glob
from PyInstaller.utils.hooks import get_module_file_attribute
from PyInstaller.compat import is_win

binaries = []

if is_win:
    dll_glob = os.path.join(os.path.dirname(
        get_module_file_attribute('scipy')), '.libs', "*.dll")
    if glob.glob(dll_glob):
        binaries.append((dll_glob, "."))

# collect library-wide utility extension modules
hiddenimports = ['scipy._lib.%s' % m for m in [
    'messagestream', "_ccallback_c", "_fpumode"]]
