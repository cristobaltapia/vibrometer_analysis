# -*- mode: python ; coding: utf-8 -*-
import os

spec_root = os.path.abspath(SPECPATH)

block_cipher = None

a = Analysis(
    ['vibrometer_analysis/qt_app.py'],
    pathex=[spec_root],
    binaries=[],
    datas=[],
    hiddenimports=[
        'pkg_resources.py2_warn',
        'scipy.special.cython_special',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vibrometer_analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='resources/icons/icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vibrometer_analysis',
)
