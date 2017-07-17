# -*- mode: python -*-

block_cipher = None


#options = [('v', None, 'OPTION')]
a = Analysis(['Lib\\site-packages\\neuralcli\\__main__.py'],
             pathex=['Lib\\site-packages', 'C:\\Users\\wincent\\Documents\\Projects\\neuralcli'],
             binaries=None,
             datas=None,
             hiddenimports=['neuralcli', 'neuralcli.cli', 'neuralcli.neuralnet', 'tkinter.filedialog'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
#          options,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='neuralcli',
          debug=False,
          strip=False,
          upx=True,
          console=True,
          icon='neuralcli.ico')
