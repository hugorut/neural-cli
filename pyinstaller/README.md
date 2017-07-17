# How to create single executable of the `neuralcli` package for Windows&reg;

 1. Install package `pyinstaller`, preferably into a virtual environment.
 2. If did not install `neuralcli` package yet, do it now.
 3. Copy the `neuralcli.spec` and `neuralcli.ico` files into the top directory of the virtual environment.
 4. Execute
```
pyinstaller neuralcli.spec
```
 5. Check for `neuralcli.exe` in the current directory. If it does not exist, look at the errors.
