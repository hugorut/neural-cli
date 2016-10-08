from setuptools import setup

setup(
  name = 'neuralcli',
  packages = ['neuralcli'],
  entry_points = {
      "console_scripts": ['neuralcli = neuralcli.cli:main']
      },
  version = '1.0',
  description = 'A command line neural network',
  author = 'Hugo Rut',
  author_email = 'hugorut@gmail.com',
  url = 'https://github.com/hugorut/neural-cli',
  download_url = 'https://github.com/hugorut/neural-cli/tarball/1.0', 
  keywords = ['machine learning', 'ai', 'neural nework'], 
  classifiers = [],
)
