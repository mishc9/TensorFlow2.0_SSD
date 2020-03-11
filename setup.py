from distutils.core import setup
import setuptools

try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = None


setup(
    name='tf2ssd',
    version='0.0.0',
    author='https://github.com/calmisential',
    packages=setuptools.find_packages(),
    description='ssd implementation in TF2.0',
    include_package_data=True,
    python_requires='>=3.6.5',
    install_requires=requirements,
)
