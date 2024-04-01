from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    install_requires = fp.read()

setup(
    name='emi1d',
    version='0.0.1',
    author='Juan Chavez Olalla',
    author_email='jchavezolalla@gmail.com',
    packages=find_packages(),
    install_requires=[
        install_requires,
    ],
)

