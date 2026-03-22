from setuptools import setup, find_namespace_packages

setup(
    name='FilterGS',
    version='0.0.0',
    description='FilterGS',
    author='Yixian Wang',
    author_email='wangyixe@163.com',
    packages=find_namespace_packages(include=['LoG', 'LoG.*', 'FilterGS', 'FilterGS.*']),
    entry_points={
        'console_scripts': [],
    },
    install_requires=[],
    data_files=[],
)
