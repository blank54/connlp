from setuptools import setup, find_packages

setup(
    name='constrnlp',
    version='0.0.5',
    license='GPL',
    author='Seonghyeon Boris Moon',
    author_email='boris.moon514@gmail.com',
    description='A bunch of python codes to analyze text data in the construction industry. Mainly reconstitute the pre-exist python libraries for Natural Language Processing (NLP)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/blank54/constrnlp_pypi.git',
    packages=find_packages(),
    classifiers=[],
)