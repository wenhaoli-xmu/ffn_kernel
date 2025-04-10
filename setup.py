from setuptools import setup

setup(
    name='ffn_kernel',
    version='1.0',
    packages=['ffn_kernel'],
    install_requires=[
        'triton==3.2.0'
    ]
)