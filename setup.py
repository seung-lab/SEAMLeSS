from setuptools import setup, find_packages

setup(
    name='seamless',
    version='0.2',
    description='SEAMLeSS alignment',
    packages=find_packages(),
    scripts=[
        'inference/client.py',
        'training/train.py',
    ],
    url="https://github.com/seung-lab/SEAMLeSS",
    setup_requires=[
        'pbr',
    ],
    pbr=True,
)
