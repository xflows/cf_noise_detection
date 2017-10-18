from setuptools import setup, find_packages
import os

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
]

dist = setup(
    name='cf_noise_detection',
    version='0.1',
    author='Darko Aleksovski',
    description='A ClowdFlows package containing noise detection widgets. The package is intended to be used along with ClowdFlows 2.0.',
    url='https://github.com/xflows/cf_noise_detection',
    license='MIT License',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'cf_core',
        'cf_data_mining',
        'cf_weka',
        'Orange==2.5a4'
    ],
)
