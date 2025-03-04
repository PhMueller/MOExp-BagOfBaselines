# -*- encoding: utf-8 -*-
import json
import os
import setuptools


def get_extra_requirements():
    """ Helper function to read in all extra requirement files in the extra
        requirement folder. """
    if not os.path.exists('./extra_requirements'):
        return {}

    extra_requirements = {}
    for file in os.listdir('./extra_requirements'):
        with open(f'./extra_requirements/{file}', encoding='utf-8') as fh:
            requirements = json.load(fh)
            extra_requirements.update(requirements)
    return extra_requirements


def read_file(file_name):
    with open(file_name, encoding='utf-8') as fh:
        text = fh.read()
    return text


setuptools.setup(
    name='MOExp-BagOfBaselines',
    author_email='philipp.mueller@stat.uni-muenchen.de',
    description='Adaptation of the bag of baselines repo',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    version='0.0.1',
    packages=setuptools.find_packages(exclude=['*.tests', '*.tests.*',
                                               'tests.*', 'tests'],),
    package_data={},
    include_package_data=True,
    python_requires='>=3.6, <=3.10',
    install_requires=read_file('requirements.txt').split('\n'),
    extras_require=get_extra_requirements(),
    test_suite='pytest',
    platforms=['Linux'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
