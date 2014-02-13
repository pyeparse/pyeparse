# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os
import pyeparse

import setuptools  # analysis:ignore noqa we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """PyLinkParse project for eye tracking data analysis."""

DISTNAME = 'pyeparse'
DESCRIPTION = descr
MAINTAINER = 'Denis A. Engemann'
MAINTAINER_EMAIL = 'd.engemann@fz-juelich.de'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/dengemann/PyLinkParse.git'
VERSION = pyeparse.__version__


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['pyeparse', 'pyeparse.tests'])
