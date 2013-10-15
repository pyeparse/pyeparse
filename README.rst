.. -*- mode: rst -*-
`PyLinkParse <https://github.com/dengemann/PyLinkParse>`_
=======================================================

This package is designed for interactive as well as scripting-oriented processing
of eye tracking data


Notes
^^^^^

Before you get started, a few things to point out:

    * This package has alpha status and is only tested for the author's use cases. Contributions are welcome but at the moment we can't provide strong support.

    * We tested the codes for different datasets and EyeLink devices. Still the parsing support will be limited to the cases we encountered. Reports and feedback are welcome.

    * The philosophy behind this package and the code itself strongly draw from MNE-Python, a package for MEG/EEG data processing hosted by the authors:
    https://github.com/mne-tools/mne-python


Get more information
^^^^^^^^^^^^^^^^^^^^

This page only contains bare-bones instructions for installing PyLinkParse.


Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using git, simply type::

    git clone git@github.com:dengemann/PyLinkParse.git

If you don't have git installed, you can download a zip or tarball
of the latest code:: 
    
    git@github.com:dengemann/PyLinkParse.git

Install PyLinkParse
^^^^^^^^^^^^^^^^^^^

As any Python packages, to install PyLink, go in the source
code directory and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user

Dependencies
^^^^^^^^^^^^

The required dependencies to build the software are Python >= 2.6,
NumPy >= 1.4, and Pandas >= 0.7.3


Licensing
^^^^^^^^^

PyLinkParse is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2012-2013, authors of PyLinkParse
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of PyLinkParse authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**