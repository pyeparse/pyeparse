# -*- coding: utf-8 -*-
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

# Iterator
try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()
next = advance_iterator

# Basestring
try:
    string_types = basestring  # noqa
except NameError:
    string_types = str

# StringIO
try:
    from cStringIO import StringIO
except ImportError:  # py3 has renamed this
    from io import StringIO  # noqa
from io import BytesIO  # noqa
