# -*- coding: utf-8 -*-
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()
next = advance_iterator


try:
    string_types = basestring  # noqa
except NameError:
    string_types = str
