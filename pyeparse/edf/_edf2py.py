'''Wrapper for edf.h

Generated with:
ctypesgen.py --insert-file edf2py_extra.py --cpp=cl -EP -a -l edfapi -o
edf2py.py edf.h

Do not modify this file.
'''

__docformat__ = 'restructuredtext'

# Begin preamble

from ctypes import *
import ctypes, sys, os

_int_types = (c_int16, c_int32)
if hasattr(ctypes, 'c_int64'):
    # Some builds of ctypes apparently do not have c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (c_int64,)
for t in _int_types:
    if sizeof(t) == sizeof(c_size_t):
        c_ptrdiff_t = t
del t
del _int_types


class c_void(Structure):
    # c_void_p is a buggy return type, converting to int, so
    # POINTER(None) == c_void_p is actually written as
    # POINTER(c_void), so it can be treated as a real pointer.
    _fields_ = [('dummy', c_int)]


def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x

        p.from_param = classmethod(from_param)

    return p


class UserString:
    def __init__(self, seq):
        if isinstance(seq, basestring):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __int__(self):
        return int(self.data)

    def __long__(self):
        return long(self.data)

    def __float__(self):
        return float(self.data)

    def __complex__(self):
        return complex(self.data)

    def __hash__(self):
        return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)

    def __contains__(self, char):
        return char in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __getslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, basestring):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other))

    def __radd__(self, other):
        if isinstance(other, basestring):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other) + self.data)

    def __mul__(self, n):
        return self.__class__(self.data * n)

    __rmul__ = __mul__

    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.__class__(self.data.capitalize())

    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))

    def count(self, sub, start=0, end=sys.maxint):
        return self.data.count(sub, start, end)

    def decode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())

    def encode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())

    def endswith(self, suffix, start=0, end=sys.maxint):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))

    def find(self, sub, start=0, end=sys.maxint):
        return self.data.find(sub, start, end)

    def index(self, sub, start=0, end=sys.maxint):
        return self.data.index(sub, start, end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))

    def lower(self):
        return self.__class__(self.data.lower())

    def lstrip(self, chars=None):
        return self.__class__(self.data.lstrip(chars))

    def partition(self, sep):
        return self.data.partition(sep)

    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))

    def rfind(self, sub, start=0, end=sys.maxint):
        return self.data.rfind(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxint):
        return self.data.rindex(sub, start, end)

    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rstrip(self, chars=None):
        return self.__class__(self.data.rstrip(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def splitlines(self, keepends=0):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=sys.maxint):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.__class__(self.data.strip(chars))

    def swapcase(self):
        return self.__class__(self.data.swapcase())

    def title(self):
        return self.__class__(self.data.title())

    def translate(self, *args):
        return self.__class__(self.data.translate(*args))

    def upper(self):
        return self.__class__(self.data.upper())

    def zfill(self, width):
        return self.__class__(self.data.zfill(width))


class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""

    def __init__(self, string=""):
        self.data = string

    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")

    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + sub + self.data[index + 1:]

    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index + 1:]

    def __setslice__(self, start, end, sub):
        start = max(start, 0)
        end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start] + sub.data + self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start] + sub + self.data[end:]
        else:
            self.data = self.data[:start] + str(sub) + self.data[end:]

    def __delslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]

    def immutable(self):
        return UserString(self.data)

    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, basestring):
            self.data += other
        else:
            self.data += str(other)
        return self

    def __imul__(self, n):
        self.data *= n
        return self


class String(MutableString, Union):
    _fields_ = [('raw', POINTER(c_char)),
                ('data', c_char_p)]

    def __init__(self, obj=""):
        if isinstance(obj, (str, unicode, UserString)):
            self.data = str(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(POINTER(c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj)

        # Convert from c_char_p
        elif isinstance(obj, c_char_p):
            return obj

        # Convert from POINTER(c_char)
        elif isinstance(obj, POINTER(c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(cast(obj, POINTER(c_char)))

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)

    from_param = classmethod(from_param)


def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)


# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to c_void_p.
def UNCHECKED(type):
    if (hasattr(type, "_type_") and isinstance(type._type_, str)
        and type._type_ != "P"):
        return type
    else:
        return c_void_p


# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self, func, restype, argtypes):
        self.func = func
        self.func.restype = restype
        self.argtypes = argtypes

    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func

    def __call__(self, *args):
        fixed_args = []
        i = 0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i += 1
        return self.func(*fixed_args + list(args[i:]))

# End preamble

_libs = {}
_libdirs = []

# Begin loader

# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import re, sys, glob
import ctypes
import ctypes.util


def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []


class LibraryLoader(object):
    def __init__(self):
        self.other_dirs = []

    def load_library(self, libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self, path):
        """Given a path to a library, load it."""
        try:
            # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
            # of the default RTLD_LOCAL.  Without this, you end up with
            # libraries not being loadable, resulting in "Symbol not found"
            # errors
            if sys.platform == 'darwin':
                return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
            else:
                return ctypes.cdll.LoadLibrary(path)
        except OSError, e:
            raise ImportError(e)

    def getpaths(self, libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # FIXME / todo return '.' and os.path.dirname(__file__)
            for path in self.getplatformpaths(libname):
                yield path

            path = ctypes.util.find_library(libname)
            if path: yield path

    def getplatformpaths(self, libname):
        return []


# Darwin (Mac OS X)

class DarwinLibraryLoader(LibraryLoader):
    name_formats = ["lib%s.dylib", "lib%s.so", "lib%s.bundle", "%s.dylib",
                    "%s.so", "%s.bundle", "%s"]

    def getplatformpaths(self, libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir, name)

    def getdirs(self, libname):
        """Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        """

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [os.path.expanduser('~/lib'),
                                          '/usr/local/lib', '/usr/lib']

        dirs = []

        if '/' in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))

        dirs.extend(self.other_dirs)
        dirs.append(".")
        dirs.append(os.path.dirname(__file__))

        if hasattr(sys, 'frozen') and sys.frozen == 'macosx_app':
            dirs.append(os.path.join(
                os.environ['RESOURCEPATH'],
                '..',
                'Frameworks'))

        dirs.extend(dyld_fallback_library_path)

        return dirs


# Posix

class PosixLibraryLoader(LibraryLoader):
    _ld_so_cache = None

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = []
        for name in ("LD_LIBRARY_PATH",
                     "SHLIB_PATH",  # HPUX
                     "LIBPATH",  # OS/2, AIX
                     "LIBRARY_PATH",  # BE/OS
        ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")
        directories.append(os.path.dirname(__file__))

        try:
            directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError:
            pass

        directories.extend(['/lib', '/usr/lib', '/lib64', '/usr/lib64'])

        cache = {}
        lib_re = re.compile(r"lib(.*)\.s[ol]")
        ext_re = re.compile(r"\.s[ol]$")
        for dir in directories:
            try:
                for path in glob.glob("%s/*.s[ol]*" % dir):
                    file = os.path.basename(path)

                    # Index by filename
                    if file not in cache:
                        cache[file] = path

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname)
        if result: yield result

        path = ctypes.util.find_library(libname)
        if path: yield os.path.join("/lib", path)


# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try:
            return getattr(self.cdll, name)
        except AttributeError:
            try:
                return getattr(self.windll, name)
            except AttributeError:
                raise


class WindowsLibraryLoader(LibraryLoader):
    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll"]

    def load_library(self, libname):
        try:
            result = LibraryLoader.load_library(self, libname)
        except ImportError:
            result = None
            if os.path.sep not in libname:
                for name in self.name_formats:
                    try:
                        result = getattr(ctypes.cdll, name % libname)
                        if result:
                            break
                    except WindowsError:
                        result = None
            if result is None:
                try:
                    result = getattr(ctypes.cdll, libname)
                except WindowsError:
                    result = None
            if result is None:
                raise ImportError("%s not found." % libname)
        return result

    def load(self, path):
        return _WindowsLibrary(path)

    def getplatformpaths(self, libname):
        if os.path.sep not in libname:
            for name in self.name_formats:
                dll_in_current_dir = os.path.abspath(name % libname)
                if os.path.exists(dll_in_current_dir):
                    yield dll_in_current_dir
                path = ctypes.util.find_library(name % libname)
                if path:
                    yield path

# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin": DarwinLibraryLoader,
    "cygwin": WindowsLibraryLoader,
    "win32": WindowsLibraryLoader
}

loader = loaderclass.get(sys.platform, PosixLibraryLoader)()


def add_library_search_dirs(other_dirs):
    loader.other_dirs = other_dirs


load_library = loader.load_library

del loaderclass

# End loader

add_library_search_dirs([])

# Begin libraries

_libs["edfapi"] = load_library("edfapi")

# 1 libraries
# End libraries

# No modules

byte = c_ubyte  # <input>: 101

INT16 = c_short  # <input>: 102

INT32 = c_int  # <input>: 103

UINT16 = c_ushort  # <input>: 104

UINT32 = c_uint  # <input>: 105

UINT64 = c_ulonglong  # <input>: 107

INT64 = c_longlong  # <input>: 108

# <input>: 115
class struct_anon_1(Structure):
    pass


struct_anon_1.__slots__ = [
    'msec',
    'usec',
]
struct_anon_1._fields_ = [
    ('msec', INT32),
    ('usec', INT16),
]

MICRO = struct_anon_1  # <input>: 115

# <input>: 186
class struct_anon_2(Structure):
    pass


struct_anon_2.__slots__ = [
    'len',
    'c',
]
struct_anon_2._fields_ = [
    ('len', INT16),
    ('c', c_char),
]

LSTRING = struct_anon_2  # <input>: 186

# <input>: 242
class struct_anon_3(Structure):
    pass


struct_anon_3.__slots__ = [
    'time',
    'px',
    'py',
    'hx',
    'hy',
    'pa',
    'gx',
    'gy',
    'rx',
    'ry',
    'gxvel',
    'gyvel',
    'hxvel',
    'hyvel',
    'rxvel',
    'ryvel',
    'fgxvel',
    'fgyvel',
    'fhxvel',
    'fhyvel',
    'frxvel',
    'fryvel',
    'hdata',
    'flags',
    'input',
    'buttons',
    'htype',
    'errors',
]
struct_anon_3._fields_ = [
    ('time', UINT32),
    ('px', c_float * 2),
    ('py', c_float * 2),
    ('hx', c_float * 2),
    ('hy', c_float * 2),
    ('pa', c_float * 2),
    ('gx', c_float * 2),
    ('gy', c_float * 2),
    ('rx', c_float),
    ('ry', c_float),
    ('gxvel', c_float * 2),
    ('gyvel', c_float * 2),
    ('hxvel', c_float * 2),
    ('hyvel', c_float * 2),
    ('rxvel', c_float * 2),
    ('ryvel', c_float * 2),
    ('fgxvel', c_float * 2),
    ('fgyvel', c_float * 2),
    ('fhxvel', c_float * 2),
    ('fhyvel', c_float * 2),
    ('frxvel', c_float * 2),
    ('fryvel', c_float * 2),
    ('hdata', INT16 * 8),
    ('flags', UINT16),
    ('input', UINT16),
    ('buttons', UINT16),
    ('htype', INT16),
    ('errors', UINT16),
]

FSAMPLE = struct_anon_3  # <input>: 242

# <input>: 288
class struct_anon_4(Structure):
    pass


struct_anon_4.__slots__ = [
    'time',
    'type',
    'read',
    'sttime',
    'entime',
    'hstx',
    'hsty',
    'gstx',
    'gsty',
    'sta',
    'henx',
    'heny',
    'genx',
    'geny',
    'ena',
    'havx',
    'havy',
    'gavx',
    'gavy',
    'ava',
    'avel',
    'pvel',
    'svel',
    'evel',
    'supd_x',
    'eupd_x',
    'supd_y',
    'eupd_y',
    'eye',
    'status',
    'flags',
    'input',
    'buttons',
    'parsedby',
    'message',
]
struct_anon_4._fields_ = [
    ('time', UINT32),
    ('type', INT16),
    ('read', UINT16),
    ('sttime', UINT32),
    ('entime', UINT32),
    ('hstx', c_float),
    ('hsty', c_float),
    ('gstx', c_float),
    ('gsty', c_float),
    ('sta', c_float),
    ('henx', c_float),
    ('heny', c_float),
    ('genx', c_float),
    ('geny', c_float),
    ('ena', c_float),
    ('havx', c_float),
    ('havy', c_float),
    ('gavx', c_float),
    ('gavy', c_float),
    ('ava', c_float),
    ('avel', c_float),
    ('pvel', c_float),
    ('svel', c_float),
    ('evel', c_float),
    ('supd_x', c_float),
    ('eupd_x', c_float),
    ('supd_y', c_float),
    ('eupd_y', c_float),
    ('eye', INT16),
    ('status', UINT16),
    ('flags', UINT16),
    ('input', UINT16),
    ('buttons', UINT16),
    ('parsedby', UINT16),
    ('message', POINTER(LSTRING)),
]

FEVENT = struct_anon_4  # <input>: 288

# <input>: 297
class struct_anon_5(Structure):
    pass


struct_anon_5.__slots__ = [
    'time',
    'type',
    'length',
    'text',
]
struct_anon_5._fields_ = [
    ('time', UINT32),
    ('type', INT16),
    ('length', UINT16),
    ('text', byte * 260),
]

IMESSAGE = struct_anon_5  # <input>: 297

# <input>: 306
class struct_anon_6(Structure):
    pass


struct_anon_6.__slots__ = [
    'time',
    'type',
    'data',
]
struct_anon_6._fields_ = [
    ('time', UINT32),
    ('type', INT16),
    ('data', UINT16),
]

IOEVENT = struct_anon_6  # <input>: 306

# <input>: 330
class struct_anon_7(Structure):
    pass


struct_anon_7.__slots__ = [
    'time',
    'sample_rate',
    'eflags',
    'sflags',
    'state',
    'record_type',
    'pupil_type',
    'recording_mode',
    'filter_type',
    'pos_type',
    'eye',
]
struct_anon_7._fields_ = [
    ('time', UINT32),
    ('sample_rate', c_float),
    ('eflags', UINT16),
    ('sflags', UINT16),
    ('state', byte),
    ('record_type', byte),
    ('pupil_type', byte),
    ('recording_mode', byte),
    ('filter_type', byte),
    ('pos_type', byte),
    ('eye', byte),
]

RECORDINGS = struct_anon_7  # <input>: 330

# <input>: 341
class union_anon_8(Union):
    pass


union_anon_8.__slots__ = [
    'fe',
    'im',
    'io',
    'fs',
    'rec',
]
union_anon_8._fields_ = [
    ('fe', FEVENT),
    ('im', IMESSAGE),
    ('io', IOEVENT),
    ('fs', FSAMPLE),
    ('rec', RECORDINGS),
]

ALLF_DATA = union_anon_8  # <input>: 341

enum_anon_9 = c_int  # <input>: 534

GAZE = 0  # <input>: 534

HREF = (GAZE + 1)  # <input>: 534

RAW = (HREF + 1)  # <input>: 534

position_type = enum_anon_9  # <input>: 534

# <input>: 565
class struct_anon_10(Structure):
    pass


struct_anon_10.__slots__ = [
    'rec',
    'duration',
    'starttime',
    'endtime',
]
struct_anon_10._fields_ = [
    ('rec', POINTER(RECORDINGS)),
    ('duration', c_uint),
    ('starttime', c_uint),
    ('endtime', c_uint),
]

TRIAL = struct_anon_10  # <input>: 565

# <input>: 571
class struct__EDFFILE(Structure):
    pass


EDFFILE = struct__EDFFILE  # <input>: 571

# <input>: 585
class struct_anon_11(Structure):
    pass


struct_anon_11.__slots__ = [
    'id',
]
struct_anon_11._fields_ = [
    ('id', c_uint),
]

BOOKMARK = struct_anon_11  # <input>: 585

# <input>: 628
if hasattr(_libs['edfapi'], 'edf_open_file'):
    open_file = _libs['edfapi'].edf_open_file
    open_file.argtypes = [String, c_int, c_int, c_int, POINTER(c_int)]
    open_file.restype = POINTER(EDFFILE)

# <input>: 649
if hasattr(_libs['edfapi'], 'edf_close_file'):
    close_file = _libs['edfapi'].edf_close_file
    close_file.argtypes = [POINTER(EDFFILE)]
    close_file.restype = c_int

# <input>: 718
if hasattr(_libs['edfapi'], 'edf_get_next_data'):
    get_next_data = _libs['edfapi'].edf_get_next_data
    get_next_data.argtypes = [POINTER(EDFFILE)]
    get_next_data.restype = c_int

# <input>: 738
if hasattr(_libs['edfapi'], 'edf_get_float_data'):
    get_float_data = _libs['edfapi'].edf_get_float_data
    get_float_data.argtypes = [POINTER(EDFFILE)]
    get_float_data.restype = POINTER(ALLF_DATA)

# <input>: 748
if hasattr(_libs['edfapi'], 'edf_get_sample_close_to_time'):
    get_sample_close_to_time = _libs['edfapi'].edf_get_sample_close_to_time
    get_sample_close_to_time.argtypes = [POINTER(EDFFILE), c_uint]
    get_sample_close_to_time.restype = POINTER(ALLF_DATA)

# <input>: 760
if hasattr(_libs['edfapi'], 'edf_get_element_count'):
    get_element_count = _libs['edfapi'].edf_get_element_count
    get_element_count.argtypes = [POINTER(EDFFILE)]
    get_element_count.restype = c_uint

# <input>: 780
if hasattr(_libs['edfapi'], 'edf_get_preamble_text'):
    get_preamble_text = _libs['edfapi'].edf_get_preamble_text
    get_preamble_text.argtypes = [POINTER(EDFFILE), String, c_int]
    get_preamble_text.restype = c_int

# <input>: 795
if hasattr(_libs['edfapi'], 'edf_get_preamble_text_length'):
    get_preamble_text_length = _libs['edfapi'].edf_get_preamble_text_length
    get_preamble_text_length.argtypes = [POINTER(EDFFILE)]
    get_preamble_text_length.restype = c_int

# <input>: 808
if hasattr(_libs['edfapi'], 'edf_get_revision'):
    get_revision = _libs['edfapi'].edf_get_revision
    get_revision.argtypes = [POINTER(EDFFILE)]
    get_revision.restype = c_int

# <input>: 818
if hasattr(_libs['edfapi'], 'edf_get_eyelink_revision'):
    get_eyelink_revision = _libs['edfapi'].edf_get_eyelink_revision
    get_eyelink_revision.argtypes = [POINTER(EDFFILE)]
    get_eyelink_revision.restype = c_int

# <input>: 868
if hasattr(_libs['edfapi'], 'edf_set_trial_identifier'):
    set_trial_identifier = _libs['edfapi'].edf_set_trial_identifier
    set_trial_identifier.argtypes = [POINTER(EDFFILE), String, String]
    set_trial_identifier.restype = c_int

# <input>: 883
if hasattr(_libs['edfapi'], 'edf_get_start_trial_identifier'):
    get_start_trial_identifier = _libs['edfapi'].edf_get_start_trial_identifier
    get_start_trial_identifier.argtypes = [POINTER(EDFFILE)]
    if sizeof(c_int) == sizeof(c_void_p):
        get_start_trial_identifier.restype = ReturnString
    else:
        get_start_trial_identifier.restype = String
        get_start_trial_identifier.errcheck = ReturnString

# <input>: 895
if hasattr(_libs['edfapi'], 'edf_get_end_trial_identifier'):
    get_end_trial_identifier = _libs['edfapi'].edf_get_end_trial_identifier
    get_end_trial_identifier.argtypes = [POINTER(EDFFILE)]
    if sizeof(c_int) == sizeof(c_void_p):
        get_end_trial_identifier.restype = ReturnString
    else:
        get_end_trial_identifier.restype = String
        get_end_trial_identifier.errcheck = ReturnString

# <input>: 907
if hasattr(_libs['edfapi'], 'edf_get_trial_count'):
    get_trial_count = _libs['edfapi'].edf_get_trial_count
    get_trial_count.argtypes = [POINTER(EDFFILE)]
    get_trial_count.restype = c_int

# <input>: 922
if hasattr(_libs['edfapi'], 'edf_jump_to_trial'):
    jump_to_trial = _libs['edfapi'].edf_jump_to_trial
    jump_to_trial.argtypes = [POINTER(EDFFILE), c_int]
    jump_to_trial.restype = c_int

# <input>: 943
if hasattr(_libs['edfapi'], 'edf_get_trial_header'):
    get_trial_header = _libs['edfapi'].edf_get_trial_header
    get_trial_header.argtypes = [POINTER(EDFFILE), POINTER(TRIAL)]
    get_trial_header.restype = c_int

# <input>: 958
if hasattr(_libs['edfapi'], 'edf_goto_previous_trial'):
    goto_previous_trial = _libs['edfapi'].edf_goto_previous_trial
    goto_previous_trial.argtypes = [POINTER(EDFFILE)]
    goto_previous_trial.restype = c_int

# <input>: 971
if hasattr(_libs['edfapi'], 'edf_goto_next_trial'):
    goto_next_trial = _libs['edfapi'].edf_goto_next_trial
    goto_next_trial.argtypes = [POINTER(EDFFILE)]
    goto_next_trial.restype = c_int

# <input>: 984
if hasattr(_libs['edfapi'], 'edf_goto_trial_with_start_time'):
    goto_trial_with_start_time = _libs['edfapi'].edf_goto_trial_with_start_time
    goto_trial_with_start_time.argtypes = [POINTER(EDFFILE), c_uint]
    goto_trial_with_start_time.restype = c_int

# <input>: 997
if hasattr(_libs['edfapi'], 'edf_goto_trial_with_end_time'):
    goto_trial_with_end_time = _libs['edfapi'].edf_goto_trial_with_end_time
    goto_trial_with_end_time.argtypes = [POINTER(EDFFILE), c_uint]
    goto_trial_with_end_time.restype = c_int

# <input>: 1034
if hasattr(_libs['edfapi'], 'edf_set_bookmark'):
    set_bookmark = _libs['edfapi'].edf_set_bookmark
    set_bookmark.argtypes = [POINTER(EDFFILE), POINTER(BOOKMARK)]
    set_bookmark.restype = c_int

# <input>: 1049
if hasattr(_libs['edfapi'], 'edf_free_bookmark'):
    free_bookmark = _libs['edfapi'].edf_free_bookmark
    free_bookmark.argtypes = [POINTER(EDFFILE), POINTER(BOOKMARK)]
    free_bookmark.restype = c_int

# <input>: 1065
if hasattr(_libs['edfapi'], 'edf_goto_bookmark'):
    goto_bookmark = _libs['edfapi'].edf_goto_bookmark
    goto_bookmark.argtypes = [POINTER(EDFFILE), POINTER(BOOKMARK)]
    goto_bookmark.restype = c_int

# <input>: 1074
if hasattr(_libs['edfapi'], 'edf_goto_next_bookmark'):
    goto_next_bookmark = _libs['edfapi'].edf_goto_next_bookmark
    goto_next_bookmark.argtypes = [POINTER(EDFFILE)]
    goto_next_bookmark.restype = c_int

# <input>: 1082
if hasattr(_libs['edfapi'], 'edf_goto_previous_bookmark'):
    goto_previous_bookmark = _libs['edfapi'].edf_goto_previous_bookmark
    goto_previous_bookmark.argtypes = [POINTER(EDFFILE)]
    goto_previous_bookmark.restype = c_int

# <input>: 1097
if hasattr(_libs['edfapi'], 'edf_get_version'):
    get_version = _libs['edfapi'].edf_get_version
    get_version.argtypes = []
    if sizeof(c_int) == sizeof(c_void_p):
        get_version.restype = ReturnString
    else:
        get_version.restype = String
        get_version.errcheck = ReturnString

# <input>: 1108
if hasattr(_libs['edfapi'], 'edf_get_event'):
    get_event = _libs['edfapi'].edf_get_event
    get_event.argtypes = [POINTER(ALLF_DATA)]
    get_event.restype = POINTER(FEVENT)

# <input>: 1116
if hasattr(_libs['edfapi'], 'edf_get_sample'):
    get_sample = _libs['edfapi'].edf_get_sample
    get_sample.argtypes = [POINTER(ALLF_DATA)]
    get_sample.restype = POINTER(FSAMPLE)

# <input>: 1124
if hasattr(_libs['edfapi'], 'edf_get_recording'):
    get_recording = _libs['edfapi'].edf_get_recording
    get_recording.argtypes = [POINTER(ALLF_DATA)]
    get_recording.restype = POINTER(RECORDINGS)

# <input>: 1142
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_raw_pupil'):
    get_uncorrected_raw_pupil = _libs['edfapi'].edf_get_uncorrected_raw_pupil
    get_uncorrected_raw_pupil.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                          c_int, POINTER(c_float)]
    get_uncorrected_raw_pupil.restype = None

# <input>: 1143
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_raw_cr'):
    get_uncorrected_raw_cr = _libs['edfapi'].edf_get_uncorrected_raw_cr
    get_uncorrected_raw_cr.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                       c_int, POINTER(c_float)]
    get_uncorrected_raw_cr.restype = None

# <input>: 1144
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_pupil_area'):
    get_uncorrected_pupil_area = _libs['edfapi'].edf_get_uncorrected_pupil_area
    get_uncorrected_pupil_area.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                           c_int]
    get_uncorrected_pupil_area.restype = UINT32

# <input>: 1145
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_cr_area'):
    get_uncorrected_cr_area = _libs['edfapi'].edf_get_uncorrected_cr_area
    get_uncorrected_cr_area.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                        c_int]
    get_uncorrected_cr_area.restype = UINT32

# <input>: 1146
if hasattr(_libs['edfapi'], 'edf_get_pupil_dimension'):
    get_pupil_dimension = _libs['edfapi'].edf_get_pupil_dimension
    get_pupil_dimension.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int,
                                    POINTER(UINT32)]
    get_pupil_dimension.restype = None

# <input>: 1147
if hasattr(_libs['edfapi'], 'edf_get_cr_dimension'):
    get_cr_dimension = _libs['edfapi'].edf_get_cr_dimension
    get_cr_dimension.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                 POINTER(UINT32)]
    get_cr_dimension.restype = None

# <input>: 1148
if hasattr(_libs['edfapi'], 'edf_get_window_position'):
    get_window_position = _libs['edfapi'].edf_get_window_position
    get_window_position.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                    POINTER(UINT32)]
    get_window_position.restype = None

# <input>: 1149
if hasattr(_libs['edfapi'], 'edf_get_pupil_cr'):
    get_pupil_cr = _libs['edfapi'].edf_get_pupil_cr
    get_pupil_cr.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int,
                             POINTER(c_float)]
    get_pupil_cr.restype = None

# <input>: 1150
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_cr2_area'):
    get_uncorrected_cr2_area = _libs['edfapi'].edf_get_uncorrected_cr2_area
    get_uncorrected_cr2_area.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                         c_int]
    get_uncorrected_cr2_area.restype = UINT32

# <input>: 1151
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_raw_cr2'):
    get_uncorrected_raw_cr2 = _libs['edfapi'].edf_get_uncorrected_raw_cr2
    get_uncorrected_raw_cr2.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE),
                                        c_int, POINTER(c_float)]
    get_uncorrected_raw_cr2.restype = None

# <input>: 1159
if hasattr(_libs['edfapi'], 'edf_get_event_data'):
    get_event_data = _libs['edfapi'].edf_get_event_data
    get_event_data.argtypes = [POINTER(EDFFILE)]
    get_event_data.restype = POINTER(FEVENT)

# <input>: 1160
if hasattr(_libs['edfapi'], 'edf_get_sample_data'):
    get_sample_data = _libs['edfapi'].edf_get_sample_data
    get_sample_data.argtypes = [POINTER(EDFFILE)]
    get_sample_data.restype = POINTER(FSAMPLE)

# <input>: 1161
if hasattr(_libs['edfapi'], 'edf_get_recording_data'):
    get_recording_data = _libs['edfapi'].edf_get_recording_data
    get_recording_data.argtypes = [POINTER(EDFFILE)]
    get_recording_data.restype = POINTER(RECORDINGS)

# <input>: 1167
if hasattr(_libs['edfapi'], 'edf_set_log_function'):
    set_log_function = _libs['edfapi'].edf_set_log_function
    set_log_function.argtypes = [CFUNCTYPE(UNCHECKED(None), String)]
    set_log_function.restype = None

_EDFFILE = struct__EDFFILE  # <input>: 571

# Begin inserted files

# Begin "edf2py_extra.py"

# Contents of this file are appended to the end of the autogenerated
# edf2py.py script.


import ctypes as ct


def edf_file(file_name):
    error_code = ct.c_int(1)
    file_path = os.path.normpath(os.path.abspath(file_name))
    edf_ptr = open_file(file_path, 2, 1, 1, ct.byref(error_code))
    if edf_ptr is None or error_code.value != 0:
        return None
    return edf_ptr


def preamble_text(edfptr):
    preambleText = String()
    tlen = get_preamble_text_length(edfptr)
    get_preamble_text(edfptr, preambleText, tlen + 1)
    return preambleText


from ._defines import edf_constants as constants
from ._defines import event_constants as event_constants
from ._defines import eye_constants as eye_constants

# End "edf2py_extra.py"

# 1 inserted files
# End inserted files
