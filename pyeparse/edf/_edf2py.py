'''Wrapper for edf.h

Generated with:
/home/larsoner/.local/bin/ctypesgen.py -a -l edfapi -o ../pyeparse/edf/_edf2py.py /usr/include/edf.h

Do not modify this file.
'''

__docformat__ =  'restructuredtext'

# Begin preamble

import ctypes, os, sys
from ctypes import *

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
    def __str__(self): return str(self.data)
    def __repr__(self): return repr(self.data)
    def __int__(self): return int(self.data)
    def __long__(self): return long(self.data)
    def __float__(self): return float(self.data)
    def __complex__(self): return complex(self.data)
    def __hash__(self): return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)
    def __contains__(self, char):
        return char in self.data

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.__class__(self.data[index])
    def __getslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
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
        return self.__class__(self.data*n)
    __rmul__ = __mul__
    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self): return self.__class__(self.data.capitalize())
    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))
    def count(self, sub, start=0, end=sys.maxint):
        return self.data.count(sub, start, end)
    def decode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())
    def encode(self, encoding=None, errors=None): # XXX improve this?
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
    def isalpha(self): return self.data.isalpha()
    def isalnum(self): return self.data.isalnum()
    def isdecimal(self): return self.data.isdecimal()
    def isdigit(self): return self.data.isdigit()
    def islower(self): return self.data.islower()
    def isnumeric(self): return self.data.isnumeric()
    def isspace(self): return self.data.isspace()
    def istitle(self): return self.data.istitle()
    def isupper(self): return self.data.isupper()
    def join(self, seq): return self.data.join(seq)
    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))
    def lower(self): return self.__class__(self.data.lower())
    def lstrip(self, chars=None): return self.__class__(self.data.lstrip(chars))
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
    def rstrip(self, chars=None): return self.__class__(self.data.rstrip(chars))
    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)
    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)
    def splitlines(self, keepends=0): return self.data.splitlines(keepends)
    def startswith(self, prefix, start=0, end=sys.maxint):
        return self.data.startswith(prefix, start, end)
    def strip(self, chars=None): return self.__class__(self.data.strip(chars))
    def swapcase(self): return self.__class__(self.data.swapcase())
    def title(self): return self.__class__(self.data.title())
    def translate(self, *args):
        return self.__class__(self.data.translate(*args))
    def upper(self): return self.__class__(self.data.upper())
    def zfill(self, width): return self.__class__(self.data.zfill(width))

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
        self.data = self.data[:index] + sub + self.data[index+1:]
    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index+1:]
    def __setslice__(self, start, end, sub):
        start = max(start, 0); end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start]+sub.data+self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start]+sub+self.data[end:]
        else:
            self.data =  self.data[:start]+str(sub)+self.data[end:]
    def __delslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
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
    def __init__(self,func,restype,argtypes):
        self.func=func
        self.func.restype=restype
        self.argtypes=argtypes
    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func
    def __call__(self,*args):
        fixed_args=[]
        i=0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i+=1
        return self.func(*fixed_args+list(args[i:]))

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

import os.path, re, sys, glob
import platform
import ctypes
import ctypes.util

def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []

class LibraryLoader(object):
    def __init__(self):
        self.other_dirs=[]

    def load_library(self,libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self,path):
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
        except OSError,e:
            raise ImportError(e)

    def getpaths(self,libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # FIXME / TODO return '.' and os.path.dirname(__file__)
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

    def getplatformpaths(self,libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir,name)

    def getdirs(self,libname):
        '''Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        '''

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
                     "SHLIB_PATH", # HPUX
                     "LIBPATH", # OS/2, AIX
                     "LIBRARY_PATH", # BE/OS
                    ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")
        directories.append(os.path.dirname(__file__))

        try: directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError: pass

        unix_lib_dirs_list = ['/lib', '/usr/lib', '/lib64', '/usr/lib64']
        if sys.platform.startswith('linux'):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            bitage = platform.architecture()[0]
            if bitage.startswith('32'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/i386-linux-gnu', '/usr/lib/i386-linux-gnu']
            elif bitage.startswith('64'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu']
            else:
                # guess...
                unix_lib_dirs_list += glob.glob('/lib/*linux-gnu')
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r'lib(.*)\.s[ol]')
        ext_re = re.compile(r'\.s[ol]$')
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
        if path: yield os.path.join("/lib",path)

# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try: return getattr(self.cdll,name)
        except AttributeError:
            try: return getattr(self.windll,name)
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
    "darwin":   DarwinLibraryLoader,
    "cygwin":   WindowsLibraryLoader,
    "win32":    WindowsLibraryLoader
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

byte = c_ubyte # /usr/include/edftypes.h: 19

INT16 = c_short # /usr/include/edftypes.h: 20

INT32 = c_int # /usr/include/edftypes.h: 21

UINT16 = c_ushort # /usr/include/edftypes.h: 22

UINT32 = c_uint # /usr/include/edftypes.h: 23

UINT64 = c_ulonglong # /usr/include/edftypes.h: 28

INT64 = c_longlong # /usr/include/edftypes.h: 29

# /usr/include/edftypes.h: 40
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

MICRO = struct_anon_1 # /usr/include/edftypes.h: 40

# /usr/include/edf_data.h: 135
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

LSTRING = struct_anon_2 # /usr/include/edf_data.h: 135

# /usr/include/edf_data.h: 184
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

FSAMPLE = struct_anon_3 # /usr/include/edf_data.h: 184

# /usr/include/edf_data.h: 226
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

FEVENT = struct_anon_4 # /usr/include/edf_data.h: 226

# /usr/include/edf_data.h: 244
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

IMESSAGE = struct_anon_5 # /usr/include/edf_data.h: 244

# /usr/include/edf_data.h: 259
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

IOEVENT = struct_anon_6 # /usr/include/edf_data.h: 259

# /usr/include/edf_data.h: 283
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

RECORDINGS = struct_anon_7 # /usr/include/edf_data.h: 283

# /usr/include/edf_data.h: 301
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

ALLF_DATA = union_anon_8 # /usr/include/edf_data.h: 301

enum_anon_9 = c_int # /usr/include/edf.h: 79

GAZE = 0 # /usr/include/edf.h: 79

HREF = (GAZE + 1) # /usr/include/edf.h: 79

RAW = (HREF + 1) # /usr/include/edf.h: 79

position_type = enum_anon_9 # /usr/include/edf.h: 79

# /usr/include/edf.h: 92
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

TRIAL = struct_anon_10 # /usr/include/edf.h: 92

# /usr/include/edf.h: 94
class struct__EDFFILE(Structure):
    pass

EDFFILE = struct__EDFFILE # /usr/include/edf.h: 94

# /usr/include/edf.h: 98
class struct_anon_11(Structure):
    pass

struct_anon_11.__slots__ = [
    'id',
]
struct_anon_11._fields_ = [
    ('id', c_uint),
]

BOOKMARK = struct_anon_11 # /usr/include/edf.h: 98

# /usr/include/edf.h: 118
if hasattr(_libs['edfapi'], 'edf_open_file'):
    edf_open_file = _libs['edfapi'].edf_open_file
    edf_open_file.argtypes = [String, c_int, c_int, c_int, POINTER(c_int)]
    edf_open_file.restype = POINTER(EDFFILE)

# /usr/include/edf.h: 131
if hasattr(_libs['edfapi'], 'edf_close_file'):
    edf_close_file = _libs['edfapi'].edf_close_file
    edf_close_file.argtypes = [POINTER(EDFFILE)]
    edf_close_file.restype = c_int

# /usr/include/edf.h: 173
if hasattr(_libs['edfapi'], 'edf_get_next_data'):
    edf_get_next_data = _libs['edfapi'].edf_get_next_data
    edf_get_next_data.argtypes = [POINTER(EDFFILE)]
    edf_get_next_data.restype = c_int

# /usr/include/edf.h: 183
if hasattr(_libs['edfapi'], 'edf_get_float_data'):
    edf_get_float_data = _libs['edfapi'].edf_get_float_data
    edf_get_float_data.argtypes = [POINTER(EDFFILE)]
    edf_get_float_data.restype = POINTER(ALLF_DATA)

# /usr/include/edf.h: 193
if hasattr(_libs['edfapi'], 'edf_get_sample_close_to_time'):
    edf_get_sample_close_to_time = _libs['edfapi'].edf_get_sample_close_to_time
    edf_get_sample_close_to_time.argtypes = [POINTER(EDFFILE), c_uint]
    edf_get_sample_close_to_time.restype = POINTER(ALLF_DATA)

# /usr/include/edf.h: 201
if hasattr(_libs['edfapi'], 'edf_get_element_count'):
    edf_get_element_count = _libs['edfapi'].edf_get_element_count
    edf_get_element_count.argtypes = [POINTER(EDFFILE)]
    edf_get_element_count.restype = c_uint

# /usr/include/edf.h: 212
if hasattr(_libs['edfapi'], 'edf_get_preamble_text'):
    edf_get_preamble_text = _libs['edfapi'].edf_get_preamble_text
    edf_get_preamble_text.argtypes = [POINTER(EDFFILE), String, c_int]
    edf_get_preamble_text.restype = c_int

# /usr/include/edf.h: 222
if hasattr(_libs['edfapi'], 'edf_get_preamble_text_length'):
    edf_get_preamble_text_length = _libs['edfapi'].edf_get_preamble_text_length
    edf_get_preamble_text_length.argtypes = [POINTER(EDFFILE)]
    edf_get_preamble_text_length.restype = c_int

# /usr/include/edf.h: 235
if hasattr(_libs['edfapi'], 'edf_get_revision'):
    edf_get_revision = _libs['edfapi'].edf_get_revision
    edf_get_revision.argtypes = [POINTER(EDFFILE)]
    edf_get_revision.restype = c_int

# /usr/include/edf.h: 245
if hasattr(_libs['edfapi'], 'edf_get_eyelink_revision'):
    edf_get_eyelink_revision = _libs['edfapi'].edf_get_eyelink_revision
    edf_get_eyelink_revision.argtypes = [POINTER(EDFFILE)]
    edf_get_eyelink_revision.restype = c_int

# /usr/include/edf.h: 262
if hasattr(_libs['edfapi'], 'edf_set_trial_identifier'):
    edf_set_trial_identifier = _libs['edfapi'].edf_set_trial_identifier
    edf_set_trial_identifier.argtypes = [POINTER(EDFFILE), String, String]
    edf_set_trial_identifier.restype = c_int

# /usr/include/edf.h: 273
if hasattr(_libs['edfapi'], 'edf_get_start_trial_identifier'):
    edf_get_start_trial_identifier = _libs['edfapi'].edf_get_start_trial_identifier
    edf_get_start_trial_identifier.argtypes = [POINTER(EDFFILE)]
    if sizeof(c_int) == sizeof(c_void_p):
        edf_get_start_trial_identifier.restype = ReturnString
    else:
        edf_get_start_trial_identifier.restype = String
        edf_get_start_trial_identifier.errcheck = ReturnString

# /usr/include/edf.h: 281
if hasattr(_libs['edfapi'], 'edf_get_end_trial_identifier'):
    edf_get_end_trial_identifier = _libs['edfapi'].edf_get_end_trial_identifier
    edf_get_end_trial_identifier.argtypes = [POINTER(EDFFILE)]
    if sizeof(c_int) == sizeof(c_void_p):
        edf_get_end_trial_identifier.restype = ReturnString
    else:
        edf_get_end_trial_identifier.restype = String
        edf_get_end_trial_identifier.errcheck = ReturnString

# /usr/include/edf.h: 289
if hasattr(_libs['edfapi'], 'edf_get_trial_count'):
    edf_get_trial_count = _libs['edfapi'].edf_get_trial_count
    edf_get_trial_count.argtypes = [POINTER(EDFFILE)]
    edf_get_trial_count.restype = c_int

# /usr/include/edf.h: 298
if hasattr(_libs['edfapi'], 'edf_jump_to_trial'):
    edf_jump_to_trial = _libs['edfapi'].edf_jump_to_trial
    edf_jump_to_trial.argtypes = [POINTER(EDFFILE), c_int]
    edf_jump_to_trial.restype = c_int

# /usr/include/edf.h: 309
if hasattr(_libs['edfapi'], 'edf_get_trial_header'):
    edf_get_trial_header = _libs['edfapi'].edf_get_trial_header
    edf_get_trial_header.argtypes = [POINTER(EDFFILE), POINTER(TRIAL)]
    edf_get_trial_header.restype = c_int

# /usr/include/edf.h: 320
if hasattr(_libs['edfapi'], 'edf_goto_previous_trial'):
    edf_goto_previous_trial = _libs['edfapi'].edf_goto_previous_trial
    edf_goto_previous_trial.argtypes = [POINTER(EDFFILE)]
    edf_goto_previous_trial.restype = c_int

# /usr/include/edf.h: 329
if hasattr(_libs['edfapi'], 'edf_goto_next_trial'):
    edf_goto_next_trial = _libs['edfapi'].edf_goto_next_trial
    edf_goto_next_trial.argtypes = [POINTER(EDFFILE)]
    edf_goto_next_trial.restype = c_int

# /usr/include/edf.h: 338
if hasattr(_libs['edfapi'], 'edf_goto_trial_with_start_time'):
    edf_goto_trial_with_start_time = _libs['edfapi'].edf_goto_trial_with_start_time
    edf_goto_trial_with_start_time.argtypes = [POINTER(EDFFILE), c_uint]
    edf_goto_trial_with_start_time.restype = c_int

# /usr/include/edf.h: 347
if hasattr(_libs['edfapi'], 'edf_goto_trial_with_end_time'):
    edf_goto_trial_with_end_time = _libs['edfapi'].edf_goto_trial_with_end_time
    edf_goto_trial_with_end_time.argtypes = [POINTER(EDFFILE), c_uint]
    edf_goto_trial_with_end_time.restype = c_int

# /usr/include/edf.h: 372
if hasattr(_libs['edfapi'], 'edf_set_bookmark'):
    edf_set_bookmark = _libs['edfapi'].edf_set_bookmark
    edf_set_bookmark.argtypes = [POINTER(EDFFILE), POINTER(BOOKMARK)]
    edf_set_bookmark.restype = c_int

# /usr/include/edf.h: 380
if hasattr(_libs['edfapi'], 'edf_free_bookmark'):
    edf_free_bookmark = _libs['edfapi'].edf_free_bookmark
    edf_free_bookmark.argtypes = [POINTER(EDFFILE), POINTER(BOOKMARK)]
    edf_free_bookmark.restype = c_int

# /usr/include/edf.h: 389
if hasattr(_libs['edfapi'], 'edf_goto_bookmark'):
    edf_goto_bookmark = _libs['edfapi'].edf_goto_bookmark
    edf_goto_bookmark.argtypes = [POINTER(EDFFILE), POINTER(BOOKMARK)]
    edf_goto_bookmark.restype = c_int

# /usr/include/edf.h: 398
if hasattr(_libs['edfapi'], 'edf_goto_next_bookmark'):
    edf_goto_next_bookmark = _libs['edfapi'].edf_goto_next_bookmark
    edf_goto_next_bookmark.argtypes = [POINTER(EDFFILE)]
    edf_goto_next_bookmark.restype = c_int

# /usr/include/edf.h: 406
if hasattr(_libs['edfapi'], 'edf_goto_previous_bookmark'):
    edf_goto_previous_bookmark = _libs['edfapi'].edf_goto_previous_bookmark
    edf_goto_previous_bookmark.argtypes = [POINTER(EDFFILE)]
    edf_goto_previous_bookmark.restype = c_int

# /usr/include/edf.h: 416
if hasattr(_libs['edfapi'], 'edf_get_version'):
    edf_get_version = _libs['edfapi'].edf_get_version
    edf_get_version.argtypes = []
    if sizeof(c_int) == sizeof(c_void_p):
        edf_get_version.restype = ReturnString
    else:
        edf_get_version.restype = String
        edf_get_version.errcheck = ReturnString

# /usr/include/edf.h: 427
if hasattr(_libs['edfapi'], 'edf_get_event'):
    edf_get_event = _libs['edfapi'].edf_get_event
    edf_get_event.argtypes = [POINTER(ALLF_DATA)]
    edf_get_event.restype = POINTER(FEVENT)

# /usr/include/edf.h: 435
if hasattr(_libs['edfapi'], 'edf_get_sample'):
    edf_get_sample = _libs['edfapi'].edf_get_sample
    edf_get_sample.argtypes = [POINTER(ALLF_DATA)]
    edf_get_sample.restype = POINTER(FSAMPLE)

# /usr/include/edf.h: 443
if hasattr(_libs['edfapi'], 'edf_get_recording'):
    edf_get_recording = _libs['edfapi'].edf_get_recording
    edf_get_recording.argtypes = [POINTER(ALLF_DATA)]
    edf_get_recording.restype = POINTER(RECORDINGS)

# /usr/include/edf.h: 450
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_raw_pupil'):
    edf_get_uncorrected_raw_pupil = _libs['edfapi'].edf_get_uncorrected_raw_pupil
    edf_get_uncorrected_raw_pupil.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int, POINTER(c_float)]
    edf_get_uncorrected_raw_pupil.restype = None

# /usr/include/edf.h: 451
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_raw_cr'):
    edf_get_uncorrected_raw_cr = _libs['edfapi'].edf_get_uncorrected_raw_cr
    edf_get_uncorrected_raw_cr.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int, POINTER(c_float)]
    edf_get_uncorrected_raw_cr.restype = None

# /usr/include/edf.h: 452
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_pupil_area'):
    edf_get_uncorrected_pupil_area = _libs['edfapi'].edf_get_uncorrected_pupil_area
    edf_get_uncorrected_pupil_area.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int]
    edf_get_uncorrected_pupil_area.restype = UINT32

# /usr/include/edf.h: 453
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_cr_area'):
    edf_get_uncorrected_cr_area = _libs['edfapi'].edf_get_uncorrected_cr_area
    edf_get_uncorrected_cr_area.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int]
    edf_get_uncorrected_cr_area.restype = UINT32

# /usr/include/edf.h: 454
if hasattr(_libs['edfapi'], 'edf_get_pupil_dimension'):
    edf_get_pupil_dimension = _libs['edfapi'].edf_get_pupil_dimension
    edf_get_pupil_dimension.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int, POINTER(UINT32)]
    edf_get_pupil_dimension.restype = None

# /usr/include/edf.h: 455
if hasattr(_libs['edfapi'], 'edf_get_cr_dimension'):
    edf_get_cr_dimension = _libs['edfapi'].edf_get_cr_dimension
    edf_get_cr_dimension.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), POINTER(UINT32)]
    edf_get_cr_dimension.restype = None

# /usr/include/edf.h: 456
if hasattr(_libs['edfapi'], 'edf_get_window_position'):
    edf_get_window_position = _libs['edfapi'].edf_get_window_position
    edf_get_window_position.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), POINTER(UINT32)]
    edf_get_window_position.restype = None

# /usr/include/edf.h: 457
if hasattr(_libs['edfapi'], 'edf_get_pupil_cr'):
    edf_get_pupil_cr = _libs['edfapi'].edf_get_pupil_cr
    edf_get_pupil_cr.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int, POINTER(c_float)]
    edf_get_pupil_cr.restype = None

# /usr/include/edf.h: 458
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_cr2_area'):
    edf_get_uncorrected_cr2_area = _libs['edfapi'].edf_get_uncorrected_cr2_area
    edf_get_uncorrected_cr2_area.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int]
    edf_get_uncorrected_cr2_area.restype = UINT32

# /usr/include/edf.h: 459
if hasattr(_libs['edfapi'], 'edf_get_uncorrected_raw_cr2'):
    edf_get_uncorrected_raw_cr2 = _libs['edfapi'].edf_get_uncorrected_raw_cr2
    edf_get_uncorrected_raw_cr2.argtypes = [POINTER(EDFFILE), POINTER(FSAMPLE), c_int, POINTER(c_float)]
    edf_get_uncorrected_raw_cr2.restype = None

# /usr/include/edf.h: 467
if hasattr(_libs['edfapi'], 'edf_get_event_data'):
    edf_get_event_data = _libs['edfapi'].edf_get_event_data
    edf_get_event_data.argtypes = [POINTER(EDFFILE)]
    edf_get_event_data.restype = POINTER(FEVENT)

# /usr/include/edf.h: 468
if hasattr(_libs['edfapi'], 'edf_get_sample_data'):
    edf_get_sample_data = _libs['edfapi'].edf_get_sample_data
    edf_get_sample_data.argtypes = [POINTER(EDFFILE)]
    edf_get_sample_data.restype = POINTER(FSAMPLE)

# /usr/include/edf.h: 469
if hasattr(_libs['edfapi'], 'edf_get_recording_data'):
    edf_get_recording_data = _libs['edfapi'].edf_get_recording_data
    edf_get_recording_data.argtypes = [POINTER(EDFFILE)]
    edf_get_recording_data.restype = POINTER(RECORDINGS)

# /usr/include/edf.h: 475
if hasattr(_libs['edfapi'], 'edf_set_log_function'):
    edf_set_log_function = _libs['edfapi'].edf_set_log_function
    edf_set_log_function.argtypes = [CFUNCTYPE(UNCHECKED(None), String)]
    edf_set_log_function.restype = None

# /tmp/tmpcxCqxA.h: 1
try:
    __STDC__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __STDC_HOSTED__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GNUC__ = 4
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GNUC_MINOR__ = 7
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GNUC_PATCHLEVEL__ = 3
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __VERSION__ = '4.7.3'
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ATOMIC_RELAXED = 0
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ATOMIC_SEQ_CST = 5
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ATOMIC_ACQUIRE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ATOMIC_RELEASE = 3
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ATOMIC_ACQ_REL = 4
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ATOMIC_CONSUME = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FINITE_MATH_ONLY__ = 0
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    _LP64 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LP64__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_INT__ = 4
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_LONG__ = 8
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_LONG_LONG__ = 8
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_SHORT__ = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_FLOAT__ = 4
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_DOUBLE__ = 8
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_LONG_DOUBLE__ = 16
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_SIZE_T__ = 8
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __CHAR_BIT__ = 8
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __BIGGEST_ALIGNMENT__ = 16
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ORDER_LITTLE_ENDIAN__ = 1234
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ORDER_BIG_ENDIAN__ = 4321
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ORDER_PDP_ENDIAN__ = 3412
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __BYTE_ORDER__ = __ORDER_LITTLE_ENDIAN__
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLOAT_WORD_ORDER__ = __ORDER_LITTLE_ENDIAN__
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_POINTER__ = 8
except:
    pass

__SIZE_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__PTRDIFF_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__WCHAR_TYPE__ = c_int # /tmp/tmpcxCqxA.h: 1

__WINT_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__INTMAX_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__UINTMAX_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__CHAR16_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__CHAR32_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__SIG_ATOMIC_TYPE__ = c_int # /tmp/tmpcxCqxA.h: 1

__INT8_TYPE__ = c_char # /tmp/tmpcxCqxA.h: 1

__INT16_TYPE__ = c_int # /tmp/tmpcxCqxA.h: 1

__INT32_TYPE__ = c_int # /tmp/tmpcxCqxA.h: 1

__INT64_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__UINT8_TYPE__ = c_ubyte # /tmp/tmpcxCqxA.h: 1

__UINT16_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__UINT32_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__UINT64_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__INT_LEAST8_TYPE__ = c_char # /tmp/tmpcxCqxA.h: 1

__INT_LEAST16_TYPE__ = c_int # /tmp/tmpcxCqxA.h: 1

__INT_LEAST32_TYPE__ = c_int # /tmp/tmpcxCqxA.h: 1

__INT_LEAST64_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__UINT_LEAST8_TYPE__ = c_ubyte # /tmp/tmpcxCqxA.h: 1

__UINT_LEAST16_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__UINT_LEAST32_TYPE__ = c_uint # /tmp/tmpcxCqxA.h: 1

__UINT_LEAST64_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__INT_FAST8_TYPE__ = c_char # /tmp/tmpcxCqxA.h: 1

__INT_FAST16_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__INT_FAST32_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__INT_FAST64_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__UINT_FAST8_TYPE__ = c_ubyte # /tmp/tmpcxCqxA.h: 1

__UINT_FAST16_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__UINT_FAST32_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__UINT_FAST64_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

__INTPTR_TYPE__ = c_long # /tmp/tmpcxCqxA.h: 1

__UINTPTR_TYPE__ = c_ulong # /tmp/tmpcxCqxA.h: 1

# /tmp/tmpcxCqxA.h: 1
try:
    __GXX_ABI_VERSION = 1002
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SCHAR_MAX__ = 127
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SHRT_MAX__ = 32767
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_MAX__ = 2147483647
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LONG_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LONG_LONG_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __WCHAR_MAX__ = 2147483647
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __WCHAR_MIN__ = ((-__WCHAR_MAX__) - 1)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __WINT_MAX__ = 4294967295
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __WINT_MIN__ = 0
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __PTRDIFF_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZE_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INTMAX_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINTMAX_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIG_ATOMIC_MAX__ = 2147483647
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIG_ATOMIC_MIN__ = ((-__SIG_ATOMIC_MAX__) - 1)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT8_MAX__ = 127
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT16_MAX__ = 32767
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT32_MAX__ = 2147483647
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT64_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT8_MAX__ = 255
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT16_MAX__ = 65535
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT32_MAX__ = 4294967295
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT64_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_LEAST8_MAX__ = 127
except:
    pass

# /tmp/tmpcxCqxA.h: 1
def __INT8_C(c):
    return c

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_LEAST16_MAX__ = 32767
except:
    pass

# /tmp/tmpcxCqxA.h: 1
def __INT16_C(c):
    return c

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_LEAST32_MAX__ = 2147483647
except:
    pass

# /tmp/tmpcxCqxA.h: 1
def __INT32_C(c):
    return c

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_LEAST64_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_LEAST8_MAX__ = 255
except:
    pass

# /tmp/tmpcxCqxA.h: 1
def __UINT8_C(c):
    return c

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_LEAST16_MAX__ = 65535
except:
    pass

# /tmp/tmpcxCqxA.h: 1
def __UINT16_C(c):
    return c

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_LEAST32_MAX__ = 4294967295
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_LEAST64_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_FAST8_MAX__ = 127
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_FAST16_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_FAST32_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INT_FAST64_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_FAST8_MAX__ = 255
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_FAST16_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_FAST32_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINT_FAST64_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __INTPTR_MAX__ = 9223372036854775807L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __UINTPTR_MAX__ = 18446744073709551615L
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_EVAL_METHOD__ = 0
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC_EVAL_METHOD__ = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_RADIX__ = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MANT_DIG__ = 24
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_DIG__ = 6
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MIN_EXP__ = (-125)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MIN_10_EXP__ = (-37)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MAX_EXP__ = 128
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MAX_10_EXP__ = 38
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_DECIMAL_DIG__ = 9
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MAX__ = 3.4028234663852886e+38
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_MIN__ = 1.1754943508222875e-38
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_EPSILON__ = 1.1920928955078125e-07
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_DENORM_MIN__ = 1.401298464324817e-45
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_HAS_DENORM__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_HAS_INFINITY__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __FLT_HAS_QUIET_NAN__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MANT_DIG__ = 53
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_DIG__ = 15
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MIN_EXP__ = (-1021)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MIN_10_EXP__ = (-307)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MAX_EXP__ = 1024
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MAX_10_EXP__ = 308
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_DECIMAL_DIG__ = 17
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MAX__ = 1.7976931348623157e+308
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_MIN__ = 2.2250738585072014e-308
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_EPSILON__ = 2.220446049250313e-16
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_DENORM_MIN__ = 5e-324
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_HAS_DENORM__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_HAS_INFINITY__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DBL_HAS_QUIET_NAN__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MANT_DIG__ = 64
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_DIG__ = 18
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MIN_EXP__ = (-16381)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MIN_10_EXP__ = (-4931)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MAX_EXP__ = 16384
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MAX_10_EXP__ = 4932
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DECIMAL_DIG__ = 21
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MAX__ = float('inf')
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_MIN__ = 0.0
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_EPSILON__ = 1.0842021724855044e-19
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_DENORM_MIN__ = 0.0
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_HAS_DENORM__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_HAS_INFINITY__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __LDBL_HAS_QUIET_NAN__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC32_MANT_DIG__ = 7
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC32_MIN_EXP__ = (-94)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC32_MAX_EXP__ = 97
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC64_MANT_DIG__ = 16
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC64_MIN_EXP__ = (-382)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC64_MAX_EXP__ = 385
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC128_MANT_DIG__ = 34
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC128_MIN_EXP__ = (-6142)
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DEC128_MAX_EXP__ = 6145
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GNUC_GNU_INLINE__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __NO_INLINE__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_BOOL_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_CHAR_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_CHAR16_T_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_CHAR32_T_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_WCHAR_T_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_SHORT_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_INT_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_LONG_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_LLONG_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_TEST_AND_SET_TRUEVAL = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_ATOMIC_POINTER_LOCK_FREE = 2
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __GCC_HAVE_DWARF2_CFI_ASM = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __PRAGMA_REDEFINE_EXTNAME = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SSP__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_INT128__ = 16
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_WCHAR_T__ = 4
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_WINT_T__ = 4
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SIZEOF_PTRDIFF_T__ = 8
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __amd64 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __amd64__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __x86_64 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __x86_64__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __k8 = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __k8__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __code_model_small__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __MMX__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SSE__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SSE2__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SSE_MATH__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __SSE2_MATH__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __gnu_linux__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __linux = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __linux__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    linux = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __unix = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __unix__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    unix = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __ELF__ = 1
except:
    pass

# /tmp/tmpcxCqxA.h: 1
try:
    __DECIMAL_BID_FORMAT__ = 1
except:
    pass

__const = c_int # <command-line>: 4

# <command-line>: 7
try:
    CTYPESGEN = 1
except:
    pass

# /usr/include/edftypes.h: 17
try:
    BYTEDEF = 1
except:
    pass

# /usr/include/edftypes.h: 36
try:
    MICRODEF = 1
except:
    pass

# /usr/include/edf_data.h: 75
try:
    MISSING_DATA = (-32768)
except:
    pass

# /usr/include/edf_data.h: 76
try:
    MISSING = (-32768)
except:
    pass

# /usr/include/edf_data.h: 77
try:
    INaN = (-32768)
except:
    pass

# /usr/include/edf_data.h: 84
try:
    LEFT_EYE = 0
except:
    pass

# /usr/include/edf_data.h: 85
try:
    RIGHT_EYE = 1
except:
    pass

# /usr/include/edf_data.h: 86
try:
    LEFTEYEI = 0
except:
    pass

# /usr/include/edf_data.h: 87
try:
    RIGHTEYEI = 1
except:
    pass

# /usr/include/edf_data.h: 88
try:
    LEFT = 0
except:
    pass

# /usr/include/edf_data.h: 89
try:
    RIGHT = 1
except:
    pass

# /usr/include/edf_data.h: 91
try:
    BINOCULAR = 2
except:
    pass

# /usr/include/edf_data.h: 107
try:
    SAMPLE_LEFT = 32768
except:
    pass

# /usr/include/edf_data.h: 108
try:
    SAMPLE_RIGHT = 16384
except:
    pass

# /usr/include/edf_data.h: 110
try:
    SAMPLE_TIMESTAMP = 8192
except:
    pass

# /usr/include/edf_data.h: 112
try:
    SAMPLE_PUPILXY = 4096
except:
    pass

# /usr/include/edf_data.h: 113
try:
    SAMPLE_HREFXY = 2048
except:
    pass

# /usr/include/edf_data.h: 114
try:
    SAMPLE_GAZEXY = 1024
except:
    pass

# /usr/include/edf_data.h: 115
try:
    SAMPLE_GAZERES = 512
except:
    pass

# /usr/include/edf_data.h: 116
try:
    SAMPLE_PUPILSIZE = 256
except:
    pass

# /usr/include/edf_data.h: 117
try:
    SAMPLE_STATUS = 128
except:
    pass

# /usr/include/edf_data.h: 118
try:
    SAMPLE_INPUTS = 64
except:
    pass

# /usr/include/edf_data.h: 119
try:
    SAMPLE_BUTTONS = 32
except:
    pass

# /usr/include/edf_data.h: 121
try:
    SAMPLE_HEADPOS = 16
except:
    pass

# /usr/include/edf_data.h: 122
try:
    SAMPLE_TAGGED = 8
except:
    pass

# /usr/include/edf_data.h: 123
try:
    SAMPLE_UTAGGED = 4
except:
    pass

# /usr/include/edf_data.h: 124
try:
    SAMPLE_ADD_OFFSET = 2
except:
    pass

# /usr/include/edf_data.h: 128
try:
    LSTRINGDEF = 1
except:
    pass

# /usr/include/edf_data.h: 140
try:
    FSAMPLEDEF = 1
except:
    pass

# /usr/include/edf_data.h: 192
try:
    FEVENTDEF = 1
except:
    pass

# /usr/include/edf_data.h: 308
try:
    SAMPLE_TYPE = 200
except:
    pass

# /usr/include/edf_data.h: 317
try:
    STARTPARSE = 1
except:
    pass

# /usr/include/edf_data.h: 318
try:
    ENDPARSE = 2
except:
    pass

# /usr/include/edf_data.h: 319
try:
    BREAKPARSE = 10
except:
    pass

# /usr/include/edf_data.h: 322
try:
    STARTBLINK = 3
except:
    pass

# /usr/include/edf_data.h: 323
try:
    ENDBLINK = 4
except:
    pass

# /usr/include/edf_data.h: 324
try:
    STARTSACC = 5
except:
    pass

# /usr/include/edf_data.h: 325
try:
    ENDSACC = 6
except:
    pass

# /usr/include/edf_data.h: 326
try:
    STARTFIX = 7
except:
    pass

# /usr/include/edf_data.h: 327
try:
    ENDFIX = 8
except:
    pass

# /usr/include/edf_data.h: 328
try:
    FIXUPDATE = 9
except:
    pass

# /usr/include/edf_data.h: 334
try:
    STARTSAMPLES = 15
except:
    pass

# /usr/include/edf_data.h: 335
try:
    ENDSAMPLES = 16
except:
    pass

# /usr/include/edf_data.h: 336
try:
    STARTEVENTS = 17
except:
    pass

# /usr/include/edf_data.h: 337
try:
    ENDEVENTS = 18
except:
    pass

# /usr/include/edf_data.h: 343
try:
    MESSAGEEVENT = 24
except:
    pass

# /usr/include/edf_data.h: 348
try:
    BUTTONEVENT = 25
except:
    pass

# /usr/include/edf_data.h: 349
try:
    INPUTEVENT = 28
except:
    pass

# /usr/include/edf_data.h: 351
try:
    LOST_DATA_EVENT = 63
except:
    pass

# /usr/include/edf_data.h: 358
try:
    READ_ENDTIME = 64
except:
    pass

# /usr/include/edf_data.h: 361
try:
    READ_GRES = 512
except:
    pass

# /usr/include/edf_data.h: 362
try:
    READ_SIZE = 128
except:
    pass

# /usr/include/edf_data.h: 363
try:
    READ_VEL = 256
except:
    pass

# /usr/include/edf_data.h: 364
try:
    READ_STATUS = 8192
except:
    pass

# /usr/include/edf_data.h: 366
try:
    READ_BEG = 1
except:
    pass

# /usr/include/edf_data.h: 367
try:
    READ_END = 2
except:
    pass

# /usr/include/edf_data.h: 368
try:
    READ_AVG = 4
except:
    pass

# /usr/include/edf_data.h: 371
try:
    READ_PUPILXY = 1024
except:
    pass

# /usr/include/edf_data.h: 372
try:
    READ_HREFXY = 2048
except:
    pass

# /usr/include/edf_data.h: 373
try:
    READ_GAZEXY = 4096
except:
    pass

# /usr/include/edf_data.h: 375
try:
    READ_BEGPOS = 8
except:
    pass

# /usr/include/edf_data.h: 376
try:
    READ_ENDPOS = 16
except:
    pass

# /usr/include/edf_data.h: 377
try:
    READ_AVGPOS = 32
except:
    pass

# /usr/include/edf_data.h: 381
try:
    FRIGHTEYE_EVENTS = 32768
except:
    pass

# /usr/include/edf_data.h: 382
try:
    FLEFTEYE_EVENTS = 16384
except:
    pass

# /usr/include/edf_data.h: 387
try:
    LEFTEYE_EVENTS = 32768
except:
    pass

# /usr/include/edf_data.h: 388
try:
    RIGHTEYE_EVENTS = 16384
except:
    pass

# /usr/include/edf_data.h: 389
try:
    BLINK_EVENTS = 8192
except:
    pass

# /usr/include/edf_data.h: 390
try:
    FIXATION_EVENTS = 4096
except:
    pass

# /usr/include/edf_data.h: 391
try:
    FIXUPDATE_EVENTS = 2048
except:
    pass

# /usr/include/edf_data.h: 392
try:
    SACCADE_EVENTS = 1024
except:
    pass

# /usr/include/edf_data.h: 393
try:
    MESSAGE_EVENTS = 512
except:
    pass

# /usr/include/edf_data.h: 394
try:
    BUTTON_EVENTS = 64
except:
    pass

# /usr/include/edf_data.h: 395
try:
    INPUT_EVENTS = 32
except:
    pass

# /usr/include/edf_data.h: 401
try:
    EVENT_VELOCITY = 32768
except:
    pass

# /usr/include/edf_data.h: 402
try:
    EVENT_PUPILSIZE = 16384
except:
    pass

# /usr/include/edf_data.h: 403
try:
    EVENT_GAZERES = 8192
except:
    pass

# /usr/include/edf_data.h: 404
try:
    EVENT_STATUS = 4096
except:
    pass

# /usr/include/edf_data.h: 406
try:
    EVENT_GAZEXY = 1024
except:
    pass

# /usr/include/edf_data.h: 407
try:
    EVENT_HREFXY = 512
except:
    pass

# /usr/include/edf_data.h: 408
try:
    EVENT_PUPILXY = 256
except:
    pass

# /usr/include/edf_data.h: 410
try:
    FIX_AVG_ONLY = 8
except:
    pass

# /usr/include/edf_data.h: 411
try:
    START_TIME_ONLY = 4
except:
    pass

# /usr/include/edf_data.h: 413
try:
    PARSEDBY_GAZE = 192
except:
    pass

# /usr/include/edf_data.h: 414
try:
    PARSEDBY_HREF = 128
except:
    pass

# /usr/include/edf_data.h: 415
try:
    PARSEDBY_PUPIL = 64
except:
    pass

# /usr/include/edf_data.h: 420
try:
    LED_TOP_WARNING = 128
except:
    pass

# /usr/include/edf_data.h: 421
try:
    LED_BOT_WARNING = 64
except:
    pass

# /usr/include/edf_data.h: 422
try:
    LED_LEFT_WARNING = 32
except:
    pass

# /usr/include/edf_data.h: 423
try:
    LED_RIGHT_WARNING = 16
except:
    pass

# /usr/include/edf_data.h: 424
try:
    HEAD_POSITION_WARNING = 240
except:
    pass

# /usr/include/edf_data.h: 426
try:
    LED_EXTRA_WARNING = 8
except:
    pass

# /usr/include/edf_data.h: 427
try:
    LED_MISSING_WARNING = 4
except:
    pass

# /usr/include/edf_data.h: 428
try:
    HEAD_VELOCITY_WARNING = 1
except:
    pass

# /usr/include/edf_data.h: 430
try:
    CALIBRATION_AREA_WARNING = 2
except:
    pass

# /usr/include/edf_data.h: 432
try:
    MATH_ERROR_WARNING = 8192
except:
    pass

# /usr/include/edf_data.h: 438
try:
    INTERP_SAMPLE_WARNING = 4096
except:
    pass

# /usr/include/edf_data.h: 444
try:
    INTERP_PUPIL_WARNING = 32768
except:
    pass

# /usr/include/edf_data.h: 447
try:
    CR_WARNING = 3840
except:
    pass

# /usr/include/edf_data.h: 448
try:
    CR_LEFT_WARNING = 1280
except:
    pass

# /usr/include/edf_data.h: 449
try:
    CR_RIGHT_WARNING = 2560
except:
    pass

# /usr/include/edf_data.h: 452
try:
    CR_LOST_WARNING = 768
except:
    pass

# /usr/include/edf_data.h: 453
try:
    CR_LOST_LEFT_WARNING = 256
except:
    pass

# /usr/include/edf_data.h: 454
try:
    CR_LOST_RIGHT_WARNING = 512
except:
    pass

# /usr/include/edf_data.h: 457
try:
    CR_RECOV_WARNING = 3072
except:
    pass

# /usr/include/edf_data.h: 458
try:
    CR_RECOV_LEFT_WARNING = 1024
except:
    pass

# /usr/include/edf_data.h: 459
try:
    CR_RECOV_RIGHT_WARNING = 2048
except:
    pass

# /usr/include/edf_data.h: 465
try:
    TFLAG_MISSING = 16384
except:
    pass

# /usr/include/edf_data.h: 466
try:
    TFLAG_ANGLE = 8192
except:
    pass

# /usr/include/edf_data.h: 467
try:
    TFLAG_NEAREYE = 4096
except:
    pass

# /usr/include/edf_data.h: 469
try:
    TFLAG_CLOSE = 2048
except:
    pass

# /usr/include/edf_data.h: 470
try:
    TFLAG_FAR = 1024
except:
    pass

# /usr/include/edf_data.h: 472
try:
    TFLAG_T_TSIDE = 128
except:
    pass

# /usr/include/edf_data.h: 473
try:
    TFLAG_T_BSIDE = 64
except:
    pass

# /usr/include/edf_data.h: 474
try:
    TFLAG_T_LSIDE = 32
except:
    pass

# /usr/include/edf_data.h: 475
try:
    TFLAG_T_RSIDE = 16
except:
    pass

# /usr/include/edf_data.h: 477
try:
    TFLAG_E_TSIDE = 8
except:
    pass

# /usr/include/edf_data.h: 478
try:
    TFLAG_E_BSIDE = 4
except:
    pass

# /usr/include/edf_data.h: 479
try:
    TFLAG_E_LSIDE = 2
except:
    pass

# /usr/include/edf_data.h: 480
try:
    TFLAG_E_RSIDE = 1
except:
    pass

# /usr/include/edf.h: 57
try:
    NO_PENDING_ITEMS = 0
except:
    pass

# /usr/include/edf.h: 58
try:
    RECORDING_INFO = 30
except:
    pass

# /usr/include/edf.h: 83
try:
    PUPIL_ONLY_250 = 0
except:
    pass

# /usr/include/edf.h: 84
try:
    PUPIL_ONLY_500 = 1
except:
    pass

# /usr/include/edf.h: 85
try:
    PUPIL_CR = 2
except:
    pass

# /usr/include/edf.h: 101
def FLOAT_TIME(x):
    return (((x.contents.time).value) + (((x.contents.flags).value) & SAMPLE_ADD_OFFSET) and 0.5 or 0.0)

_EDFFILE = struct__EDFFILE # /usr/include/edf.h: 94

# No inserted files

