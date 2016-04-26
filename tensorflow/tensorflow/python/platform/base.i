/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Helper macros and typemaps for use in Tensorflow swig files.
//
%{
  #include <memory>
  #include "tensorflow/core/platform/types.h"
  using tensorflow::uint64;
  using tensorflow::string;

  template<class T>
      bool _PyObjAs(PyObject *pystr, T* cstr) {
    T::undefined;  // You need to define specialization _PyObjAs<T>
    return false;
  }

  template<class T>
      PyObject *_PyObjFrom(const T& c) {
    T::undefined;  // You need to define specialization _PyObjFrom<T>
    return NULL;
  }

#ifdef HAS_GLOBAL_STRING
  template<>
      bool _PyObjAs(PyObject *pystr, ::string* cstr) {
    char *buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(pystr, &buf, &len) == -1) return false;
    if (cstr) cstr->assign(buf, len);
    return true;
  }
#endif
  template<>
      bool _PyObjAs(PyObject *pystr, std::string* cstr) {
    char *buf;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(pystr, &buf, &len) == -1) return false;
    if (cstr) cstr->assign(buf, len);
    return true;
  }
#ifdef HAS_GLOBAL_STRING
  template<>
      PyObject* _PyObjFrom(const ::string& c) {
    return PyBytes_FromStringAndSize(c.data(), c.size());
  }
#endif
  template<>
      PyObject* _PyObjFrom(const std::string& c) {
    return PyBytes_FromStringAndSize(c.data(), c.size());
  }

  PyObject* _SwigBytes_FromString(const string& s) {
    return PyBytes_FromStringAndSize(s.data(), s.size());
  }

  // The string must be both ASCII and Unicode compatible, so this routine
  // should be used only for error messages and the like.
  PyObject* _SwigSimpleStr_FromString(const string& s) {
#if PY_MAJOR_VERSION < 3
    return PyString_FromStringAndSize(s.data(), s.size());
#else
    return PyUnicode_FromStringAndSize(s.data(), s.size());
#endif
  }
%}

%typemap(in) string {
  if (!_PyObjAs<string>($input, &$1)) return NULL;
}

%typemap(in) const string& (string temp) {
  if (!_PyObjAs<string>($input, &temp)) return NULL;
  $1 = &temp;
}

%typemap(out) string {
  $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}

%typemap(out) const string& {
  $result = PyBytes_FromStringAndSize($1->data(), $1->size());
}

%typemap(in, numinputs = 0) string* OUTPUT (string temp) {
  $1 = &temp;
}

%typemap(argout) string * OUTPUT {
  PyObject *str = PyBytes_FromStringAndSize($1->data(), $1->length());
  if (!str) SWIG_fail;
  %append_output(str);
}

%typemap(argout) string* INOUT = string* OUTPUT;

%typemap(varout) string {
  $result = PyBytes_FromStringAndSize($1.data(), $1.size());
}

%define _LIST_OUTPUT_TYPEMAP(type, py_converter)
    %typemap(in) std::vector<type>(std::vector<type> temp) {
  if (!vector_input_helper($input, &temp, _PyObjAs<type>)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "sequence(type) expected");
    return NULL;
  }
  $1 = temp;
}
%typemap(in) const std::vector<type>& (std::vector<type> temp),
   const std::vector<type>* (std::vector<type> temp) {
  if (!vector_input_helper($input, &temp, _PyObjAs<type>)) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "sequence(type) expected");
    return NULL;
  }
  $1 = &temp;
}
%typemap(in,numinputs=0)
std::vector<type>* OUTPUT (std::vector<type> temp),
   hash_set<type>* OUTPUT (hash_set<type> temp),
   set<type>* OUTPUT (set<type> temp) {
  $1 = &temp;
}
%typemap(argout) std::vector<type>* OUTPUT, set<type>* OUTPUT, hash_set<type>* OUTPUT {
  %append_output(list_output_helper($1, &py_converter));
}
%typemap(out) std::vector<type> {
  $result = vector_output_helper(&$1, &py_converter);
}
%typemap(out) std::vector<type>*, const std::vector<type>& {
  $result = vector_output_helper($1, &py_converter);
}
%enddef

_LIST_OUTPUT_TYPEMAP(string, _SwigBytes_FromString);
_LIST_OUTPUT_TYPEMAP(unsigned long long, PyLong_FromUnsignedLongLong);

%typemap(in) uint64 {
  // TODO(gps): Check if another implementation
  // from hosting/images/util/image-hosting-utils.swig is better. May be not.
%#if PY_MAJOR_VERSION < 3
  if (PyInt_Check($input)) {
    $1 = static_cast<uint64>(PyInt_AsLong($input));
  } else
%#endif
  if (PyLong_Check($input)) {
    $1 = static_cast<uint64>(PyLong_AsUnsignedLongLong($input));
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "int or long value expected for argument \"$1_name\"");
  }
  // TODO(mrovner): Make consistent use of SWIG_fail vs. return NULL.
  if (PyErr_Occurred()) return NULL;
}

%define _COPY_TYPEMAPS(oldtype, newtype)
    typedef oldtype newtype;
%apply oldtype * OUTPUT { newtype * OUTPUT };
%apply oldtype & OUTPUT { newtype & OUTPUT };
%apply oldtype * INPUT { newtype * INPUT };
%apply oldtype & INPUT { newtype & INPUT };
%apply oldtype * INOUT { newtype * INOUT };
%apply oldtype & INOUT { newtype & INOUT };
%apply std::vector<oldtype> * OUTPUT { std::vector<newtype> * OUTPUT };
%enddef

_COPY_TYPEMAPS(unsigned long long, uint64);

// SWIG macros for explicit API declaration.
// Usage:
//
// %ignoreall
// %unignore SomeName;   // namespace / class / method
// %include "somelib.h"
// %unignoreall  // mandatory closing "bracket"
%define %ignoreall %ignore ""; %enddef
%define %unignore %rename("%s") %enddef
%define %unignoreall %rename("%s") ""; %enddef
