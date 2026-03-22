#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cmath>
#include <limits>
#include <stdexcept>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "travel_time_marcher.h"

#ifndef NPY_ARRAY_CARRAY_RO
#define NPY_ARRAY_CARRAY_RO (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)
#endif

namespace {

PyArrayObject* as_double_array(PyObject* obj, int min_dim, int max_dim, const char* name)
{
  auto* array = reinterpret_cast<PyArrayObject*>(
    PyArray_FROMANY(obj, NPY_DOUBLE, min_dim, max_dim, NPY_ARRAY_CARRAY_RO));
  if (array == nullptr)
  {
    PyErr_Format(PyExc_ValueError, "%s must be a %dD to %dD array of float64", name, min_dim, max_dim);
  }
  return array;
}

bool validate_same_shape(PyArrayObject* lhs, PyArrayObject* rhs)
{
  if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs))
  {
    return false;
  }
  for (int axis = 0; axis < PyArray_NDIM(lhs); ++axis)
  {
    if (PyArray_DIM(lhs, axis) != PyArray_DIM(rhs, axis))
    {
      return false;
    }
  }
  return true;
}

PyObject* travel_time(PyObject*, PyObject* args)
{
  PyObject* pphi = nullptr;
  PyObject* pspeed = nullptr;
  PyObject* pdx = nullptr;
  int order = 2;

  if (!PyArg_ParseTuple(args, "OOO|i", &pphi, &pspeed, &pdx, &order))
  {
    return nullptr;
  }

  if (order != 1 && order != 2)
  {
    PyErr_SetString(PyExc_ValueError, "order must be 1 or 2");
    return nullptr;
  }

  PyArrayObject* phi = as_double_array(pphi, 1, MaximumDimension, "phi");
  if (phi == nullptr)
  {
    return nullptr;
  }

  PyArrayObject* speed = as_double_array(pspeed, 1, MaximumDimension, "speed");
  if (speed == nullptr)
  {
    Py_DECREF(phi);
    return nullptr;
  }

  PyArrayObject* dx = as_double_array(pdx, 1, 1, "dx");
  if (dx == nullptr)
  {
    Py_DECREF(phi);
    Py_DECREF(speed);
    return nullptr;
  }

  if (!validate_same_shape(phi, speed))
  {
    PyErr_SetString(PyExc_ValueError, "phi and speed must have the same shape");
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  if (PyArray_DIM(dx, 0) != PyArray_NDIM(phi))
  {
    PyErr_SetString(PyExc_ValueError, "dx must have length phi.ndim");
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  auto* phi_ptr = static_cast<double*>(PyArray_DATA(phi));
  auto* speed_ptr = static_cast<double*>(PyArray_DATA(speed));
  auto* dx_ptr = static_cast<double*>(PyArray_DATA(dx));

  for (int axis = 0; axis < PyArray_NDIM(phi); ++axis)
  {
    if (!std::isfinite(dx_ptr[axis]) || dx_ptr[axis] <= 0.0)
    {
      PyErr_SetString(PyExc_ValueError, "dx must be finite and strictly positive");
      Py_DECREF(phi);
      Py_DECREF(speed);
      Py_DECREF(dx);
      return nullptr;
    }
  }

  const npy_intp size = PyArray_SIZE(phi);
  for (npy_intp i = 0; i < size; ++i)
  {
    if (!std::isfinite(phi_ptr[i]))
    {
      PyErr_SetString(PyExc_ValueError, "phi must be finite");
      Py_DECREF(phi);
      Py_DECREF(speed);
      Py_DECREF(dx);
      return nullptr;
    }
    if (!std::isfinite(speed_ptr[i]))
    {
      PyErr_SetString(PyExc_ValueError, "speed must be finite");
      Py_DECREF(phi);
      Py_DECREF(speed);
      Py_DECREF(dx);
      return nullptr;
    }
    if (speed_ptr[i] <= 0.0)
    {
      PyErr_SetString(PyExc_ValueError, "speed must be strictly positive");
      Py_DECREF(phi);
      Py_DECREF(speed);
      Py_DECREF(dx);
      return nullptr;
    }
  }

  int shape[MaximumDimension];
  npy_intp shape_npy[MaximumDimension];
  for (int axis = 0; axis < PyArray_NDIM(phi); ++axis)
  {
    shape[axis] = static_cast<int>(PyArray_DIM(phi, axis));
    shape_npy[axis] = PyArray_DIM(phi, axis);
  }

  auto* distance = reinterpret_cast<PyArrayObject*>(
    PyArray_ZEROS(PyArray_NDIM(phi), shape_npy, NPY_DOUBLE, 0));
  auto* flag = reinterpret_cast<PyArrayObject*>(
    PyArray_ZEROS(PyArray_NDIM(phi), shape_npy, NPY_LONGLONG, 0));
  if (distance == nullptr || flag == nullptr)
  {
    Py_XDECREF(distance);
    Py_XDECREF(flag);
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  auto* distance_ptr = static_cast<double*>(PyArray_DATA(distance));
  auto* flag_ptr = static_cast<long long*>(PyArray_DATA(flag));

  travelTimeMarcher marcher(
    phi_ptr,
    dx_ptr,
    flag_ptr,
    distance_ptr,
    PyArray_NDIM(phi),
    shape,
    false,
    order,
    speed_ptr,
    0.0,
    0);

  try
  {
    marcher.march();
  }
  catch (const std::exception& exn)
  {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
    Py_DECREF(distance);
    Py_DECREF(flag);
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  const int error = marcher.getError();
  if (error == 2)
  {
    PyErr_SetString(PyExc_ValueError, "the array phi contains no zero contour (no zero level set)");
    Py_DECREF(distance);
    Py_DECREF(flag);
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }
  if (error != 0)
  {
    PyErr_SetString(PyExc_RuntimeError, "an unknown error occurred in recovar fast marching");
    Py_DECREF(distance);
    Py_DECREF(flag);
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  for (npy_intp i = 0; i < size; ++i)
  {
    if (distance_ptr[i] == maxDouble)
    {
      distance_ptr[i] = std::numeric_limits<double>::infinity();
    }
  }

  Py_DECREF(flag);
  Py_DECREF(phi);
  Py_DECREF(speed);
  Py_DECREF(dx);
  return reinterpret_cast<PyObject*>(distance);
}

PyMethodDef methods[] = {
  {"travel_time", reinterpret_cast<PyCFunction>(travel_time), METH_VARARGS, "Compute fast marching travel time."},
  {nullptr, nullptr, 0, nullptr},
};

#if PY_MAJOR_VERSION >= 3
PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_fast_marching_native",
  "Native fast marching solver for recovar.",
  -1,
  methods,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};
#endif

}  // namespace

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__fast_marching_native(void)
#else
init_fast_marching_native(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  PyObject* module = PyModule_Create(&moduledef);
  if (module == nullptr)
  {
    return nullptr;
  }
  import_array();
  return module;
#else
  PyObject* module = Py_InitModule3("_fast_marching_native", methods, "Native fast marching solver for recovar.");
  if (module == nullptr)
  {
    return;
  }
  import_array();
#endif
}
