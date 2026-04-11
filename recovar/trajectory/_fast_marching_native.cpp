#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <array>
#include <utility>
#include <vector>

#include "Python.h"
#include "numpy/arrayobject.h"

#ifndef NPY_ARRAY_CARRAY_RO
#define NPY_ARRAY_CARRAY_RO (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)
#endif

namespace {

constexpr int kMaxDimension = 12;
constexpr std::int8_t kFar = 0;
constexpr std::int8_t kNarrow = 1;
constexpr std::int8_t kFrozen = 2;
constexpr std::int8_t kMask = 3;

constexpr double kDoubleEps = std::numeric_limits<double>::epsilon();
constexpr double kMaxDouble = std::numeric_limits<double>::max();
constexpr double kSecondOrderScale = 9.0 / 4.0;
constexpr double kOneThird = 1.0 / 3.0;

struct HeapEntry
{
  double key;
  npy_intp index;

  bool operator>(const HeapEntry& other) const
  {
    if (key != other.key)
    {
      return key > other.key;
    }
    return index > other.index;
  }
};

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

int count_bits(unsigned int mask)
{
  int count = 0;
  while (mask != 0U)
  {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

class TravelTimeMarcher
{
public:
  TravelTimeMarcher(
    const double* phi,
    const double* speed,
    const double* dx,
    int ndim,
    const npy_intp* shape,
    int order,
    double* distance)
    : phi_(phi),
      speed_(speed),
      ndim_(ndim),
      size_(1),
      order_(order),
      shape_(ndim),
      shift_(ndim),
      dx_(ndim),
      idx2_(ndim),
      flags_(),
      distance_(distance)
  {
    for (int axis = 0; axis < ndim_; ++axis)
    {
      shape_[axis] = shape[axis];
      size_ *= shape_[axis];
      dx_[axis] = dx[axis];
      idx2_[axis] = 1.0 / (dx_[axis] * dx_[axis]);
    }

    for (int axis = 0; axis < ndim_; ++axis)
    {
      npy_intp prod = 1;
      for (int inner = axis + 1; inner < ndim_; ++inner)
      {
        prod *= shape_[inner];
      }
      shift_[axis] = prod;
    }

    flags_.assign(static_cast<std::size_t>(size_), kFar);
    for (npy_intp i = 0; i < size_; ++i)
    {
      distance_[i] = 0.0;
      if (speed_[i] < kDoubleEps)
      {
        flags_[static_cast<std::size_t>(i)] = kMask;
      }
    }
  }

  void march()
  {
    initialize_frozen();
    if (!has_frozen_)
    {
      throw std::invalid_argument("the array phi contains no zero contour (no zero level set)");
    }
    initialize_narrow();
    solve();

    for (npy_intp i = 0; i < size_; ++i)
    {
      if (flags_[static_cast<std::size_t>(i)] != kFrozen)
      {
        distance_[i] = std::numeric_limits<double>::infinity();
      }
    }
  }

private:
  void initialize_frozen()
  {
    for (npy_intp i = 0; i < size_; ++i)
    {
      if (flags_[static_cast<std::size_t>(i)] != kMask && phi_[i] == 0.0)
      {
        flags_[static_cast<std::size_t>(i)] = kFrozen;
        distance_[i] = 0.0;
        has_frozen_ = true;
      }
    }

    for (npy_intp i = 0; i < size_; ++i)
    {
      if (flags_[static_cast<std::size_t>(i)] != kFar)
      {
        continue;
      }

      std::array<double, kMaxDimension> local_distance{};
      bool borders_zero_level_set = false;

      for (int axis = 0; axis < ndim_; ++axis)
      {
        for (int direction : {-1, 1})
        {
          const npy_intp neighbor = get_neighbor(i, axis, direction, kMask);
          if (neighbor == -1)
          {
            continue;
          }
          if (phi_[i] * phi_[neighbor] >= 0.0)
          {
            continue;
          }

          borders_zero_level_set = true;
          const double dist_to_interface =
            dx_[axis] * phi_[i] / (phi_[i] - phi_[neighbor]);
          if (local_distance[axis] == 0.0 || local_distance[axis] > dist_to_interface)
          {
            local_distance[axis] = dist_to_interface;
          }
        }
      }

      if (!borders_zero_level_set)
      {
        continue;
      }

      double dsum = 0.0;
      for (int axis = 0; axis < ndim_; ++axis)
      {
        if (local_distance[axis] > 0.0)
        {
          dsum += 1.0 / (local_distance[axis] * local_distance[axis]);
        }
      }

      const double dist = std::sqrt(1.0 / dsum);
      distance_[i] = phi_[i] < 0.0 ? -dist : dist;
      flags_[static_cast<std::size_t>(i)] = kFrozen;
      has_frozen_ = true;
    }

    for (npy_intp i = 0; i < size_; ++i)
    {
      if (flags_[static_cast<std::size_t>(i)] == kFrozen)
      {
        distance_[i] = std::abs(distance_[i] / speed_[i]);
      }
    }
  }

  void initialize_narrow()
  {
    for (npy_intp i = 0; i < size_; ++i)
    {
      if (flags_[static_cast<std::size_t>(i)] != kFar)
      {
        continue;
      }

      bool found_frozen_neighbor = false;
      for (int axis = 0; axis < ndim_ && !found_frozen_neighbor; ++axis)
      {
        for (int direction : {-1, 1})
        {
          const npy_intp neighbor = get_neighbor(i, axis, direction, kMask);
          if (neighbor != -1 && flags_[static_cast<std::size_t>(neighbor)] == kFrozen)
          {
            found_frozen_neighbor = true;
            break;
          }
        }
      }

      if (!found_frozen_neighbor)
      {
        continue;
      }

      flags_[static_cast<std::size_t>(i)] = kNarrow;
      distance_[i] = update_point(i);
      push(i);
    }
  }

  void solve()
  {
    double value = 0.0;
    npy_intp address = -1;
    while (pop_valid(&value, &address))
    {
      tied_indices_.clear();
      tied_indices_.push_back(address);
      flags_[static_cast<std::size_t>(address)] = kFrozen;

      double tied_value = 0.0;
      npy_intp tied_address = -1;
      while (peek_valid(&tied_value, &tied_address) && tied_value == value)
      {
        pop_valid(&tied_value, &tied_address);
        flags_[static_cast<std::size_t>(tied_address)] = kFrozen;
        tied_indices_.push_back(tied_address);
      }

      for (npy_intp frozen_addr : tied_indices_)
      {
        for (int axis = 0; axis < ndim_; ++axis)
        {
          for (int direction : {-1, 1})
          {
            const npy_intp neighbor = get_neighbor(frozen_addr, axis, direction, kFrozen);
            if (neighbor != -1 && flags_[static_cast<std::size_t>(neighbor)] != kFrozen)
            {
              const double updated_distance = update_point(neighbor);
              if (updated_distance != 0.0)
              {
                distance_[neighbor] = updated_distance;
                if (flags_[static_cast<std::size_t>(neighbor)] == kFar)
                {
                  flags_[static_cast<std::size_t>(neighbor)] = kNarrow;
                }
                push(neighbor);
              }
            }

            if (order_ != 2)
            {
              continue;
            }

            const npy_intp local_neighbor = get_neighbor(frozen_addr, axis, direction, kMask);
            if (local_neighbor == -1 || flags_[static_cast<std::size_t>(local_neighbor)] != kFrozen)
            {
              continue;
            }

            const npy_intp second_order_neighbor =
              get_neighbor(frozen_addr, axis, direction * 2, kFrozen);
            if (
              second_order_neighbor == -1 ||
              flags_[static_cast<std::size_t>(second_order_neighbor)] != kNarrow)
            {
              continue;
            }

            const double updated_distance = update_point(second_order_neighbor);
            if (updated_distance != 0.0)
            {
              distance_[second_order_neighbor] = updated_distance;
              push(second_order_neighbor);
            }
          }
        }
      }
    }
  }

  double update_point(npy_intp index)
  {
    if (order_ == 2)
    {
      return update_point_order_two(index, 0U);
    }
    return update_point_order_one(index);
  }

  double update_point_order_one(npy_intp index)
  {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;

    for (int axis = 0; axis < ndim_; ++axis)
    {
      double value = kMaxDouble;
      for (int direction : {-1, 1})
      {
        const npy_intp neighbor = get_neighbor(index, axis, direction, kMask);
        if (
          neighbor != -1 &&
          flags_[static_cast<std::size_t>(neighbor)] == kFrozen &&
          std::abs(distance_[neighbor]) < std::abs(value))
        {
          value = distance_[neighbor];
        }
      }

      if (value < kMaxDouble)
      {
        a += idx2_[axis];
        b -= idx2_[axis] * 2.0 * value;
        c += idx2_[axis] * value * value;
      }
    }

    try
    {
      return solve_quadratic(index, a, b, c);
    }
    catch (const std::runtime_error&)
    {
      return -b / (2.0 * a);
    }
  }

  double update_point_order_two(npy_intp index, unsigned int avoid_mask)
  {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;

    for (int axis = 0; axis < ndim_; ++axis)
    {
      if ((avoid_mask & (1U << axis)) != 0U)
      {
        continue;
      }

      double value1 = kMaxDouble;
      double value2 = kMaxDouble;

      for (int direction : {-1, 1})
      {
        const npy_intp neighbor = get_neighbor(index, axis, direction, kMask);
        if (neighbor == -1 || flags_[static_cast<std::size_t>(neighbor)] != kFrozen)
        {
          continue;
        }
        if (std::abs(distance_[neighbor]) >= std::abs(value1))
        {
          continue;
        }

        value1 = distance_[neighbor];
        const npy_intp neighbor2 = get_neighbor(index, axis, direction * 2, kMask);
        if (
          neighbor2 != -1 &&
          flags_[static_cast<std::size_t>(neighbor2)] == kFrozen &&
          ((distance_[neighbor2] <= value1 && value1 >= 0.0) ||
           (distance_[neighbor2] >= value1 && value1 <= 0.0)))
        {
          value2 = distance_[neighbor2];
          if (phi_[neighbor2] * phi_[neighbor] < 0.0 || phi_[neighbor2] * phi_[index] < 0.0)
          {
            value2 *= -1.0;
          }
        }
      }

      if (value2 < kMaxDouble)
      {
        const double tp = kOneThird * (4.0 * value1 - value2);
        a += idx2_[axis] * kSecondOrderScale;
        b -= idx2_[axis] * 2.0 * kSecondOrderScale * tp;
        c += idx2_[axis] * kSecondOrderScale * tp * tp;
      }
      else if (value1 < kMaxDouble)
      {
        a += idx2_[axis];
        b -= idx2_[axis] * 2.0 * value1;
        c += idx2_[axis] * value1 * value1;
      }
    }

    try
    {
      return solve_quadratic(index, a, b, c);
    }
    catch (const std::runtime_error&)
    {
      if (count_bits(avoid_mask) == ndim_)
      {
        return std::numeric_limits<double>::infinity();
      }

      double best = std::numeric_limits<double>::infinity();
      bool has_solution = false;
      for (int axis = 0; axis < ndim_; ++axis)
      {
        if ((avoid_mask & (1U << axis)) != 0U)
        {
          continue;
        }
        const double candidate = update_point_order_two(index, avoid_mask | (1U << axis));
        if (!has_solution || candidate < best)
        {
          best = candidate;
          has_solution = true;
        }
      }
      if (!has_solution)
      {
        return std::numeric_limits<double>::infinity();
      }
      return best;
    }
  }

  double solve_quadratic(npy_intp index, double a, double b, double c) const
  {
    c -= 1.0 / (speed_[index] * speed_[index]);
    const double determinant = b * b - 4.0 * a * c;
    if (determinant < 0.0)
    {
      throw std::runtime_error("Negative discriminant in travel-time quadratic");
    }
    return (-b + std::sqrt(determinant)) / (2.0 * a);
  }

  void push(npy_intp index)
  {
    heap_.push(HeapEntry{std::abs(distance_[index]), index});
  }

  bool peek_valid(double* key, npy_intp* index)
  {
    while (!heap_.empty())
    {
      const HeapEntry& entry = heap_.top();
      if (
        flags_[static_cast<std::size_t>(entry.index)] == kNarrow &&
        entry.key == std::abs(distance_[entry.index]))
      {
        *key = entry.key;
        *index = entry.index;
        return true;
      }
      heap_.pop();
    }
    return false;
  }

  bool pop_valid(double* key, npy_intp* index)
  {
    while (!heap_.empty())
    {
      HeapEntry entry = heap_.top();
      heap_.pop();
      if (
        flags_[static_cast<std::size_t>(entry.index)] == kNarrow &&
        entry.key == std::abs(distance_[entry.index]))
      {
        *key = entry.key;
        *index = entry.index;
        return true;
      }
    }
    return false;
  }

  npy_intp get_neighbor(npy_intp index, int axis, int direction, std::int8_t blocked_flag) const
  {
    const npy_intp coord = (index / shift_[axis]) % shape_[axis];
    const npy_intp new_coord = coord + static_cast<npy_intp>(direction);
    if (new_coord >= shape_[axis] || new_coord < 0)
    {
      return -1;
    }

    const npy_intp neighbor = index + static_cast<npy_intp>(direction) * shift_[axis];
    if (flags_[static_cast<std::size_t>(neighbor)] == blocked_flag)
    {
      return -1;
    }
    return neighbor;
  }

  const double* phi_;
  const double* speed_;
  int ndim_;
  npy_intp size_;
  int order_;
  std::vector<npy_intp> shape_;
  std::vector<npy_intp> shift_;
  std::vector<double> dx_;
  std::vector<double> idx2_;
  std::vector<std::int8_t> flags_;
  std::vector<npy_intp> tied_indices_;
  double* distance_;
  bool has_frozen_ = false;
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap_;
};

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

  PyArrayObject* phi = as_double_array(pphi, 1, kMaxDimension, "phi");
  if (phi == nullptr)
  {
    return nullptr;
  }

  PyArrayObject* speed = as_double_array(pspeed, 1, kMaxDimension, "speed");
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

  const int ndim = PyArray_NDIM(phi);
  if (PyArray_DIM(dx, 0) != ndim)
  {
    PyErr_SetString(PyExc_ValueError, "dx must have length phi.ndim");
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  const auto* phi_ptr = static_cast<const double*>(PyArray_DATA(phi));
  const auto* speed_ptr = static_cast<const double*>(PyArray_DATA(speed));
  const auto* dx_ptr = static_cast<const double*>(PyArray_DATA(dx));
  const npy_intp size = PyArray_SIZE(phi);

  for (int axis = 0; axis < ndim; ++axis)
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

  auto* distance = reinterpret_cast<PyArrayObject*>(
    PyArray_ZEROS(ndim, PyArray_DIMS(phi), NPY_DOUBLE, 0));
  if (distance == nullptr)
  {
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

  auto* distance_ptr = static_cast<double*>(PyArray_DATA(distance));

  try
  {
    TravelTimeMarcher marcher(
      phi_ptr,
      speed_ptr,
      dx_ptr,
      ndim,
      PyArray_DIMS(phi),
      order,
      distance_ptr);
    marcher.march();
  }
  catch (const std::invalid_argument& exn)
  {
    PyErr_SetString(PyExc_ValueError, exn.what());
    Py_DECREF(distance);
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }
  catch (const std::exception& exn)
  {
    PyErr_SetString(PyExc_RuntimeError, exn.what());
    Py_DECREF(distance);
    Py_DECREF(phi);
    Py_DECREF(speed);
    Py_DECREF(dx);
    return nullptr;
  }

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
