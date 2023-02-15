import numpy as np
import numpy.typing as npt

def getAxisFromTwoNodesAndBeta(coordi: npt.NDArray[np.float64], coordj: npt.NDArray[np.float64], beta: float) -> npt.NDArray[np.float64]:
  xij = (coordj - coordi)/np.linalg.norm(coordj - coordi)
  yij = computeLocalYAxis(xij, beta)
  zij = np.cross(xij, yij)
  return np.array([xij, yij, zij])

def computeLocalYAxis(dirX: npt.NDArray[np.float64], beta: float) -> npt.NDArray[np.float64]:
  """
  Using Rodrigues' rotation formula for performing a rotation for beta angle
  For more details goto https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
  """
  unrotated = np.array([-dirX[1], dirX[0], 0])
  K = computePreCrossProductTransform(dirX)
  R: npt.NDArray[np.float64] = np.identity(3) + np.sin(beta/(2*np.pi))*K + (1 - np.cos(beta/(2*np.pi))) * K @ K
  return unrotated @ R.T

def computePreCrossProductTransform(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

def globalToLocalBasisChangeMatrix(globalAxis:npt.NDArray[np.float64], localAxis:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  return localAxis @ globalAxis.T

def ensure1DNumpyArray(arr, dtype, default=[]):
  import collections.abc
  if isinstance(arr, collections.abc.Sequence):
    return np.array(arr, dtype=dtype)
  if isinstance(arr, np.ndarray):
    if arr.ndim == 1:
      return arr
    else:
      raise Exception(f"Unable to reduce {arr.ndim}-dimensional array to 1D")
  return np.array(default, dtype=dtype)

def ensure2DVectorNumpyArray(arr, dtype, default=[[]]):
  import collections.abc
  if isinstance(arr, collections.abc.Sequence):
    return np.array(arr, dtype=dtype)
  if isinstance(arr, np.ndarray):
    if arr.ndim == 1:
      return np.array([arr], dtype=dtype).T
    elif arr.ndim == 2 and arr.shape[1] == 1:
      return arr
    elif arr.ndim == 2 and arr.shape[0] == 1:
      return arr.T
    else:
      raise Exception(f"Unable to reduce {arr.ndim}D array of shape {arr.shape} to 2D Vector")
  return np.array(default, dtype=dtype)

def ensure2DSquareNumpyArray(arr, dtype, default=[[]]):
  import collections.abc
  if isinstance(arr, collections.abc.Sequence):
    return np.array(arr, dtype=dtype)
  if isinstance(arr, np.ndarray):
    if arr.ndim == 1:
      return np.diag(arr, dtype=dtype)
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
      return arr
    else:
      raise Exception(f"Unable to reduce {arr.ndim}D array of shape {arr.shape} to 2D Square Matrix")
  return np.array(default, dtype=dtype)