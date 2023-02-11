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

