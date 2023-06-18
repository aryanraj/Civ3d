import os
nLoadCases = 4500
os.environ["MAX_LOADCASE"] = str(nLoadCases)

from Examples.RDSO45mTruss.Structure3D import Structure3D
from Models import DOFClass, Beam
from pathlib import Path
import time

startTime = time.time()

vehicleAxleLoadDistance:list[tuple[float,float]] = [
  (166702, 0),
  (166702, 2.7),
  (166702, 14.9),
  (166702, 17.6),
  (166702, 24),
  (166702, 26.7),
  (166702, 38.9),
  (166702, 41.6),
  (166702, 48),
  (166702, 50.7),
  (166702, 62.9),
  (166702, 65.6),
  (166702, 72),
  (166702, 74.7),
  (166702, 86.9),
  (166702, 89.6),
  (166702, 96),
  (166702, 98.7),
  (166702, 110.9),
  (166702, 113.6),
  (166702, 120),
  (166702, 122.7),
  (166702, 134.9),
  (166702, 137.6),
  (166702, 144),
  (166702, 146.7),
  (166702, 158.9),
  (166702, 161.6),
  (166702, 168),
  (166702, 170.7),
  (166702, 182.9),
  (166702, 185.6),
  (166702, 192),
  (166702, 194.7),
  (166702, 206.9),
  (166702, 209.6),
  (166702, 216),
  (166702, 218.7),
  (166702, 230.9),
  (166702, 233.6),
  (166702, 240),
  (166702, 242.7),
  (166702, 254.9),
  (166702, 257.6),
  (166702, 264),
  (166702, 266.7),
  (166702, 278.9),
  (166702, 281.6),
  (166702, 288),
  (166702, 290.7),
  (166702, 302.9),
  (166702, 305.6),
  (166702, 312),
  (166702, 314.7),
  (166702, 326.9),
  (166702, 329.6),
  (166702, 336),
  (166702, 338.7),
  (166702, 350.9),
  (166702, 353.6),
  (166702, 360),
  (166702, 362.7),
  (166702, 374.9),
  (166702, 377.6),
]

print("**** Defining the structure ****")
structure = Structure3D()
stringerMainList1 = [_.main[0] for _ in structure.stringers]
stringerMainList2 = [_.main[1] for _ in structure.stringers]
print(f"Execution Time = {time.time() - startTime:.2f} secs")

print("**** Adding Load Cases ****")
startingDistance = -10
increment = 450/nLoadCases
loadCases = list(range(nLoadCases))

loadList = [-_[0]/2 for _ in vehicleAxleLoadDistance]
loadStartingDistList = [startingDistance - _[1] for _ in vehicleAxleLoadDistance]
Beam.addMovingPointLoadsToBeamList(stringerMainList1, 2, loadList, loadStartingDistList, increment, loadCases)
Beam.addMovingPointLoadsToBeamList(stringerMainList2, 2, loadList, loadStartingDistList, increment, loadCases)
print(f"Execution Time = {time.time() - startTime:.2f} secs")

print("**** Saving Load Cases ****")
from scipy.sparse import save_npz
save_npz(Path(__file__).parent.joinpath(f'DataExchange/.ImbalancedActionVector{nLoadCases}.npz'), DOFClass.ImbalancedActionVector.tocsc())
print(f"Execution Time = {time.time() - startTime:.2f} secs")
input("Press Enter to continue...")