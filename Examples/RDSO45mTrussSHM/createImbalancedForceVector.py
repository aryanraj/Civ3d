import os
nLoadCases = 150
os.environ["MAX_LOADCASE"] = str(nLoadCases)

from Examples.RDSO45mTruss.Structure3D import Structure3D
from Models import DOFClass, Beam
from pathlib import Path
import time

startTime = time.time()

vehicleAxleLoadDistance:list[tuple[float,float]] = [
  (191295, 0.0),
  (191295, 1.9),
  (191295, 3.8),
  (191295, 10.39),
  (191295, 12.29),
  (191295, 14.19),
  (199192.05, 17.923),
  (199192.05, 19.923),
  (199192.05, 25.393),
  (199192.05, 27.393),
  (199388.25, 29.923),
  (199388.25, 31.923),
  (199388.25, 37.393),
  (199388.25, 39.393),
  (199339.2, 41.923),
  (199339.2, 43.923),
  (199339.2, 49.393),
  (199339.2, 51.393),
  (199388.25, 53.923),
  (199388.25, 55.923),
  (199388.25, 61.393),
  (199388.25, 63.393),
  (199314.675, 65.923),
  (199314.675, 67.923),
  (199314.675, 73.393),
  (199314.675, 75.393),
  (199167.525, 77.923),
  (199167.525, 79.923),
  (199167.525, 85.393),
  (199167.525, 87.393),
]

print("**** Defining the structure ****")
structure = Structure3D()
stringerMainList1 = [_.main[0] for _ in structure.stringers]
stringerMainList2 = [_.main[1] for _ in structure.stringers]
print(f"Execution Time = {time.time() - startTime:.2f} secs")

print("**** Adding Load Cases ****")
for loadCase in range(nLoadCases):
  # Starting distance from -10 and incrementing till 140 distance
  startingDistance = -10 + loadCase*150/nLoadCases
  for axleLoad, distance in vehicleAxleLoadDistance:
    Beam.addPointLoadToBeamList(stringerMainList1, 2, -axleLoad/2, startingDistance - distance, [loadCase])
    Beam.addPointLoadToBeamList(stringerMainList2, 2, -axleLoad/2, startingDistance - distance, [loadCase])
print(f"Execution Time = {time.time() - startTime:.2f} secs")

print("**** Adding Load Cases ****")
from scipy.sparse import save_npz
save_npz(Path(__file__).parent.joinpath(f'DataExchange/.ImbalancedActionVector{nLoadCases}.npz'), DOFClass.ImbalancedActionVector.tocsc())
print(f"Execution Time = {time.time() - startTime:.2f} secs")