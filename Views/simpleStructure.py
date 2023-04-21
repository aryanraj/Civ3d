from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder
from OCC.Core.AIS import AIS_Shape, AIS_ViewCube
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Ax2, gp_Dir, gp_Vec, gp_Trsf
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TopLoc import TopLoc_Location
from typing import Union
import numpy as np
import numpy.typing as npt
from Models import Node, FixedBeam, Beam

class SimpleView():

  def __init__(self, nodes:list[Node]=[], beamsORfixedbeams:list[Union[Beam, FixedBeam]]=[], ModeShapes:Union[npt.NDArray[np.float64],None]=None):
    self.nodes = nodes
    self.ModeShapes = ModeShapes
    self.nodes_ais:list[AIS_Shape] = []
    self.beams:list[FixedBeam] = []
    self.beams_ais:list[AIS_Shape] = []

    for beamORfixedbeam in beamsORfixedbeams:
      if type(beamORfixedbeam) is Beam:
        self.beams.extend(beamORfixedbeam.childBeams)
      elif type(beamORfixedbeam) is FixedBeam:
        self.beams.append(beamORfixedbeam)

    for node in self.nodes:
      dirMain = gp_Dir(*node.axis[0])
      dirCrossSectionX = gp_Dir(*node.axis[1])
      ax = gp_Ax2(gp_Pnt(*node.coord*1000), dirMain, dirCrossSectionX)
      sphere = BRepPrimAPI_MakeSphere(ax, 150, 0, np.pi/2, np.pi/2)
      sphere.Build()
      sphere_shape = sphere.Shape()
      ais_sphere = AIS_Shape(sphere_shape)
      red = Quantity_Color(0.9, 0.1, 0.1, Quantity_TOC_RGB)
      ais_sphere.SetColor(red)
      self.nodes_ais.append(ais_sphere)

    for beam in self.beams:
      dirMain = gp_Dir(*beam.axis[0])
      dirCrossSectionX = gp_Dir(*beam.axis[1])
      ax = gp_Ax2(gp_Pnt(*beam.i.coord*1000), dirMain, dirCrossSectionX)
      cylinder = BRepPrimAPI_MakeCylinder(ax, 50, beam.L*1000)
      cylinder.Build()
      cylinder_shape = cylinder.Shape()
      ais_cylinder = AIS_Shape(cylinder_shape)
      self.beams_ais.append(ais_cylinder)

    self.display, self.start_display, self.add_menu, self.add_function_to_menu = init_display()
    viewCube = AIS_ViewCube()
    viewCube.SetSize(100)
    self.display.Context.Display(viewCube, False)

  def displayUndeformed(self):
    # self.display.EraseAll()
    for ais in self.nodes_ais:
      self.display.Context.Display(ais, True)
    for ais in self.beams_ais:
      self.display.Context.Display(ais, True)
    self.display.FitAll()

  def displayDeformed(self, nodeDisplacementVector:tuple[npt.NDArray, npt.NDArray], beamDisplacementVector:tuple[npt.NDArray, npt.NDArray], factor:float=1):
    for index, (displacementVector, rotationVector) in enumerate(nodeDisplacementVector):
      self._translateNode(index, displacementVector*factor, rotationVector*factor)
    for index, (iDisplacementVector, jDisplacementVector) in enumerate(beamDisplacementVector):
      self._displaceBeamByijDisplacement(index, iDisplacementVector*factor, jDisplacementVector*factor)
    self.display.Context.UpdateCurrentViewer()
  
  def _translateNode(self, index:int, displacementVector:npt.NDArray[np.float64], rotatationVector:npt.NDArray[np.float64]) -> None:
    node = self.nodes[index]
    node_ais = self.nodes_ais[index]
    nodeTranslateTrsf = gp_Trsf()
    nodeTranslateTrsf.SetTranslation(gp_Vec(*displacementVector))
    nodeRotateTrsf = gp_Trsf()
    if np.linalg.norm(rotatationVector) != 0:
      nodeRotateTrsf.SetRotation(gp_Ax1(gp_Pnt(*node.coord*1000), gp_Dir(*rotatationVector)), np.linalg.norm(rotatationVector)/1000)
    nodeToploc = TopLoc_Location(nodeTranslateTrsf*nodeRotateTrsf)
    self.display.Context.SetLocation(node_ais, nodeToploc)

  def _displaceBeamByijDisplacement(self, index:int, iDisplacementVector:npt.NDArray[np.float64], jDisplacementVector:npt.NDArray[np.float64]):
    beam_ais = self.beams_ais[index]
    beam = self.beams[index]
    beamTranslateTrsf = gp_Trsf()
    beamRotateTrsf = gp_Trsf()
    # Beam Displacement
    beamTranslateTrsf.SetTranslation(gp_Vec(*(iDisplacementVector + jDisplacementVector)/2))
    # Beam Rotation
    relDisplacement = jDisplacementVector - iDisplacementVector
    if np.linalg.norm(relDisplacement) != 0:
      rotationPoint = (beam.i.coord + beam.j.coord)/2*1000
      rotationAxisRaw = np.cross(beam.axis[0], relDisplacement)
      rotationAxis = rotationAxisRaw/np.linalg.norm(rotationAxisRaw)
      perpendicularDisplacementAxis = np.cross(rotationAxis, beam.axis[0])
      rotationAngle = np.dot(relDisplacement, perpendicularDisplacementAxis)/(beam.L*1000)
      beamRotateTrsf.SetRotation(gp_Ax1(gp_Pnt(*rotationPoint), gp_Dir(*rotationAxis)), rotationAngle)

    toploc = TopLoc_Location(beamTranslateTrsf*beamRotateTrsf)
    self.display.Context.SetLocation(beam_ais, toploc)
  
  def displayModeShape(self, modeNumber:int = 0) -> None:
    nodeDisplacementVector: list[npt.NDArray[np.float64]] = []
    for node in self.nodes:
      displacementVector = node.DOF[0].dir * self.ModeShapes[node.DOF[0].id][modeNumber] \
        + node.DOF[1].dir * self.ModeShapes[node.DOF[1].id][modeNumber] \
        + node.DOF[2].dir * self.ModeShapes[node.DOF[2].id][modeNumber]
      rotationVector = node.DOF[3].dir * self.ModeShapes[node.DOF[3].id][modeNumber] \
        + node.DOF[4].dir * self.ModeShapes[node.DOF[4].id][modeNumber] \
        + node.DOF[5].dir * self.ModeShapes[node.DOF[5].id][modeNumber]
      nodeDisplacementVector.append((displacementVector, rotationVector))

    beamDisplacementVector: list[tuple(npt.NDArray[np.float64],npt.NDArray[np.float64])] = []
    for beam in self.beams:
      iDisplacementVector = beam.i.DOF[0].dir * self.ModeShapes[beam.i.DOF[0].id][modeNumber] \
        + beam.i.DOF[1].dir * self.ModeShapes[beam.i.DOF[1].id][modeNumber] \
        + beam.i.DOF[2].dir * self.ModeShapes[beam.i.DOF[2].id][modeNumber]
      jDisplacementVector = beam.j.DOF[0].dir * self.ModeShapes[beam.j.DOF[0].id][modeNumber] \
        + beam.j.DOF[1].dir * self.ModeShapes[beam.j.DOF[1].id][modeNumber] \
        + beam.j.DOF[2].dir * self.ModeShapes[beam.j.DOF[2].id][modeNumber]
      beamDisplacementVector.append((iDisplacementVector, jDisplacementVector))

    import time
    startTimeSeconds = time.time()
    minFPS = 25
    totalDurationSeconds = 4.0
    while (currentTimeSeconds:=time.time()) - startTimeSeconds < totalDurationSeconds:
      phaseDegree = 360*(currentTimeSeconds - startTimeSeconds)/totalDurationSeconds
      displacementFactor = np.sin(np.radians(phaseDegree))*1/max(self.ModeShapes[:, modeNumber])*500*4
      self.displayDeformed(nodeDisplacementVector, beamDisplacementVector, displacementFactor)
      time.sleep(max(1/minFPS - (time.time() - currentTimeSeconds), 0))
    self.displayDeformed(nodeDisplacementVector, beamDisplacementVector, 0)

  def start(self):
    self.add_menu("Mode")
    self.add_function_to_menu("Mode", self.displayUndeformed)
    for i in range(self.ModeShapes.shape[1]):
      _callback = (lambda _:lambda:self.displayModeShape(_))(i)
      _callback.__name__ = f"Mode {i+1}"
      self.add_function_to_menu("Mode", _callback)
    self.displayUndeformed()
    self.start_display()

  @staticmethod
  def display_test():
    # Create a new sphere with a radius of 100
    sphere = BRepPrimAPI_MakeSphere(100)
    sphere.Build()
    
    # Get the shape of the sphere
    sphere_shape = sphere.Shape()
    
    # Create an AIS_Shape object to display the sphere
    ais_sphere = AIS_Shape(sphere_shape)
    
    # Create a display and show the sphere
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.Context.Display(ais_sphere, True)
    display.FitAll()
    start_display()
