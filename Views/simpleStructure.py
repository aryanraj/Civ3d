from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCylinder
from OCC.Core.AIS import AIS_Shape
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from typing import Union
from Models import Node, FixedBeam, Beam

class SimpleView():
  def display(self, nodes:list[Node] = [], beamsORfixedbeams:list[Union[Beam, FixedBeam]] = []):
    beams:list[FixedBeam] = []
    if len(beamsORfixedbeams) != 0 and type(beamsORfixedbeams[0]) is Beam:
      for _ in beamsORfixedbeams:
        beams.extend(_.childBeams)
    elif len(beamsORfixedbeams) != 0 and type(beamsORfixedbeams[0]) is FixedBeam:
      beams.extend(beamsORfixedbeams)
    nodes_ais = []
    for node in nodes:
      # Create sphere with a radius of 100
      sphere = BRepPrimAPI_MakeSphere(gp_Pnt(*node.coord*1000), 100)
      sphere.Build()
    
      # Get the shape of the sphere
      sphere_shape = sphere.Shape()
    
      # Create an AIS_Shape object to display the sphere
      ais_sphere = AIS_Shape(sphere_shape)
      red = Quantity_Color(0.9, 0.1, 0.1, Quantity_TOC_RGB)
      ais_sphere.SetColor(red)

      # Appending the AIS_Shape object to the list
      nodes_ais.append(ais_sphere)

    beams_ais = []
    for beam in beams:
      dir = gp_Dir(gp_Vec(gp_Pnt(*beam.i.coord*1000), gp_Pnt(*beam.j.coord*1000)))
      ax = gp_Ax2(gp_Pnt(*beam.i.coord*1000), dir)
      cylinder = BRepPrimAPI_MakeCylinder(ax, 50, beam.L*1000)
      cylinder.Build()
      cylinder_shape = cylinder.Shape()
      ais_cylinder = AIS_Shape(cylinder_shape)
      beams_ais.append(ais_cylinder)

    # Create a display and show the sphere
    display, start_display, add_menu, add_function_to_menu = init_display()
    for ais in nodes_ais:
      display.Context.Display(ais, True)
    for ais in beams_ais:
      display.Context.Display(ais, True)
    display.FitAll()
    start_display()

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

