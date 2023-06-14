from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class BeamSection:
  Area: float = 1.
  Ixx: float = 0.
  Iyy: float = 1.
  Izz: float = 1.
  E: float = 1.
  mu: float = 0.3
  rho: float = 1
  rhog: float = 9.806

  @property
  def G(self) -> float:
    return self.E/(1+self.mu)/2

  @classmethod
  def Rectangle(cls, b:float, d:float, E:float, mu:float, rho:float, rhog:float) -> BeamSection:
    A = b*d
    Iyy = b*d**3/12
    Izz = d*b**3/12
    Ixx = Iyy+Izz
    return cls(A, Ixx, Iyy, Izz, E, mu, rho, rhog)