from dataclasses import dataclass, field

@dataclass
class BeamSection:
  Area: float = 1.
  Ixx: float = 0.
  Iyy: float = 1.
  Izz: float = 1.
  E: float = 1.
  mu: float = 0.3

  @property
  def G(self) -> float:
    return self.E/(1+self.mu)/2