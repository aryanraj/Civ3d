import numpy as np
import numpy.typing as npt

def generateTimeHistoryRecurrence(p:npt.NDArray[np.float64], dt:float, wn:npt.NDArray[np.float64], Zeta:npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    from numpy import exp, sin, cos, sqrt
    wd = wn*sqrt(1-Zeta**2)
    
    m = 1.0
    c = 2*Zeta*wn
    k = wn**2
    
    wndt = wn*dt
    wddt = wd*dt
    Zeta2 = Zeta**2
    _2Zeta = 2*Zeta
    _2Zeta2 = 2*Zeta**2
    eZetawndt = exp(-Zeta*wndt)
    wn2 = wn**2
    sq1_Zeta2 = sqrt(1-Zeta2)
    
    A = eZetawndt*(Zeta/sq1_Zeta2*sin(wddt)+cos(wddt))
    B = eZetawndt*(1/wd*sin(wddt))
    C = 1/k*(
        _2Zeta/wndt
        + eZetawndt*(
            ((1-_2Zeta2)/wddt-Zeta/sq1_Zeta2)*sin(wddt)
            - (1+_2Zeta/wndt)*cos(wddt)
        )
    )
    D = 1/k*(
        1 - _2Zeta/wndt
        + eZetawndt*(
            ((_2Zeta2-1)/wddt)*sin(wddt)
            + (_2Zeta/wndt)*cos(wddt)
        )
    )
    _A = -eZetawndt*(wn/sq1_Zeta2*sin(wddt))
    _B = eZetawndt*(cos(wddt)-Zeta/sq1_Zeta2*sin(wddt))
    _C = 1/k * (
        -1/dt+eZetawndt*(
            (wn/sq1_Zeta2+Zeta/(dt*sq1_Zeta2))*sin(wddt)
            + 1/dt*cos(wddt)
        )
    )
    _D = 1/(k*dt)*(1-eZetawndt*(Zeta/sq1_Zeta2*sin(wddt)+cos(wddt)))

    u = np.zeros(p.shape, dtype=np.float64)
    ud = np.zeros(p.shape, dtype=np.float64)
    udd = np.zeros(p.shape, dtype=np.float64)
    
    udd[:,0] = (p[:,0]-c*ud[:,0]-k*u[:,0])/m
    for i in range(p.shape[1]-1):
        u[:,i+1] = A*u[:,i]+B*ud[:,i]+C*p[:,i]+D*p[:,i+1]
        ud[:,i+1] = _A*u[:,i]+_B*ud[:,i]+_C*p[:,i]+_D*p[:,i+1]
        udd[:,i+1] = (p[:,i+1]-c*ud[:,i+1]-k*u[:,i+1])/m
    
    return u, ud, udd
