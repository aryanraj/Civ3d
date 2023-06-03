from __future__ import annotations
from pathlib import Path
from collections import UserList
import dataclasses
import json
import pandas as pd
import numpy as np
import numpy.typing as npt
import PyOMA as oma

class ModalParameterListEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, ModalParameterList):
            return o.data
        if isinstance(o, ModalParameter):
            return dataclasses.asdict(o)
        return super().default(o)

class ModeShapeDescriptor:
    def __init__(self, *, defaultFactory):
        self._default = defaultFactory()

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default
        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        maxIndex = np.abs(value).argmax()
        setattr(obj, self._name, np.array(value)/value[maxIndex])

@dataclasses.dataclass
class ModalParameter():
    omega:float
    zeta:float
    modeShape:ModeShapeDescriptor = ModeShapeDescriptor(defaultFactory=lambda:np.array([]))
    q0:float = 0.0
    q0dot:float = 0.0

    def generateDisplacement(self, tau:npt.NDArray) -> npt.NDArray[np.float64]:
        omegad = self.omega*(1-self.zeta**2)**0.5
        U = np.exp(-self.zeta*self.omega*np.abs(tau))*(np.cos(omegad*tau)+self.zeta/(1-self.zeta**2)**0.5*np.sin(omegad*tau))
        V = np.exp(-self.zeta*self.omega*np.abs(tau))/omegad*np.sin(omegad*tau)
        qm = (self.q0*U + self.q0dot*V).reshape((1,tau.size))
        return self.modeShape.reshape((self.modeShape.size, 1)) @ qm
    
class ModalParameterList(UserList[ModalParameter]):
    def __init__(self, omegas:list[float], zetas:list[float], displacements:pd.DataFrame, source:str):
        dt = displacements.index[1] - displacements.index[0]
        fps = 1/dt
        frequencyResolution = max(0.01, 4*displacements.index[-1]/fps)
        _, self.FDD = oma.FDDsvp(displacements.to_numpy(), fps, frequencyResolution)
        self.Corr_Matrix = self.getCorr_Matrix(displacements, source)
        self.Ntau = len(self.Corr_Matrix.index)
        self.Nm = len(omegas)
        self.N0 = len(self.Corr_Matrix.columns)
        modeShapes = self.getModeShapeAtOmegas(omegas)
        q0s, q0dots = self.getInitialConditions(omegas, zetas, modeShapes, self.Corr_Matrix)
        super().__init__(ModalParameter(*args) for args in zip(omegas, zetas, modeShapes.T, q0s, q0dots))
        self.setq0q0dots(*self.getInitialConditions(self.omegas, self.zetas, self.modeShapes.to_numpy(), self.Corr_Matrix))
    
    omegas = property(lambda self: [_.omega for _ in self.data])
    zetas = property(lambda self: [_.zeta for _ in self.data])
    modeShapes = property(lambda self: pd.DataFrame(np.array([_.modeShape for _ in self.data]).T, index=self.Corr_Matrix.columns))
    q0s = property(lambda self: [_.q0 for _ in self.data])
    q0dots = property(lambda self: [_.q0dot for _ in self.data])

    def setModeShapes(self, modeShapes:npt.NDArray[np.float64]) -> None:
        for obj, modeShape in zip(self.data, modeShapes.T):
            obj.modeShape = modeShape
    
    def setq0q0dots(self, q0s:list[float], q0dots:list[float]):
        for obj, q0, q0dot in zip(self.data, q0s, q0dots):
            obj.q0 = q0
            obj.q0dot = q0dot

    def optimize(self, nIterations=10):
        from scipy.optimize import fmin
        for _ in range(nIterations):
            for index, obj in enumerate(self.data):
                initialX0 = [obj.omega, obj.zeta]
                parameters = self
                obj.omega, obj.zeta = fmin(self.costFunction, initialX0, (index, parameters))
                self.setModeShapes(self.getModeShapeAtOmegas(self.omegas))
                self.setq0q0dots(*self.getInitialConditions(self.omegas, self.zetas, self.modeShapes.to_numpy(), self.Corr_Matrix))
    
    def getModeShapeAtOmegas(self, omegas:list[float]) -> npt.NDArray[np.float64]:
        Res_FDD = oma.FDDmodEX(np.array(omegas)/2/np.pi, self.FDD, ndf=0)
        modeShapes:npt.NDArray[np.float64] = Res_FDD['Mode Shapes'].real.copy()
        maxIndex:npt.NDArray[np.float64] = np.argmax(np.abs(modeShapes), axis=0)
        for colIndex, maxRowIndex in enumerate(maxIndex.tolist()):
            modeShapes[:,colIndex] /= modeShapes[maxRowIndex, colIndex]
        return modeShapes
    
    def generateDisplacement(self) -> pd.DataFrame:
        tau = self.Corr_Matrix.index.to_numpy()
        accumulatedDisplacement = np.zeros((self.N0, self.Ntau))
        for parameter in self.data:
            accumulatedDisplacement += parameter.generateDisplacement(tau)
        return pd.DataFrame(accumulatedDisplacement.T, index=tau, columns=self.Corr_Matrix.columns)

    @staticmethod
    def costFunction(variableX:npt.NDArray[np.float64], indexX:int, parameterList:ModalParamterList):
        Ntau, Nm, N0 = parameterList.Ntau, parameterList.Nm, parameterList.N0
        omegas = parameterList.omegas
        zetas = parameterList.zetas
        modeShapes = parameterList.modeShapes.to_numpy()
        q0s = parameterList.q0s
        q0dots = parameterList.q0dots
        omegas[indexX] = variableX[0]
        zetas[indexX] = variableX[1]
        
        tau = parameterList.Corr_Matrix.index.to_numpy()
        omegads = [omega*(1-zeta**2)**0.5 for omega, zeta in zip(omegas, zetas)]
        U = np.array([np.exp(-zeta*omega*np.abs(tau))*(np.cos(omegad*tau)+zeta/(1-zeta**2)**0.5*np.sin(omegad*tau)) for omega,omegad,zeta in zip(omegas,omegads,zetas)]).T
        V = np.array([np.exp(-zeta*omega*np.abs(tau))/omegad*np.sin(omegad*tau) for omega,omegad,zeta in zip(omegas,omegads,zetas)]).T
        q0q0dot = np.hstack([q0s, q0dots]).reshape((2*Nm,1))
        
        totalSum = 0
        for i in range(N0):
            si = np.diag(modeShapes[i,:])
            Ai = np.hstack([U@si,V@si])
            Rsi = parameterList.Corr_Matrix.iloc[:,i].to_numpy().reshape((Ntau,1))
            totalSum += np.square(Rsi - Ai @ q0q0dot).sum()
        return totalSum/Ntau/N0

    @staticmethod
    def getInitialConditions(omegas:list[float], zetas:list[float], modeShapes:npt.NDArray[np.float64], Corr_Matrix:pd.DataFrame) -> tuple[list[float], list[float]]:
        tau = Corr_Matrix.index
        Ntau, Nm, N0 = len(Corr_Matrix.index), len(omegas), len(Corr_Matrix.columns)
        omegads = [omega*(1-zeta**2)**0.5 for omega, zeta in zip(omegas, zetas)]
        U = np.array([np.exp(-zeta*omega*tau)*(np.cos(omegad*tau)+zeta/(1-zeta**2)**0.5*np.sin(omegad*tau)) for omega,omegad,zeta in zip(omegas,omegads,zetas)]).T
        V = np.array([np.exp(-zeta*omega*tau)/omegad*np.sin(omegad*tau) for omega,omegad,zeta in zip(omegas,omegads,zetas)]).T
        sumAiRi = np.zeros((2*Nm, 1))
        sumAiAi = np.zeros((2*Nm, 2*Nm))
        for i in range(N0):
            si = np.diag(modeShapes[i,:])
            Ai = np.hstack([U@si,V@si])
            Rsi = Corr_Matrix.iloc[:,i].to_numpy().reshape((Ntau,1))
            sumAiRi += Ai.T @ Rsi
            sumAiAi += Ai.T @ Ai
        res = np.linalg.inv(sumAiAi) @ sumAiRi
        return res[:Nm,0].flatten().tolist(), res[Nm:,0].flatten().tolist()

    @staticmethod
    def getCorr_Matrix(displacements:pd.DataFrame, source:str) -> pd.DataFrame:
        from scipy.signal import correlate, correlation_lags
        dt = displacements.index[1] - displacements.index[0]
        corr = np.array([correlate(displacements[source],displacements[dest], mode='full') for dest in displacements.columns])
        lags = correlation_lags(displacements.shape[0],displacements.shape[0], mode='full')*dt
        return pd.DataFrame(corr.T, index=lags, columns=displacements.columns)