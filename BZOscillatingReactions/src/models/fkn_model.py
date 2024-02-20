# -*- coding: utf-8 -*-

# FUNCION XI
def f1(x,y,q,e):
    dxdt = (q*y - x*y + x*(1-x))/e
    return dxdt

# FUNCION ETA
def f2(x,y,z,q,f,d):
    dydt = (-q*y - x*y + f*z)/d
    return dydt

# FUNCION RHO
def f3(x,z):
    dzdt = x-z
    return dzdt