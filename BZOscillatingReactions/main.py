# -*- coding: utf-8 -*-
"""
Este programa forma parte de un trabajo de la asignatura de Biofísica, de 4°
de grado en Física. El objetivo de este script es realizar una simulación de
una reacción de Belousov-Zabotinsky, es decir, una reacción oscilante.

Para ello, se toman dos de los modelos más conocidos para este tipo de reacciones,
los modelos de Field-Koros-Noyes y de Györgyi–Field, este último bajo condiciones
de flujo bajo y de flujo alto.

Para la resolución de las ecuaciones diferenciales se emplean métodos de
Runge-Kutta de 4° (modelo FKN) y de 5° orden (modelo GF).
"""

import numpy as np
import matplotlib.pyplot as plt
from src import runge_kutta
from src.models import fkn_model, gf_model

# Resolucion del modelo Field-Koros-Noyes (Oregonator)


# PARAMETROS
e = 0.1
d = 0.0004
q = 0.0008
f = 1.

# CONDICIONES INICIALES
x0 = 0.00013
y0 = 0.2834
z0 = 0.198
t0 = 0.

# PASO DE TIEMPO Y LONGITUD DE LA MUESTRA
deltat = 0.0001
l1=500000

# REALIZAMOS CALCULOS Y REPRESENTAMOS
xl,yl,zl,tl = runge_kutta.rungekutta4(l1,x0,y0,z0,t0,deltat,e,q,d,f)

fig1 = plt.figure(1)
plt.plot(tl,xl*0.0025,label='HBrO2')
plt.xlabel('t (s)')
plt.ylabel('[HBrO2] mol/L')
plt.legend(loc='upper left')

fig2 = plt.figure(2)
plt.plot(tl,yl/100000,label='Br-')
plt.xlabel('t (s)')
plt.ylabel('[Br-] mol/L')
plt.legend(loc='upper left')

fig3 = plt.figure(3)
plt.plot(tl,zl*0.05,label='Ce4+')
plt.xlabel('t (s)')
plt.ylabel('[Ce4+] mol/L')
plt.legend(loc='upper left')

# REPRESENTACION LOGARITMICA
fig4, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(tl, np.log(xl*0.0025),label='HBrO2')
ax2.plot(tl, np.log(yl/100000),label='Br-')
ax3.plot(tl, np.log(zl*0.05),label='Ce4+')
ax1.set(xlabel='t (s)', ylabel='log[HBrO2]')
ax2.set(xlabel='t (s)', ylabel='log[Br-]')
ax3.set(xlabel='t (s)', ylabel='loc[Ce4+]')

# REPRESENTACION 3D
zeros = np.zeros(l1+1)

fig5 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(xl*0.0025,yl/100000,zeros)
ax.plot(xl*0.0025,zeros,zl*0.05)
ax.plot(zeros,yl/100000,zl*0.05)
ax.plot(xl*0.0025,yl/100000,zl*0.05)

ax.set_xlabel('HBrO2')
ax.set_ylabel('$Br-$')
ax.set_zlabel('$Ce4+$')


# RESOLVEMOS AHORA EL MODELO DE GYORGYI-FIELD DE BAJO FLUJO
# kf = 3.9 x 10^-4

A = 0.1
M = 0.25
H = 0.26
C = 0.000833

kgf1 = 4.0*1000000
kgf2 = 2.
kgf3 = 3000.
kgf4 = 55.2
kgf5 = 7000.
kgf6 = 0.09
kgf7 = 0.23

kf = 3.9/10000

alpha = 666.67
beta = 0.3478

T0 = 1/(10*kgf2*A*H*C)
X0 = kgf2*A*H*H/kgf5
Z0 = C*A/40./M
V0 = 4.*A*H*C/M/M
Y0 = 4.*kgf2*A*H*H/kgf5

# CONDICIONES INICIALES
xg,zg,vg = 0.045, 0.9, 0.85
yg = gf_model.fg4(xg,vg,zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)

# REDUCIMOS EL PASO DEL TIEMPO Y AUMENTAMOS EL NUMERO DE VALORES
deltat = 0.00001
l2=20000

# RESOLVEMOS

xgl,vgl,zgl,ygl,tgl = runge_kutta.rungekutta5(xg,vg,zg,yg,t0,deltat,l2,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)

# RREALIZAMOS REPRESENTACIONES

fig4, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(tgl, xgl)
ax2.plot(tgl, zgl)
ax3.plot(tgl, vgl)
ax1.set(ylabel='x')
ax2.set(ylabel='z')
ax3.set(xlabel='$\u03C4$', ylabel='v')

fig5, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(xgl, zgl)
ax2.plot(xgl, vgl)
ax1.set(ylabel='z')
ax2.set(xlabel='x',ylabel='v')

fig4 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(zgl,xgl,vgl)
ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('v')

# RESOLVEMOS AHORA EL MODELO DE GYORGYI-FIELD DE ALTO FLUJO
# kf = 6.18 x 10^-4 (COMPORTAMIENTO OSCILATORIO)

# PARÁMETROS
A = 0.14
M = 0.3
H = 0.26
C = 0.001
alpha = 333.33
beta = 0.2609
kf = 6.18/10000

xg,zg,vg = 0.045, 0.9, 0.85
yg = gf_model.fg4(xg,vg,zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)

xgl,vgl,zgl,ygl,tgl = runge_kutta.rungekutta5(xg,vg,zg,yg,t0,deltat,l2,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)

# REPRESENTAMOS
fig4, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(tgl, xgl)
ax2.plot(tgl, zgl)
ax3.plot(tgl, vgl)
ax1.set(ylabel='x')
ax2.set(ylabel='z')
ax3.set(xlabel='\u03C4 (s)', ylabel='v')

fig5, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(xgl, zgl)
ax2.plot(xgl, vgl)
ax1.set(ylabel='z')
ax2.set(xlabel='x', ylabel='v')

fig6 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(zgl,xgl,vgl)
ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('v')
