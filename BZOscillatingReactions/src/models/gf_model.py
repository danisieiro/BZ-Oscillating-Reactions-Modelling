
def fg1(x,v,z,y,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0):
    dxdt = T0*(-kgf1*H*Y0*x*y + y*(kgf2*A*H*H*Y0/X0) - 2*kgf3*X0*x*x + 0.5*kgf4*A**0.5 * H**1.5 * X0**(-0.5) * (C-Z0*z) * x**0.5 - 0.5*kgf5*Z0*x*z - kf*x)
    return dxdt

def fg2(x,v,z,y,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0):
    dvdt = T0*(x*y*2*kgf1*H*X0*Y0/V0 + y*kgf2*A*H*H*Y0/V0 + x*x*kgf3*X0*X0/V0 - alpha*kgf6*Z0*z*v - kf*v)
    return dvdt

def fg3(x,v,z,y,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0):
    dzdt = T0*(kgf4 * A**0.5 * (H**1.5) * X0**0.5 * (C/Z0 - z) * x**0.5 - kgf5*X0*x*z - alpha*kgf6*V0*z*v - beta*kgf7*M*z - kf*z)
    return dzdt

def fg4(x,v,z,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0):
    dydt = (alpha*kgf6*Z0*V0*z*v/(kgf1*H*X0*x + kgf2*A*H*H + kf))/Y0
    return dydt
