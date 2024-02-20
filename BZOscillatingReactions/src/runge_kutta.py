
import numpy as np
from src.models import fkn_model, gf_model

def rungekutta4(l,x,y,z,t,deltat,e,q,d,f,f1=fkn_model.f1,f2=fkn_model.f2,f3=fkn_model.f3):
    
    # CREO LOS ARRAYS DE X,Y,Z,T y AÑADO CONDICIONES INICIALES
    xlist=np.zeros(l+1)
    ylist=np.zeros(l+1)
    zlist=np.zeros(l+1)
    tlist=np.zeros(l+1)

    
    xlist[0]=x
    ylist[0]=y
    zlist[0]=z
    tlist[0]=t
        
    # CALCULOS A CADA PASO DE TIEMPO dt    
    for i in range(l):
        k1x=deltat*f1(x,y,q,e)
        k1y=deltat*f2(x,y,z,q,f,d)
        k1z=deltat*f3(x,z)
        
        k2x=deltat*f1(x+0.5*k1x,y+0.5*k1y,q,e)
        k2y=deltat*f2(x+0.5*k1x,y+0.5*k1y,z+0.5*k1z,q,f,d)
        k2z=deltat*f3(x+0.5*k1x,z+0.5*k1z)

        k3x=deltat*f1(x+0.5*k2x,y+0.5*k2y,q,e)
        k3y=deltat*f2(x+0.5*k2x,y+0.5*k2y,z+0.5*k2z,q,f,d)
        k3z=deltat*f3(x+0.5*k2x,z+0.5*k2z)
  
        k4x=deltat*f1(x+k3x,y+k3y,q,e)
        k4y=deltat*f2(x+k3x,y+k3y,z+k3z,q,f,d)
        k4z=deltat*f3(x+k3x,z+k3z)
        
        # VARIABLES A TIEMPO t[i] + deltat
        t=t+deltat
        x = x + k1x/6. + k2x/3. + k3x/3. + k4x/6.
        y = y + k1y/6. + k2y/3. + k3y/3. + k4y/6.
        z = z + k1z/6. + k2z/3. + k3z/3. + k4z/6.
        
        xlist[i+1]=x
        ylist[i+1]=y
        zlist[i+1]=z
        tlist[i+1]=t
        
    return xlist,ylist,zlist,tlist

def rungekutta5(xg,vg,zg,yg,tg,deltat,l,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0,fg1=gf_model.fg1,fg2=gf_model.fg2,fg3=gf_model.fg3,fg4=gf_model.fg4):
    
    xglist,vglist,zglist,yglist,tglist = np.zeros(l+1),np.zeros(l+1),np.zeros(l+1),np.zeros(l+1),np.zeros(l+1)
    
    xglist[0],vglist[0],zglist[0],yglist[0],tglist[0] = xg,vg,zg,yg,tg
    
    for i in range(l):
        
        k1xg = deltat*fg1(xg,vg,zg,yg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k1vg = deltat*fg2(xg,vg,zg,yg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k1zg = deltat*fg3(xg,vg,zg,yg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        
        k2xg = deltat*fg1(xg+0.25*k1xg,vg+0.25*k1vg,zg+0.25*k1zg,yg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k2vg = deltat*fg2(xg+0.25*k1xg,vg+0.25*k1vg,zg+0.25*k1zg,yg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k2zg = deltat*fg3(xg+0.25*k1xg,vg+0.25*k1vg,zg+0.25*k1zg,yg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        y2g = fg4(xg+0.25*k1xg,vg+0.25*k1vg,zg+0.25*k1zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        
        k3xg = deltat*fg1(xg+0.125*k1xg+0.125*k2xg,vg+0.125*k1vg+0.125*k2vg,zg+0.125*k1zg+0.125*k2zg,y2g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k3vg = deltat*fg2(xg+0.125*k1xg+0.125*k2xg,vg+0.125*k1vg+0.125*k2vg,zg+0.125*k1zg+0.125*k2zg,y2g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k3zg = deltat*fg3(xg+0.125*k1xg+0.125*k2xg,vg+0.125*k1vg+0.125*k2vg,zg+0.125*k1zg+0.125*k2zg,y2g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        y3g = fg4(xg+0.125*k1xg+0.125*k2xg,vg+0.125*k1vg+0.125*k2vg,zg+0.125*k1zg+0.125*k2zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        
        k4xg = deltat*fg1(xg-0.5*k2xg+k3xg,vg-0.5*k2vg+k3vg,zg-0.5*k2zg+k3zg,y3g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k4vg = deltat*fg2(xg-0.5*k2xg+k3xg,vg-0.5*k2vg+k3vg,zg-0.5*k2zg+k3zg,y3g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k4zg = deltat*fg3(xg-0.5*k2xg+k3xg,vg-0.5*k2vg+k3vg,zg-0.5*k2zg+k3zg,y3g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        y4g = fg4(xg-0.5*k2xg+k3xg,vg-0.5*k2vg+k3vg,zg-0.5*k2zg+k3zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        
        k5xg = deltat*fg1(xg+3/16*k1xg+9/16*k4xg,vg+3/16*k1vg+9/16*k4vg,zg+3/16*k1zg+9/16*k4zg,y4g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k5vg = deltat*fg2(xg+3/16*k1xg+9/16*k4xg,vg+3/16*k1vg+9/16*k4vg,zg+3/16*k1zg+9/16*k4zg,y4g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k5zg = deltat*fg3(xg+3/16*k1xg+9/16*k4xg,vg+3/16*k1vg+9/16*k4vg,zg+3/16*k1zg+9/16*k4zg,y4g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        y5g = fg4(xg+3/16*k1xg+9/16*k4xg,vg+3/16*k1vg+9/16*k4vg,zg+3/16*k1zg+9/16*k4zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        
        k6xg = deltat*fg1(xg-3/7*k1xg+2/7*k2xg+12/7*k3xg-12/7*k4xg+8/7*k5xg,vg-3/7*k1vg+2/7*k2vg+12/7*k3vg-12/7*k4vg+8/7*k5vg,zg-3/7*k5zg+2/7*k2zg+12/7*k3zg-12/7*k4zg+8/7*k5zg,y5g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k6vg = deltat*fg2(xg-3/7*k1xg+2/7*k2xg+12/7*k3xg-12/7*k4xg+8/7*k5xg,vg-3/7*k1vg+2/7*k2vg+12/7*k3vg-12/7*k4vg+8/7*k5vg,zg-3/7*k5zg+2/7*k2zg+12/7*k3zg-12/7*k4zg+8/7*k5zg,y5g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        k6zg = deltat*fg3(xg-3/7*k1xg+2/7*k2xg+12/7*k3xg-12/7*k4xg+8/7*k5xg,vg-3/7*k1vg+2/7*k2vg+12/7*k3vg-12/7*k4vg+8/7*k5vg,zg-3/7*k5zg+2/7*k2zg+12/7*k3zg-12/7*k4zg+8/7*k5zg,y5g,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        
        xg = xg + 1./90 *(7*k1xg + 32*k3xg + 12*k4xg + 32*k5xg + 7*k6xg)
        vg = vg + 1./90 *(7*k1vg + 32*k3vg + 12*k4vg + 32*k5vg + 7*k6vg)
        zg = zg + 1./90 *(7*k1zg + 32*k3zg + 12*k4zg + 32*k5zg + 7*k6zg)
        yg = fg4(xg,vg,zg,X0,Y0,Z0,V0,A,H,M,C,kgf1,kgf2,kgf3,kgf4,kgf5,kgf6,kgf7,alpha,beta,kf,T0)
        tg = tg + deltat
        
        xglist[i+1]=xg
        vglist[i+1]=vg
        zglist[i+1]=zg
        yglist[i+1]=yg
        tglist[i+1]=tg
    
    return xglist,vglist,zglist,yglist,tglist
