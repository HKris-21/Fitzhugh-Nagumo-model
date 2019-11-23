import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib as mpl

dt = 0.1 

def dyn_fn(xinit, tmax, dt, args): 
    x = np.zeros((int(tmax/dt), len(xinit)))
    x[0] = xinit
    for i in range(1,int(tmax/dt)):
        x[i] = x[i-1] + dt*eqs_fn(x[i-1],args) 
    return x

def eqs_fn(x,args): 
    I, a, gamma, eps = args[0], args[1],args[2],args[3]
    v = x[0]
    w = x[1]
    dvdt = a*v**2 -a*v - v**3 + v**2 - w + I
    dwdt = eps*(v - gamma*w)
    z = np.array([dvdt, dwdt])
    return z
def v_nullcline(v,I):
    return v**2 -v-v**3+v**2 + I

def w_nullcline(v,gamma):
    return v - gamma

params = {'axes.labelsize': 16,
          'text.fontsize': 16,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10}

mpl.rc('mathtext', fontset='stixsans',default='regular')
# print(v,w)


fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([.1,.1,.85,.275])
ax1 = fig.add_axes([.1,.475,.425,.425])
ax.set_ylim(-2.5,4.5)
ax1.set_xlim(-3,3)
ax1.set_ylim(-3,3)
ax.set_xlabel('Time')
ax.set_ylabel('v or w')
ax1.set_xlabel('v')
ax1.set_ylabel('w')
axcolor = 'lightgoldenrodyellow'
axsI = plt.axes([0.6, 0.7, 0.3, 0.04])
axsa  = plt.axes([0.6, 0.65, 0.3, 0.04])
axsgamma  = plt.axes([0.6, 0.6, 0.3, 0.04])
axseps  = plt.axes([0.6, 0.55, 0.3, 0.04])
axsT  = plt.axes([0.6, 0.5, 0.3, 0.04])
axh = plt.axes([-1,-1,1,1])


sI = Slider(axsI, r'$I$', -.7, .5, valinit=0,color='maroon')
sa = Slider(axsa, r'$a$', 0, .5, valinit=0,color='g')
sgamma = Slider(axsgamma, r'$gamma$', 0.0, 10.0, valinit=1,color='midnightblue')
seps = Slider(axseps, r'$\epsilon$', 0.002, 0.008, valinit=0.1,color='m')
sT = Slider(axsT, r'$T$', 0, 200.0, valinit=200,color='b')
sxinit = Slider(axh ,'x',-5,5,valinit=0)
syinit = Slider(axh,'y',-5,5,valinit=0)


l, = ax.plot(0,0, lw=2, color='r',label='v')
lb, = ax.plot(0,0, lw=2, color='b',label='w')
l1 = ax.legend(loc=2,frameon=False)
l1, = ax1.plot(0,0, lw=2, color='k')
us = np.linspace(-3,3,1000)
l1n1, = ax1.plot(us,v_nullcline(us,sI.val),lw=2,color='r',ls='--',label='v nullcline')
l1n2, = ax1.plot(us,w_nullcline(us,sgamma.val),lw=2,color='b',ls='--',label='w nullcline')


fig.text(.5,.95,'FitzHugh-Nagumo model',ha='center',size=18)
fig.text(.625,.86,r'$\frac{dv}{dt} = a v^2 -av-v^3+v^2 - w + I$',size=14,color='r')
fig.text(.625,.8,r'$\frac{dw}{dt} = \epsilon (v -gamma w) $',size=14,color='b')


def update(val):
    xinit = [sxinit.val,syinit.val]
    T = sT.val
    I = sI.val
    a = sa.val
    gamma = sgamma.val
    eps = seps.val
    args = (I, a,gamma,eps)
    m = dyn_fn(xinit,T,dt,args)
    t = np.arange(0,T,dt)
    l.set_xdata(t)
    lb.set_xdata(t)
    l.set_ydata(m[:,0])
    lb.set_ydata(m[:,1])
    ax.set_xlim(0,T)
    us = np.linspace(-3,3,500)
    l1n1.set_xdata(us)
    l1n2.set_xdata(us)
    l1n1.set_ydata(v_nullcline(us,I)) # v nullcline
    l1n2.set_ydata(w_nullcline(us,gamma)) # w nullcline
    l1.set_xdata(m[:,0])
    l1.set_ydata(m[:,1])
    plt.draw()
sI.on_changed(update)
sa.on_changed(update)
sgamma.on_changed(update)
seps.on_changed(update)
sT.on_changed(update)

resetax = plt.axes([0.8, 0.45, 0.1, 0.04]) # Reset button
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sI.reset()
    sgamma.reset()
    sa.reset()
    seps.reset()
    sT.reset()
button.on_clicked(reset)

def onpick4(event):
    if event.inaxes == ax1:
        sxinit.val = event.xdata
        syinit.val = event.ydata
        update(0)

fig.canvas.mpl_connect('button_press_event',onpick4)

plt.show()