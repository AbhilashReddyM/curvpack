import numpy as np
import matplotlib
from cycler import cycler
matplotlib.use('agg')

import matplotlib.pyplot as plt
# fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"]="stix"

def plot_formatter(ax,xmajortick_sp=2.5,xmaj_fmt='%0.1f',ymajortick_sp=0.1,ymaj_fmt='%0.1f',xminortick_num=5,yminortick_num=5):
  from cycler import cycler
  import matplotlib
  from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator # for ticks formatting
  # axis line formatting
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)

  # tick formatting
  x_majorLocator   = MultipleLocator(xmajortick_sp)
  x_majorFormatter = FormatStrFormatter(xmaj_fmt)
  x_minorLocator   = AutoMinorLocator(xminortick_num)
  
  y_majorLocator   = MultipleLocator(ymajortick_sp)
  y_majorFormatter = FormatStrFormatter(ymaj_fmt)
  y_minorLocator   = AutoMinorLocator(yminortick_num)
  
  ax.xaxis.set_major_locator(x_majorLocator)
  ax.xaxis.set_major_formatter(x_majorFormatter)
  ax.xaxis.set_minor_locator(x_minorLocator)
  
  ax.yaxis.set_major_locator(y_majorLocator)
  ax.yaxis.set_major_formatter(y_majorFormatter)
  ax.yaxis.set_minor_locator(y_minorLocator)
  
  ax.tick_params(which='major',direction='in', length=8, width=1  , bottom=True, top=True, left=True, right=True ,labelbottom=True, labeltop=False, labelleft=True, labelright=False,)
  ax.tick_params(which='minor',direction='in', length=4, width=0.5, bottom=True, top=True, left=True, right=True ,labelbottom=True, labeltop=False, labelleft=True, labelright=False,)
  return

fig,ax=plt.subplots(figsize=(4,4))

leg_text=[]

line_cycler =  ( cycler('color', ['k', 'r', 'g', 'b', 'm', 'c']) *
                 cycler('lw', [0.5,]) * cycler('linestyle', ['-',]))

marker_cycler =  (cycler('marker', ['o', 'd', 's', 'X','x','v','^','<','>','p','+','P']))

data=np.loadtxt('Errors_C1L1.dat')

dxm1=1./data[:,0]

ax.loglog(dxm1,data[:,3]      ,'-o',lw=0.75,label='normals')
ax.loglog(dxm1,data[:,5]      ,'-s',lw=0.75,label='mean curvature')
ax.loglog(dxm1,data[:,11]     ,'-d',lw=0.75,label='membrane force')
ax.loglog(dxm1,3*data[:,0]**1    ,'k:' ,lw=0.9,label='$\Delta X$')
ax.loglog(dxm1,50*data[:,0]**2,'k--',lw=0.9,label='$\Delta X^{2}$')

ax.set_xlabel('$\Delta X^{-1}$')
ax.set_ylabel('$l^1$ error')
#ax.set_title('Convergence rate of Curvature1 and Laplace-Beltrami1 ')

#ax.set_xlim([1e-4,1e-2])
#ax.set_ylim([1e-5,1e-2])


#plot_formatter(ax1,xmajortick_sp=40,xmaj_fmt='%.0f',ymajortick_sp=10,ymaj_fmt='%.0f',xminortick_num=5,yminortick_num=5)

ax.legend(frameon=False)

fig.tight_layout()
#fig.savefig('convergence.pdf')
fig.savefig('convergence.png',dpi=300)
