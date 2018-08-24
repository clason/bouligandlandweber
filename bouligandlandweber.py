"""
Bouligand-Landweber iteration for the nonsmooth inverse problem F(u) = y^δ
with y = F(u) solving the nonsmooth semilinear elliptic equation
(1)     -Δy + max(y,0) = u in Ω, y = 0 on ∂Ω.
The Bouligand-Landweber method is defined as
(2)     u^δ_{n+1} = u^δ_{n} + w_n (G_{u^δ_n})* (y^δ - F(u^δ_n))
for G_{u^δ_n} a Bouligand subderivative of S at u^δ_n. The iteration is
stopped with the disrepancy principle. F is evaluated by solving (1) using
a semismooth Newton method.

For details, see
Christian Clason, Vu Huu Nhu:
Bouligand-Landweber iteration for a non-smooth ill-posed problem,
arXiv:1803.02290
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

N   = 128               # number of nodes per dimension (paper: N=512)
delta = 1e-4            # noise level (componentwise)
beta = 0.005            # measure of domain where F(u^\dag) not differentiable 
figs = False            # toggle plotting (slow for N>128)

# parameters for Landweber iteration
L   = 5.e-2             # estimate of Lipschitz constant of F
mu  = 1.e-1             # constant in generalized tangential cone condition
wn  = (2-2*mu)/L**2     # step size
tau = 1.4               # parameter for discrepancy principle
maxit = 1000            # maximum number of iterations

# setup mesh
dx = 1./N             # shortest length of edges
xm = np.linspace(dx, 1.-dx, num=N-1) # inner nodes (1D)
X, Y = np.meshgrid(xm,xm)
X, Y = X.ravel(), Y.ravel()
nel = (N-1)**2
# setup stiffness matrix, lumped mass matrix
ex = np.ones(N-1) 
D2 = sp.diags([-1.*ex[0:N-2], 2.*ex, -1.*ex[0:N-2]], [-1,0,1])
Id = sp.eye(N-1)
A  = sp.kron(Id,D2) + sp.kron(D2,Id)
LM = dx*dx*sp.eye(nel)
# setup mass matrix
rows, cols, vals = [], [], []
for i in range(1,N):
    for j in range(1,N):
        # entries on diagonal
        ind_node = (N-1)*(j-1) + (i-1)
        vals.append(0.5*dx*dx)
        rows.append(ind_node)
        cols.append(ind_node)
        # entries off diagonal
        val = 1./12*dx*dx
        if i < N-1:     # right vertex
            vals.append(val)
            rows.append(ind_node+1)
            cols.append(ind_node)
            vals.append(val)
            rows.append(ind_node)
            cols.append(ind_node+1)
        if j < N-1:     # top vertex
            vals.append(val)
            rows.append(ind_node+(N-1))
            cols.append(ind_node)
            vals.append(val)
            rows.append(ind_node)
            cols.append(ind_node+N-1)
        if (i<N-1) and (j<N-1): # top right vertex
            vals.append(val)
            rows.append(ind_node+N)
            cols.append(ind_node)
            vals.append(val)
            rows.append(ind_node)
            cols.append(ind_node+N)

M = sp.csr_matrix((vals,(rows,cols)), shape = (nel,nel))

def l2norm(u):
    """compute L2 norm of u"""
    return np.sqrt(np.dot(u,M*u))

def l2relerror(u,v):
    """compute relative L2 error of u with respect to v"""
    udiff = u - v
    return l2norm(udiff)/l2norm(v)

def plot_fun(Z,title,fig=None):
    """plot function (non-blocking)"""
    if figs:
        if not fig:
            fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_trisurf(X, Y, Z, cmap=cm.viridis)
        ax.set_title(title)
        plt.draw()
        plt.pause(.001)
        return fig

def F(u,yn=None):
    """Evaluate F by solving discretized PDE (1) via SSN method"""
    if yn is None:
        yn = np.zeros(nel)
    En = (yn >= 0).astype(float)
    SSNit = 0
    converged = False
    while not converged:
        SSNit += 1
        DN = sp.diags(En)
        rhs = -A*yn - LM*np.maximum(yn,0) + M*u
        sk = spsolve(A+LM*DN,rhs)
        yn += sk
        Enew = (yn>=0).astype(float)
        converged =  np.array_equal(En,Enew)
        En = Enew
    return yn, SSNit

def GuT(y,h):
    """Solve 'adjoint' equation (A^T+DN) eta = Mh"""
    DN  = sp.diags((y>0).astype(float))
    eta = spsolve(A.T + LM*DN,M*h)
    return eta

def Landweber_iteration(u0,delta,ydelta,uexact):
    """modified Landweber iteration with discrepancy principle"""
    un = u0.copy()
    yn,nSSN = F(un)
    res = ydelta-yn
    resnorm = l2norm(res)
    BLit  = 0  # Bouligand-Landweber steps
    SSNit = 0  # total of SSN steps
    print('It\t#SSN\tresidual\trelative error')
    while resnorm > tau*delta and BLit <= maxit:
        BLit +=1; SSNit += nSSN
        un += wn*GuT(yn,res)
        yn,nSSN = F(un,yn)         # paper: F(un) (no warmstarts)
        res = ydelta-yn
        resnorm = l2norm(res)
        errnorm = l2relerror(un,uexact)
        print('%d\t%d\t%1.2e\t%1.2e' %(BLit,SSNit,resnorm,errnorm))
    if BLit > maxit:
        print("Failed to converge")
    else: 
        rate = l2norm(un-uexact)/np.sqrt(delta)
        print('Estimated convergence rate %1.2f' %rate)
    return un

def exact_sol(beta):
    """exact solution yex, uex = - Delta yex + max(yex,0)"""
    chi = ((X >= beta) & (X <= 1.-beta)).astype(float)
    yex = (X-beta)**2*(X-1.+beta)**2*np.sin(2*np.pi*Y)*chi
    uex = chi*(4*np.pi**2*yex - 
               2*((2*X-1.)**2 + 2*(X-beta)*(X-1.+beta))*np.sin(2*np.pi*Y)) +\
          np.maximum(yex,0) 
    return yex,uex
  
# exact parameter and data
y_exact, u_exact = exact_sol(beta) 
plot_fun(u_exact,"u_exact")
plot_fun(y_exact,"y_exact")

# noisy data
y_delta = y_exact + 1.5*delta*np.random.normal(size=nel)
plot_fun(y_delta,"yᵟ")
delta_l2 = l2norm(y_exact-y_delta) # noise level in L2
print('noise level delta = %1.5e, tau = %1.1f' %(delta_l2,tau))

# starting point u0 = \bar u satisfying source condition
u0 = u_exact - (10.*np.sin(np.pi*X)*np.sin(2*np.pi*Y)) 

# compute and show reconstruction
uN = Landweber_iteration(u0,delta_l2,y_delta,u_exact)
plot_fun(uN,"u_N(δ,yᵟ)")
plt.show()
