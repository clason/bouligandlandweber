"""
Bouligand-Landweber iteration for the nonsmooth inverse problem F(u) = y^δ
with y = F(u) solving the nonsmooth semilinear elliptic equation

(1)     -Δy + max(y,0) = u in Ω, y = 0 on ∂Ω.

The Bouligand-Landweber method is defined as

(2)     u^δ\\_{n+1} = u^δ\\_{n} + w\\_n (G\\_{u^δ\\_n})* (y^δ - F(u^δ\\_n))

for G\\_{u^δ\\_n} a Bouligand subderivative of S at u^δ\\_n. The iteration
is stopped with the disrepancy principle. F is evaluated by solving (1) using
a semismooth Newton method.  
For details, see

Christian Clason, Vu Huu Nhu:
Bouligand-Landweber iteration for a non-smooth ill-posed problem,
arXiv:1803.02290
"""
module BouligandLandweber

using LinearAlgebra,SparseArrays
using Printf

export run_example,FEM

"finite element structure holding mesh and assembled matrices"
struct FEM
    N::Int64                         # number of vertices per dimension
    x::Vector{Float64}               # x coordinates of inner nodes
    y::Vector{Float64}               # y coordinates of inner nodes
    A::SparseMatrixCSC{Float64}      # stiffness matrix
    AT::SparseMatrixCSC{Float64}     # adjoint stiffness matrix
    M::SparseMatrixCSC{Float64}      # mass matrix
    LM::SparseMatrixCSC{Float64}     # lumped mass matrix
    ndof::Int64                      # number of degrees of freedom
    norm                             # function computing the L2 norm
    relerror                         # function computing the relative error
end

"generate grid and assemble matrices for N×N uniform grid"
function FEM(N::Int64)
    # set up mesh
    dx = 1/N
    xm = range(dx,step=dx,length=N-1)
    xx,yy = reshape(xm, 1, N-1), reshape(xm, N-1, 1)
    x,y = repeat(xx,outer=(N-1,1))[:], repeat(yy,outer=(1,N-1))[:]
    ndof = (N-1)*(N-1)
    # set up stiffness matrix, lumped mass matrix
    e  = fill(1.0,N-1)
    D2 = spdiagm(-1=>-e[2:end],0=>2e,1=>-e[2:end])
    Id = sparse(I,N-1,N-1)
    A  = kron(D2,Id)+kron(Id,D2)
    AT = SparseMatrixCSC(A') # precompute, otherwise addition slow
    LM = sparse(dx*dx*I,ndof,ndof)
    # set up mass matrix
    rows = Int64[]; cols = Int64[]; vals = Float64[]
    for i=1:N-1
        for j=1:N-1
            # entries on diagonal
            ind_node = (N-1)*(j-1) + i-1
            append!(vals,0.5*dx*dx)
            append!(rows,ind_node)
            append!(cols,ind_node)
            # entries off diagonal
            val = 1/12*dx*dx
            if i < N-1     # right vertex
                append!(vals,val)
                append!(rows,ind_node+1)
                append!(cols,ind_node)
                append!(vals,val)
                append!(rows,ind_node)
                append!(cols,ind_node+1)
            end
            if j < N-1     # top vertex
                append!(vals,val)
                append!(rows,ind_node+(N-1))
                append!(cols,ind_node)
                append!(vals,val)
                append!(rows,ind_node)
                append!(cols,ind_node+N-1)
            end
            if (i<N-1) & (j<N-1) # top right vertex
                append!(vals,val)
                append!(rows,ind_node+N)
                append!(cols,ind_node)
                append!(vals,val)
                append!(rows,ind_node)
                append!(cols,ind_node+N)
            end
        end
    end
    M = sparse(rows.+1,cols.+1,vals)
    # functions to compute L2 norms and relative errors
    l2norm = (u)->sqrt(u'*M*u)
    relerror = (u,v)->l2norm(u-v)/l2norm(v)
    FEM(N,x,y,A,AT,M,LM,ndof,l2norm,relerror)
end

"evaluate forward mapping by solving (1) using semismooth Newton method" 
function F(fem::FEM,u,yn=zero(u))
    En = yn.>=0; Enew = similar(En)
    SSNit = 0
    converged = false
    rhs = similar(yn)
    while !converged
        SSNit += 1 
        DN = spdiagm(0=>En)
        rhs .= -fem.A*yn .- fem.LM*max.(yn,0) .+ fem.M*u
        yn .+= (fem.A+fem.LM*DN)\rhs
        Enew .= yn.>=0
        converged = Enew==En
        En .= Enew
    end
    return yn,SSNit
end

"apply adjoint Bouligand derivative by solving 'adjoint' equation"
function GuT(fem::FEM,y,h)
    DN = spdiagm(0=>(y.>=0))
    eta = (fem.AT+fem.LM*DN)\(fem.M*h) 
    return eta
end

"apply modified Landweber method BL to ydelta"
function modifiedLandweber(BLparams,delta,ydelta,uexact)
    u0,wn,tau,maxit,fem = BLparams
    un = copy(u0)
    yn,nSSN = F(fem,un)
    res = ydelta .- yn
    resnorm = fem.norm(res)
    BLit  = 0
    SSNit = 0
    @printf("It\t#SSN\tresidual\trelative error\n")
    while (resnorm > tau*delta) & (BLit <= maxit)
        BLit +=1; SSNit += nSSN
        un .+= wn*GuT(fem,yn,res)
        yn,nSSN = F(fem,un,yn)         # paper: F(un) (no warmstarts)
        res .= ydelta .- yn
        resnorm = fem.norm(res)
        errnorm = fem.relerror(un,uexact)
        @printf("%d\t%d\t%1.2e\t%1.2e\n",BLit,SSNit,resnorm,errnorm)
    end
    if BLit > maxit
        @printf("Failed to converge\n")
    else
        rate = fem.norm(un-uexact)/sqrt(delta)
        @printf("Estimated convergence rate %1.2f\n",rate)
    end
    return un
end

"compute exact solution and data with F nondifferentiable on 2β-measure set"
function exact_sol(fem,β)
    x,y = fem.x,fem.y
    chi = @. (x>=β)&(x<=1-β)
    yex = @. (x-β)^2*(x-1+β)^2*sin(2π*y)*chi
    uex = @. chi*(4π^2*yex-2*((2*x-1)^2+2*(x-β)*(x-1+β))*sin(2π*y))+max(yex,0)
    return yex,uex
end

"""
    run_example(N,δ,β)

Driver function for Bouligand-Landweber iteration: Create data and compute 
reconstruction for an N×N discretization of the unit square, noise level δ, 
and exact parameter at which the forward mapping is nondifferentiable for a 
set of measure 2β
"""
function run_example(N,δ,β)
    # setup finite element discretization, construct exact and noisy data
    fem = FEM(N)
    yexact,uexact = exact_sol(fem,β)
    ydelta = yexact .+ 1.5*δ*randn(fem.ndof)
    δ₂ = fem.norm(yexact-ydelta)      # noise level in L2
    # initialize structure for Bouligand-Landweber method
    L = 5.e-2             # estimate of Lipschitz constant of F
    μ = 1.e-1             # constant in generalized tangential cone condition
    wn = (2-2*μ)/L^2      # step size (fixed)
    τ = 1.4               # factor for discrepancy principle
    maxit = 5000          # maximum number of Landweber iterations
    # starting point u0 = \bar u satisfying source condition
    u0 = @. uexact - (10*sin(π*fem.x)*sin(2π*fem.y)) 
    # apply modified Landweber method
    @printf("noise level delta = %1.5e, tau = %1.1f\n",δ₂,τ)
    BLparams = u0,wn,τ,maxit,fem
    uN = modifiedLandweber(BLparams,δ₂,ydelta,uexact)
    return uN
end

end
