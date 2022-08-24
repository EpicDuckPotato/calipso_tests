include("./dynamical_system.jl")

function dynamics_constraint(system::DynamicalSystem, y::Vector{Float64}, x::Vector{Float64}, u::Vector{Float64})
  return xdiff(system, discrete_dynamics(system, x, u), y)
end

function dynamics_constraint_jacobians(system::DynamicalSystem, y::Vector{Float64}, x::Vector{Float64}, u::Vector{Float64})
  nx = system.nx
  nu = system.nu

  Fx = zeros(nx, nx)
  Fu = zeros(nx, nu)

  xnext = discrete_dynamics(system, x, u)
  eps = 1e-7
  for i=1:nx
    x1 = copy(x)
    x1[i] += eps
    xnext1 = discrete_dynamics(system, x1, u)
    Fx[1:nx, i] = (xnext1 - xnext)/eps
  end

  for i=1:nu
    u1 = copy(u)
    u1[i] += eps
    xnext1 = discrete_dynamics(system, x, u1)
    Fu[1:nx, i] = (xnext1 - xnext)/eps
  end

  Jxnext, Jy = Jxdiff(system, xnext, y)

  # wrt y, then x, then u
  return Jy, Jxnext*Fx, Jxnext*Fu
end
