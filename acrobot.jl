import RigidBodyDynamics as RBD
using LinearAlgebra
using StaticArrays

include("./dynamical_system.jl")

struct Acrobot <: DynamicalSystem
  robot::RBD.Mechanism
  dt::Float64
  nq::Int64
  nv::Int64
  nx::Int64
  nu::Int64

  x_lb::Vector{Float64}
  x_ub::Vector{Float64}
  u_lb::Vector{Float64}
  u_ub::Vector{Float64}
end

function Acrobot(urdf_file::String, dt::Float64)
  robot = RBD.parse_urdf(urdf_file, floating=false, gravity=[0, 0, -9.81], remove_fixed_tree_joints=false)
  return Acrobot(robot, dt, 2, 2, 4, 1, [-2*π, -2*π, -10.0, -10.0], [2*π, 2*π, 10.0, 10.0], [-20.0], [20.0]) # Shoulder is actuated
end

function xdiff(system::Acrobot, x1::Vector{Float64}, x2::Vector{Float64})
  return x2 - x1
end

function Jxdiff(system::Acrobot, x1::Vector{Float64}, x2::Vector{Float64})
  Jx1 = zeros(system.nx, system.nx) + -I(system.nx)
  Jx2 = zeros(system.nx, system.nx) + I(system.nx)
  return Jx1, Jx2
end

function discrete_dynamics(system::Acrobot, x::Vector{Float64}, u::Vector{Float64})
  robot_state = RBD.MechanismState(system.robot)

  # Populate state from x
  q = @view x[1:2]

  v = @view x[3:4]

  RBD.set_configuration!(robot_state, q)
  RBD.set_velocity!(robot_state, v)

  # Populate generalized torque vector from u
  tau = zeros(system.nv)
  tau[1] = u[1]
  result = RBD.DynamicsResult{Float64}(system.robot)
  RBD.dynamics!(result, robot_state, tau)
  
  xnext = zeros(2*system.nv)
  xnext[system.nq + 1:system.nx] = x[system.nq + 1:system.nx] + result.v̇*system.dt # Integrate velocity
  xnext[1:system.nq] = x[1:system.nq] + xnext[system.nq + 1:system.nx]*system.dt # Integrate configuration
  return xnext
end
