import RigidBodyDynamics as RBD
using LinearAlgebra
using StaticArrays
using Rotations

include("./lie_groups.jl")
include("./dynamical_system.jl")

struct Snake <: DynamicalSystem
  robot::RBD.Mechanism
  dt::Float64
  nq::Int64
  nv::Int64
  nx::Int64
  thruster_paths
  com_drag_coeffs::Vector{Float64}
  thruster_bodies::Vector{RBD.RigidBody}
  num_thrusters::Int64
  num_rotary::Int64
  nu::Int64

  body_paths_with_mass
  bodies_with_mass::Vector{RBD.RigidBody}
  coms::Vector{RBD.Spatial.Point3D}
  total_mass::Float64

  buoyancy_robot::RBD.Mechanism
  neutrally_buoyant::Bool

  x_lb::Vector{Float64}
  x_ub::Vector{Float64}
  u_lb::Vector{Float64}
  u_ub::Vector{Float64}
end

function Snake(urdf_file::String, buoyancy_urdf_file::String, neutrally_buoyant::Bool, dt::Float64, com_drag_coeffs::Vector{Float64}, joint_angle_limit::Float64, joint_torque_limit::Float64, thruster_force_limit::Float64)
  gravity = [0, 0, -9.81]
  if neutrally_buoyant
    gravity[2] = 0
  end

  robot = RBD.parse_urdf(urdf_file, floating=true, gravity=gravity, remove_fixed_tree_joints=false)
  path_type = typeof(RBD.path(robot, RBD.root_body(robot), RBD.findbody(robot, "head")))
  thruster_paths = Vector{path_type}()
  thruster_bodies = Vector{RBD.RigidBody}()
  for body=RBD.bodies(robot)
    if occursin("propeller", body.name)
      push!(thruster_paths, RBD.path(robot, RBD.root_body(robot), body))
      push!(thruster_bodies, body)
    end
  end

  nq = RBD.num_positions(robot)
  nv = RBD.num_velocities(robot)
  nx = 2*nv
  num_thrusters = length(thruster_paths)
  num_rotary = nv - 6
  nu = num_thrusters + num_rotary

  total_mass = 0
  body_paths_with_mass = Vector{path_type}()
  bodies_with_mass = Vector{RBD.RigidBody}()
  coms = Vector{RBD.Spatial.Point3D}()
  for body=RBD.bodies(robot)
    if RBD.has_defined_inertia(body) && body.inertia.mass != 0
      total_mass += body.inertia.mass
      push!(body_paths_with_mass, RBD.path(robot, RBD.root_body(robot), body))
      push!(bodies_with_mass, body)
      push!(coms, RBD.Spatial.center_of_mass(body.inertia))
    end
  end

  buoyancy_robot = RBD.parse_urdf(buoyancy_urdf_file, floating=true, gravity=-gravity, remove_fixed_tree_joints=false)

  u_ub = ones(nu)
  u_ub[1:num_thrusters] *= thruster_force_limit
  u_ub[num_thrusters + 1:nu] *= joint_torque_limit
  u_lb = -u_ub

  x_ub = ones(nx)
  x_ub[1:3] *= π
  x_ub[4:6] *= 100
  x_ub[7:nv] *= joint_angle_limit
  x_ub[nv:nx] *= 100
  x_lb = -x_ub

  return Snake(robot,
               dt,
               nq,
               nv,
               nx,
               thruster_paths,
               com_drag_coeffs,
               thruster_bodies,
               num_thrusters,
               num_rotary,
               nu,
               body_paths_with_mass,
               bodies_with_mass,
               coms,
               total_mass,
               buoyancy_robot,
               neutrally_buoyant,
               x_lb, x_ub, u_lb, u_ub)
end

function xdiff(system::Snake, x1::Vector{Float64}, x2::Vector{Float64})
  diff = x2 - x1
  H1 = zeros(4, 4)
  H1[4, 4] = 1
  H2 = copy(H1)
  H1[1:3, 1:3] = RotMatrix(RotationVec(x1[1], x1[2], x1[3]))
  H1[1:3, 4] = copy(x1[4:6])
  H2[1:3, 1:3] = RotMatrix(RotationVec(x2[1], x2[2], x2[3]))
  H2[1:3, 4] = copy(x2[4:6])
  diff[1:6] = Log6(inv(H1)*H2)
  return diff
end

function Jxdiff(system::Snake, x1::Vector{Float64}, x2::Vector{Float64})
  Jx1 = zeros(system.nx, system.nx) + -I(system.nx)
  Jx2 = zeros(system.nx, system.nx) + I(system.nx)

  H1 = zeros(4, 4)
  H1[4, 4] = 1
  H2 = copy(H1)
  H1[1:3, 1:3] = RotMatrix(RotationVec(x1[1], x1[2], x1[3]))
  H1[1:3, 4] = copy(x1[4:6])
  H2[1:3, 1:3] = RotMatrix(RotationVec(x2[1], x2[2], x2[3]))
  H2[1:3, 4] = copy(x2[4:6])
  Hdiff = inv(H1)*H2

  JLog_diff = JLog6(Hdiff)

  Jx2[1:6, 1:6] = JLog_diff
  rmat2 = RotMatrix(RotationVec(x2[1], x2[2], x2[3]))
  jExp_rotvec2 = JExp3(x2[1:3])
  Jx2[1:6, 1:3] = Jx2[1:6, 1:3]*jExp_rotvec2
  Jx2[1:6, 4:6] = Jx2[1:6, 4:6]*transpose(rmat2)

  Jx1[1:6, 1:6] = -JLog_diff*Adj_se3(inv(Hdiff))
  rmat1 = RotMatrix(RotationVec(x1[1], x1[2], x1[3]))
  jExp_rotvec1 = JExp3(x1[1:3])
  Jx1[1:6, 1:3] = Jx1[1:6, 1:3]*jExp_rotvec1
  Jx1[1:6, 4:6] = Jx1[1:6, 4:6]*transpose(rmat1)

  return Jx1, Jx2
end

function integrate(system::Snake, q::Vector{Float64}, dq::Vector{Float64})
  H = Exp6(dq[1:6])
  orientation = UnitQuaternion(q[1], q[2], q[3], q[4])
  qplus = zeros(length(q))
  qplus[5:7] = q[5:7] + orientation*H[1:3, 4] # Integrate position
  orientation_plus = UnitQuaternion(RotMatrix(orientation)*H[1:3, 1:3]) # Integrate orientation
  qplus[1:4] = Rotations.params(orientation_plus)
  qplus[8:system.nq] = q[8:system.nq] + dq[7:system.nv]
  return qplus
end

function discrete_dynamics(system::Snake, x::Vector{Float64}, u::Vector{Float64})
  robot_state = RBD.MechanismState(system.robot)

  # Populate state from x
  q = zeros(system.nq)
  orientation = RotationVec(x[1], x[2], x[3])
  q[1:4] = Rotations.params(UnitQuaternion(orientation))
  q[5:7] = x[4:6]
  for i=1:system.num_rotary
    q[7 + i] = x[6 + i]
  end

  v = zeros(system.nv)
  for i=1:system.nv
    v[i] = x[system.nv + i]
  end

  RBD.set_configuration!(robot_state, q)
  RBD.set_velocity!(robot_state, v)

  # Populate generalized torque vector from u
  tau = zeros(system.nv)
  for i=1:system.num_rotary
    tau[6 + i] = u[system.num_thrusters + i]
  end
  for i=1:system.num_thrusters
    J = RBD.geometric_jacobian(robot_state, system.thruster_paths[i])

    # J is a spatial Jacobian, expressed in the world frame, so I need
    # to multiply by an adjoint to turn it into a body Jacobian
    thruster_H = RBD.transform_to_root(robot_state, RBD.default_frame(system.thruster_bodies[i])).mat
    adj = zeros(6) # Sixth row of adjoint, corresponding to linear z body velocity of thruster frame
    adj[4:6] = view(thruster_H, 1:3, 3)
    adj[1:3] = -transpose(thruster_H[1:3, 3])*skew(thruster_H[1:3, 4])
    tau += transpose(transpose(adj[4:6])*J.linear + transpose(adj[1:3])*J.angular)*u[i]
  end

  # Apply drag force
  Jcom = zeros(3, system.nv)
  num_bodies_with_mass = length(system.bodies_with_mass)
  for i=1:num_bodies_with_mass
    J = RBD.point_jacobian(robot_state, system.body_paths_with_mass[i], system.coms[i])
    body_H = RBD.transform_to_root(robot_state, RBD.default_frame(system.bodies_with_mass[i])).mat
    Jcom[1:3, 1:system.nv] += system.bodies_with_mass[i].inertia.mass*body_H[1:3, 1:3]*J.J/system.total_mass
  end

  #=
  # Finite-diff center-of-mass Jacobian
  Jcom_fd = zeros(6, system.nv)
  com = RBD.Spatial.center_of_mass(robot_state)
  eps = 1e-7
  for i=1:system.nv
    dq = zeros(system.nv)
    dq[i] = eps
    q1 = integrate(system, q, dq)
    RBD.set_configuration!(robot_state, q1)
    com1 = RBD.Spatial.center_of_mass(robot_state)
    Jcom_fd[4:6, i] = ((com1 - com)/eps).v
  end
  Jcom_fd[1:3, 1:3] = I(3)
  RBD.set_configuration!(robot_state, q)

  # display(Jcom - Jcom_fd)
  display(Jcom)
  display(Jcom_fd)
  println()
  =#

  tau -= transpose(Jcom)*Diagonal(system.com_drag_coeffs)*Jcom*v

  if ~system.neutrally_buoyant
    buoyancy_robot_state = RBD.MechanismState(system.buoyancy_robot)
    RBD.set_configuration!(buoyancy_robot_state, q)
    RBD.set_velocity!(buoyancy_robot_state, zeros(length(v)))
    tau -= RBD.dynamics_bias(buoyancy_robot_state)
  end

  result = RBD.DynamicsResult{Float64}(system.robot)
  RBD.dynamics!(result, robot_state, tau)
  
  xnext = zeros(2*system.nv)
  xnext[system.nv + 1:system.nx] = x[system.nv + 1:system.nx] + result.v̇*system.dt # Integrate velocity
  xnext[7:system.nv] = x[7:system.nv] + xnext[system.nv + 7:system.nx]*system.dt # Integrate joint angles
  H = Exp6(xnext[system.nv + 1:system.nv + 6]*system.dt)
  xnext[4:6] = x[4:6] + orientation*H[1:3, 4] # Integrate position
  rotvec_next = RotationVec(RotMatrix(orientation)*H[1:3, 1:3]) # Integrate orientation
  xnext[1] = rotvec_next.sx
  xnext[2] = rotvec_next.sy
  xnext[3] = rotvec_next.sz
  return xnext
end
