using Rotations
using LinearAlgebra

include("./stage_cost.jl")
include("./snake.jl")

struct SnakeCost <: StageCost
  des_pos::Vector{Float64}
  des_orientation_inv
  des_joints::Vector{Float64}
  position_cost_weight::Float64
  orientation_cost_weight::Float64
  joints_cost_weight::Float64
  joints_vel_cost_weight::Float64
  control_cost_weight::Float64
end

function cost(stage_cost::SnakeCost, system::Snake, x::Vector{Float64}, u::Vector{Float64})
  position_err = x[4:6] - stage_cost.des_pos
  position_cost = 0.5*stage_cost.position_cost_weight*dot(position_err, position_err)

  orientation = RotationVec(x[1], x[2], x[3])
  orientation_err = RotationVec(stage_cost.des_orientation_inv*orientation)
  Log_orientation_err = [orientation_err.sx, orientation_err.sy, orientation_err.sz]
  orientation_cost = 0.5*stage_cost.orientation_cost_weight*dot(Log_orientation_err, Log_orientation_err)

  joints_err = x[7:system.nv] - stage_cost.des_joints
  joints_cost = 0.5*stage_cost.joints_cost_weight*dot(joints_err, joints_err)

  joints_vel_cost = 0.5*stage_cost.joints_vel_cost_weight*dot(x[system.nv + 7:system.nx], x[system.nv + 7:system.nx])

  control_cost = 0.5*stage_cost.control_cost_weight*dot(u, u)

  return position_cost + orientation_cost + joints_cost + joints_vel_cost + control_cost
end

function grads(stage_cost::SnakeCost, system::Snake, x::Vector{Float64}, u::Vector{Float64})
  Lx = zeros(system.nx)
  position_err = x[4:6] - stage_cost.des_pos
  Lx[4:6] = stage_cost.position_cost_weight*position_err

  orientation = RotationVec(x[1], x[2], x[3])
  orientation_err = RotationVec(stage_cost.des_orientation_inv*orientation)
  Log_orientation_err = [orientation_err.sx, orientation_err.sy, orientation_err.sz]
  Lx[1:3] = stage_cost.orientation_cost_weight*transpose(Log_orientation_err)*JLog3(orientation_err)*JExp3(x[1:3])

  joints_err = x[7:system.nv] - stage_cost.des_joints
  Lx[7:system.nv] = stage_cost.joints_cost_weight*joints_err

  Lx[system.nv + 7:system.nx] = stage_cost.joints_vel_cost_weight*x[system.nv + 7:system.nx]

  Lu = stage_cost.control_cost_weight*u

  return Lx, Lu
end

function hessians(stage_cost::SnakeCost, system::Snake, x::Vector{Float64}, u::Vector{Float64})
  Lxx = zeros(system.nx, system.nx) + I(system.nx)
  Lxx[4:6, 4:6] += stage_cost.position_cost_weight*I(3)
  orientation = RotationVec(x[1], x[2], x[3])
  orientation_err = RotationVec(stage_cost.des_orientation_inv*orientation)
  Log_orientation_err = [orientation_err.sx, orientation_err.sy, orientation_err.sz]
  ograd = stage_cost.orientation_cost_weight*transpose(Log_orientation_err)*JLog3(orientation_err)
  jExp3 = JExp3(x[1:3])
  Lxx[1:3, 1:3] = transpose(jExp3)*(transpose(ograd)*ograd + I(3))*jExp3
  Lxx[7:system.nv, 7:system.nv] += stage_cost.joints_cost_weight*I(system.num_rotary)
  Lxx[system.nv + 7:system.nx, system.nv + 7:system.nx] += stage_cost.joints_vel_cost_weight*I(system.num_rotary)

  Lux = zeros(system.nu, system.nx)

  Luu = stage_cost.control_cost_weight*I(system.nu)

  return Lxx, Lux, Luu
end
