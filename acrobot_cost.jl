using Rotations
using LinearAlgebra

include("./stage_cost.jl")
include("./acrobot.jl")

struct AcrobotCost <: StageCost
  des_config::Vector{Float64}
  config_cost_weight::Float64
  vel_cost_weight::Float64
  control_cost_weight::Float64
end

function cost(stage_cost::AcrobotCost, system::Acrobot, x::Vector{Float64}, u::Vector{Float64})
  config_err = x[1:2] - stage_cost.des_config
  config_cost = 0.5*stage_cost.config_cost_weight*dot(config_err, config_err)
  vel_cost = 0.5*stage_cost.vel_cost_weight*dot(x[3:4], x[3:4])
  control_cost = 0.5*stage_cost.control_cost_weight*dot(u, u)
  return config_cost + vel_cost + control_cost
end

function grads(stage_cost::AcrobotCost, system::Acrobot, x::Vector{Float64}, u::Vector{Float64})
  Lx = zeros(system.nx)
  config_err = x[1:2] - stage_cost.des_config
  Lx[1:2] = stage_cost.config_cost_weight*config_err
  Lx[3:4] = stage_cost.vel_cost_weight*x[3:4]
  Lu = stage_cost.control_cost_weight*u
  return Lx, Lu
end

function hessians(stage_cost::AcrobotCost, system::Acrobot, x::Vector{Float64}, u::Vector{Float64})
  Lxx = zeros(system.nx, system.nx)
  Lxx[1:2, 1:2] = stage_cost.config_cost_weight*I(2)
  Lxx[3:4, 3:4] = stage_cost.vel_cost_weight*I(2)
  Lux = zeros(system.nu, system.nx)
  Luu = stage_cost.control_cost_weight*I(system.nu)
  return Lxx, Lux, Luu
end
