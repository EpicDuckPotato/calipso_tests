import RigidBodyDynamics as RBD
using LinearAlgebra
using StaticArrays

using MeshCat
import MeshCatMechanisms as MCM

using GeometryBasics: HyperRectangle, Vec

import CALIPSO

using Rotations

include("./lie_groups.jl")
include("./snake.jl")
include("./snake_cost.jl")
include("./transcription.jl")

urdf_file = "urdf/snake.urdf"
buoyancy_urdf_file = "urdf/snake_buoyancy.urdf"
neutrally_buoyant = false
dt = 0.05
com_drag_coeffs = [1.0, 1.0, 1.0]
thruster_force_limit = 26.0
joint_torque_limit = 7.0
joint_angle_limit = π/2
system = Snake(urdf_file, buoyancy_urdf_file, neutrally_buoyant, dt, com_drag_coeffs, joint_angle_limit, joint_torque_limit, thruster_force_limit)

des_pos = [0, 0, 0]
des_orientation_inv = inv(RotationVec(-π/2, 0, 0)*RotationVec(0, 0, π))
des_joints = [0, -π/2]
position_cost_weight = 1
orientation_cost_weight = 1
joints_cost_weight = 1
joints_vel_cost_weight = 1
control_cost_weight = 0.01

stage_cost = SnakeCost(des_pos, des_orientation_inv, des_joints, position_cost_weight, orientation_cost_weight, joints_cost_weight, joints_vel_cost_weight, control_cost_weight)

x0 = zeros(system.nx)
rotvec = RotationVec(-π/2, 0, 0)
x0[1] = rotvec.sx
x0[2] = rotvec.sy
x0[3] = rotvec.sz

steps = 100

transcription = Transcription(system, stage_cost, steps, x0)

f!(result, w, θ) = objective!(transcription, result, w)
fw!(grad, w, θ) = gradient!(transcription, grad, w)
fθ!(grad, w, θ) = 0
# fww!(H, w, θ) = hessian!(transcription, H, w)
fww!(H, w, θ) = full_hessian!(transcription, H, w)
fwθ!(H, w, θ) = 0
# fww_sparsity = hessian_sparsity(transcription)
fww_sparsity = full_hessian_sparsity(transcription)
fwθ_sparsity = Vector{Tuple{Int64, Int64}}(undef, 0)

g!(c, w, θ) = equality!(transcription, c, w)
gw!(J, w, θ) = jacobian!(transcription, J, w)
gθ!(J, w, θ) = 0
gw_sparsity = jacobian_sparsity(transcription)
gθ_sparsity = Vector{Tuple{Int64, Int64}}(undef, 0)
gᵀy!(result, w, θ, y) = equality_dual!(transcription, result, w, y)
gᵀyw!(grad, w, θ, y) = equality_dual_gradient!(transcription, grad, w, y)
gᵀyww!(H, w, θ, y) = equality_dual_hessian!(transcription, H, w, y)
gᵀywθ!(H, w, θ, y) = 0
gᵀyww_sparsity = equality_dual_hessian_sparsity(transcription)
gᵀywθ_sparsity = Vector{Tuple{Int64, Int64}}(undef, 0)

h!(c, w, θ) = cones!(transcription, c, w)
hw!(J, w, θ) = cone_jacobian!(transcription, J, w)
hθ!(J, w, θ) = 0
hw_sparsity = cone_jacobian_sparsity(transcription)
hθ_sparsity = Vector{Tuple{Int64, Int64}}(undef, 0)
hᵀy!(result, w, θ, y) = cone_dual!(transcription, result, w, y)
hᵀyw!(grad, w, θ, y) = cone_dual_gradient!(transcription, grad, w, y)
hᵀyww!(H, w, θ, y) = cone_dual_hessian!(transcription, H, w, y)
hᵀywθ!(H, w, θ, y) = 0
hᵀyww_sparsity = cone_dual_hessian_sparsity(transcription)
hᵀywθ_sparsity = Vector{Tuple{Int64, Int64}}(undef, 0)

w0 = vcat([vcat(x0, zeros(system.nu)) for step=1:transcription.steps + 1]...)

methods = CALIPSO.ProblemMethods(
    f!, fw!, fθ!, fww!, fwθ!,
        zeros(length(fww_sparsity)), zeros(length(fwθ_sparsity)),
        fww_sparsity, fwθ_sparsity,
    g!, gw!, gθ!,
        zeros(length(gw_sparsity)), zeros(length(gθ_sparsity)),
        gw_sparsity, gθ_sparsity,
    gᵀy!, gᵀyw!, gᵀyww!, gᵀywθ!,
        zeros(length(gᵀyww_sparsity)), zeros(length(gᵀywθ_sparsity)),
        gᵀyww_sparsity, gᵀywθ_sparsity,
    h!, hw!, hθ!,
        zeros(length(hw_sparsity)), zeros(length(hθ_sparsity)),
        hw_sparsity, hθ_sparsity,
    hᵀy!, hᵀyw!, hᵀyww!, hᵀywθ!,
        zeros(length(hᵀyww_sparsity)), zeros(length(hᵀywθ_sparsity)),
        hᵀyww_sparsity, hᵀywθ_sparsity,
)

num_parameters = 0
solver = CALIPSO.Solver(methods,
                        num_variables(transcription),
                        num_parameters,
                        num_equality(transcription),
                        num_cone(transcription),
                        nonnegative_indices=nonnegative_indices(transcription))
CALIPSO.initialize!(solver, w0)
CALIPSO.solve!(solver)
ts = Vector{Float64}(undef, transcription.steps + 1)
qs = Vector{Vector{Float64}}(undef, transcription.steps + 1)
for step=1:transcription.steps + 1
  ts[step] = dt*(step - 1)
  x = @view solver.solution.variables[xstart(transcription, step):xend(transcription, step)]
  qs[step] = zeros(system.nq)
  qs[step][4:system.nq] = x[3:system.nq - 1]
  qs[step][1:4] = Rotations.params(UnitQuaternion(RotationVec(x[1], x[2], x[3])))
end

vis = Visualizer()

mvis = MCM.MechanismVisualizer(system.robot, MCM.URDFVisuals(urdf_file), vis)
for body=RBD.bodies(system.robot)
  if body.name == "head"
    MCM.setelement!(mvis, RBD.default_frame(body))
  end
end

render(vis)
open(vis)

animation = Animation(mvis, ts, qs)
setanimation!(mvis, animation)

readline()
