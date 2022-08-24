import RigidBodyDynamics as RBD
using LinearAlgebra
using StaticArrays

using MeshCat
import MeshCatMechanisms as MCM

using GeometryBasics: HyperRectangle, Vec

import CALIPSO

include("./acrobot.jl")
include("./acrobot_cost.jl")
include("./transcription.jl")

urdf_file = "urdf/Acrobot.urdf"
dt = 0.05
system = Acrobot(urdf_file, dt)

des_config = [π, 0]
config_cost_weight = 10
vel_cost_weight = 1
control_cost_weight = 1

stage_cost = AcrobotCost(des_config, config_cost_weight, vel_cost_weight, control_cost_weight)

x0 = zeros(system.nx)

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
hᵀy!(result, w, θ, y) = 0
hᵀyw!(grad, w, θ, y) = 0
hᵀyww!(H, w, θ, y) = 0
hᵀywθ!(H, w, θ, y) = 0
hᵀyww_sparsity = Vector{Tuple{Int64, Int64}}(undef, 0)
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
  qs[step] = x[1:2]
  println(x)
end
  
vis = Visualizer()

mvis = MCM.MechanismVisualizer(system.robot, MCM.URDFVisuals(urdf_file), vis)

render(vis)
open(vis)

animation = Animation(mvis, ts, qs)
setanimation!(mvis, animation)

readline()
