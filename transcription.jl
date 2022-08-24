include("./stage_cost.jl")
include("./dynamics_constraint.jl")
include("./dynamical_system.jl")

using SparseArrays

struct Transcription
  system::DynamicalSystem
  stage_cost::StageCost
  steps::Int64
  vars_per_step::Int64
  x0::Vector{Float64}
end

function Transcription(system::DynamicalSystem, stage_cost::StageCost, steps::Int64, x0::Vector{Float64})
  return Transcription(system, stage_cost, steps, system.nx + system.nu, x0)
end

function xstart(transcription::Transcription, step::Int64)::Int64
  return transcription.vars_per_step*(step - 1) + 1
end

function xend(transcription::Transcription, step::Int64)::Int64
  return transcription.vars_per_step*(step - 1) + transcription.system.nx
end

function ustart(transcription::Transcription, step::Int64)::Int64
  return transcription.vars_per_step*(step - 1) + transcription.system.nx + 1
end

function uend(transcription::Transcription, step::Int64)::Int64
  return transcription.vars_per_step*step
end

# Objective function, sums the running costs
function objective!(transcription::Transcription, result, w)::Float64
  result[1] = sum([cost(transcription.stage_cost, transcription.system,
                        w[xstart(transcription, step):xend(transcription, step)], 
                        w[ustart(transcription, step):uend(transcription, step)]) 
                   for step=1:transcription.steps + 1])*transcription.system.dt
end

function gradient_fd!(transcription::Transcription, grad, w)
  nw = num_variables(transcription)
  result = zeros(1)
  objective!(transcription, result, w)
  eps = 1e-7
  for i=1:nw
    w1 = copy(w)
    w1[i] += eps
    result1 = zeros(1)
    objective!(transcription, result1, w1)
    grad[i] = (result1[1] - result[1])/eps
  end
end

# Gradient of objective
function gradient!(transcription::Transcription, grad, w)
  for step=1:transcription.steps + 1
    xstart_idx = xstart(transcription, step)
    xend_idx = xend(transcription, step)
    ustart_idx = ustart(transcription, step)
    uend_idx = uend(transcription, step)
    Lx, Lu = grads(transcription.stage_cost,
                   transcription.system,
                   w[xstart_idx:xend_idx],
                   w[ustart_idx:uend_idx])
    grad[xstart_idx:xend_idx] = Lx*transcription.system.dt
    grad[ustart_idx:uend_idx] = Lu*transcription.system.dt
  end
end

function hessian_nnz(transcription::Transcription)::Int64
  nx = transcription.system.nx
  nu = transcription.system.nu
  return (nx*(nx + 1)/2 + nu*nx + nu*(nu + 1)/2)*(transcription.steps + 1)
end

# Hessian of objective (just populates lower triangle)
function hessian!(transcription::Transcription, H, w)
  nx = transcription.system.nx
  nu = transcription.system.nu
  idx = 1
  for step=1:transcription.steps + 1
    Lxx, Lux, Luu = hessians(transcription.stage_cost, transcription.system,
                             w[xstart(transcription, step):xend(transcription, step)],
                             w[ustart(transcription, step):uend(transcription, step)])

    # Extract lower triangle
    for col=1:nx
      for row=col:nx
        H[idx] = Lxx[row, col]*transcription.system.dt
        idx += 1
      end
    end

    next_idx = idx + nu*nx
    H[idx:next_idx - 1] = vec(Lux)*transcription.system.dt
    idx = next_idx

    # Extract lower triangle
    for col=1:nu
      for row=col:nu
        H[idx] = Luu[row, col]*transcription.system.dt
        idx += 1
      end
    end
  end
end

# Just populates lower triangle
function hessian_sparsity(transcription::Transcription)
  nx = transcription.system.nx
  nu = transcription.system.nu
  coords = Vector{Tuple{Int64, Int64}}(undef, convert(Int64, hessian_nnz(transcription)))
  idx = 1
  for step=1:transcription.steps + 1
    xstart_idx = xstart(transcription, step)
    xend_idx = xend(transcription, step)
    for col=xstart_idx:xend_idx
      for row=col:xend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end

    ustart_idx = ustart(transcription, step)
    uend_idx = uend(transcription, step)
    for col=xstart_idx:xend_idx
      for row=ustart_idx:uend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end

    for col=ustart_idx:uend_idx
      for row=col:uend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end
  end

  return coords
end

function full_hessian!(transcription::Transcription, H, w)
  nx = transcription.system.nx
  nu = transcription.system.nu
  idx = 1
  for step=1:transcription.steps + 1
    Lxx, Lux, Luu = hessians(transcription.stage_cost, transcription.system,
                             w[xstart(transcription, step):xend(transcription, step)],
                             w[ustart(transcription, step):uend(transcription, step)])

    for col=1:nx
      for row=1:nx
        H[idx] = Lxx[row, col]*transcription.system.dt
        idx += 1
      end
    end

    next_idx = idx + nu*nx
    H[idx:next_idx - 1] = vec(Lux)*transcription.system.dt
    idx = next_idx

    next_idx = idx + nu*nx
    H[idx:next_idx - 1] = vec(transpose(Lux))*transcription.system.dt
    idx = next_idx

    for col=1:nu
      for row=1:nu
        H[idx] = Luu[row, col]*transcription.system.dt
        idx += 1
      end
    end
  end
end

function full_hessian_sparsity(transcription::Transcription)
  nx = transcription.system.nx
  nu = transcription.system.nu
  coords = Vector{Tuple{Int64, Int64}}(undef, convert(Int64, (nx*nx + 2*nu*nx + nu*nu)*(transcription.steps + 1)))
  idx = 1
  for step=1:transcription.steps + 1
    xstart_idx = xstart(transcription, step)
    xend_idx = xend(transcription, step)
    for col=xstart_idx:xend_idx
      for row=xstart_idx:xend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end

    ustart_idx = ustart(transcription, step)
    uend_idx = uend(transcription, step)
    for col=xstart_idx:xend_idx
      for row=ustart_idx:uend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end

    for col=ustart_idx:uend_idx
      for row=xstart_idx:xend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end

    for col=ustart_idx:uend_idx
      for row=ustart_idx:uend_idx
        coords[idx] = (row, col)
        idx += 1
      end
    end
  end

  return coords
end


# Equality constraints
function equality!(transcription::Transcription, c, w)
  # c[:] = vcat(xdiff(transcription.system, transcription.x0, w[xstart(transcription, 1):xend(transcription, 1)]),
  c[:] = vcat(w[xstart(transcription, 1):xend(transcription, 1)] - transcription.x0,
              [dynamics_constraint(transcription.system,
                                   w[xstart(transcription, step + 1):xend(transcription, step + 1)],
                                   w[xstart(transcription, step):xend(transcription, step)],
                                   w[ustart(transcription, step):uend(transcription, step)])
               for step=1:transcription.steps]...)
end

function num_equality(transcription::Transcription)::Int64
  return transcription.system.nx*(transcription.steps + 1) # Initial state constraint, then dynamics constraints between steps
end

function num_variables(transcription::Transcription)::Int64
  return transcription.vars_per_step*(transcription.steps + 1)
end

function jacobian_nnz(transcription::Transcription)::Int64
  nx = transcription.system.nx
  nu = transcription.system.nu

  init_nz = nx*nx
  nz_per_step = nx*nx + nx*nx + nx*nu
  return init_nz + nz_per_step*transcription.steps
end

# Equality constraint Jacobian
function jacobian!(transcription::Transcription, J, w)
  nx = transcription.system.nx
  nu = transcription.system.nu

  init_nz = nx*nx
  nz_per_step = nx*nx + nx*nx + nx*nu
  # Jx0, Jx = Jxdiff(transcription.system, transcription.x0, w[xstart(transcription, 1):xend(transcription, 1)])
  # J[1:init_nz] = vec(Jx)
  J[1:init_nz] = vec(I(transcription.system.nx))

  for step=1:transcription.steps
    y = w[xstart(transcription, step + 1):xend(transcription, step + 1)]
    x = w[xstart(transcription, step):xend(transcription, step)]
    u = w[ustart(transcription, step):uend(transcription, step)]
    Jy, Jx, Ju = dynamics_constraint_jacobians(transcription.system, y, x, u)

    start_idx = init_nz + nz_per_step*(step - 1) + 1
    end_idx = start_idx + nx*nx - 1
    J[start_idx:end_idx] = vec(Jy)

    start_idx += nx*nx
    end_idx += nx*nx
    J[start_idx:end_idx] = vec(Jx)

    start_idx += nx*nx
    end_idx += nx*nu
    J[start_idx:end_idx] = vec(Ju)
  end
end

function jacobian_sparsity(transcription::Transcription)
  nx = transcription.system.nx
  nu = transcription.system.nu

  coords = Vector{Tuple{Int64, Int64}}(undef, convert(Int64, jacobian_nnz(transcription)))
  idx = 1

  for col=1:nx
    for row=1:nx
      coords[idx] = (row, col)
      idx += 1
    end
  end

  for step=1:transcription.steps
    start_row = nx*step + 1
    end_row = nx*(step + 1)
    start_col = xstart(transcription, step + 1)
    end_col = xend(transcription, step + 1)
    for col=start_col:end_col
      for row=start_row:end_row
        coords[idx] = (row, col)
        idx += 1
      end
    end

    start_col = xstart(transcription, step)
    end_col = xend(transcription, step)
    for col=start_col:end_col
      for row=start_row:end_row
        coords[idx] = (row, col)
        idx += 1
      end
    end

    start_col = ustart(transcription, step)
    end_col = uend(transcription, step)
    for col=start_col:end_col
      for row=start_row:end_row
        coords[idx] = (row, col)
        idx += 1
      end
    end
  end

  return coords
end

function jacobian_fd!(transcription::Transcription, J, w)
  num_eq = num_equality(transcription)
  num_var = num_variables(transcription)
  c = zeros(num_eq)
  equality!(transcription, c, w)
  eps = 1e-7
  Jdense = zeros(num_eq, num_var)
  for i=1:num_var
    w1 = copy(w)
    w1[i] += eps
    c1 = zeros(num_eq)
    equality!(transcription, c1, w1)
    Jdense[1:num_eq, i] = (c1 - c)/eps
  end
  J[:] = vec(Jdense)
end

function jacobian_sparsity_fd(transcription::Transcription)
  num_eq = num_equality(transcription)
  num_var = num_variables(transcription)
  coords = Vector{Tuple{Int64, Int64}}(undef, num_eq*num_var)
  idx = 1
  for col=1:num_var
    for row=1:num_eq
      coords[idx] = (row, col)
      idx += 1
    end
  end
  return coords
end

function equality_dual!(transcription::Transcription, result, w, y)
  c = zeros(num_equality(transcription))
  equality!(transcription, c, w)
  result[1] = transpose(c)*y
end

function equality_dual_gradient!(transcription::Transcription, grad, w, y)
  coords = jacobian_sparsity(transcription)
  rows = [coord[1] for coord in coords]
  cols = [coord[2] for coord in coords]
  vals = zeros(length(rows))
  jacobian!(transcription, vals, w)
  grad[:] = transpose(sparse(rows, cols, vals, num_equality(transcription), num_variables(transcription)))*y
end

# We're ignoring the dynamics Hessian
function equality_dual_hessian!(transcription::Transcription, H, w, y)
end

function equality_dual_hessian_sparsity(transcription::Transcription)
  return Vector{Tuple{Int64, Int64}}(undef, 0)
end

# Cone constraints

function num_cone(transcription::Transcription)::Int64
  bound_length = (transcription.system.nx + transcription.system.nu)*(transcription.steps + 1) # State and control bounds
  return 2*bound_length
end

function cone_jacobian_nnz(transcription::Transcription)::Int64
  bound_length = (transcription.system.nx + transcription.system.nu)*(transcription.steps + 1) # State and control bounds
  return 2*bound_length
end

function nonnegative_indices(transcription::Transcription)
  return collect(1:num_cone(transcription))
end

function cones!(transcription::Transcription, c, w)
  bound_length = (transcription.system.nx + transcription.system.nu)*(transcription.steps + 1) # State and control bounds
  c[1:bound_length] = vcat([vcat(w[xstart(transcription, step):xend(transcription, step)] - transcription.system.x_lb,
                                 w[ustart(transcription, step):uend(transcription, step)] - transcription.system.u_lb)
                           for step=1:transcription.steps + 1]...)

  c[bound_length + 1:2*bound_length] = vcat([vcat(-w[xstart(transcription, step):xend(transcription, step)] + transcription.system.x_ub,
                                                  -w[ustart(transcription, step):uend(transcription, step)] + transcription.system.u_ub)
                                            for step=1:transcription.steps + 1]...)
end

function cone_jacobian_sparsity(transcription::Transcription)
  coords = Vector{Tuple{Int64, Int64}}(undef, convert(Int64, cone_jacobian_nnz(transcription)))
  bound_length = (transcription.system.nx + transcription.system.nu)*(transcription.steps + 1) # State and control bounds

  for step=1:transcription.steps + 1
    for i=xstart(transcription, step):xend(transcription, step)
      coords[i] = (i, i)
      coords[bound_length + i] = (bound_length + i, i)
    end
    for i=ustart(transcription, step):uend(transcription, step)
      coords[i] = (i, i)
      coords[bound_length + i] = (bound_length + i, i)
    end
  end

  return coords
end

function cone_jacobian!(transcription::Transcription, J, w)
  bound_length = (transcription.system.nx + transcription.system.nu)*(transcription.steps + 1) # State and control bounds
  J[:] = vcat(ones(bound_length), -ones(bound_length))
end

function cone_dual!(transcription::Transcription, result, w, y)
  c = zeros(num_cone(transcription))
  cones!(transcription, c, w)
  result[1] = transpose(c)*y
end

function cone_dual_gradient!(transcription::Transcription, grad, w, y)
  coords = cone_jacobian_sparsity(transcription)
  rows = [coord[1] for coord in coords]
  cols = [coord[2] for coord in coords]
  vals = zeros(length(rows))
  cone_jacobian!(transcription, vals, w)
  grad[:] = transpose(sparse(rows, cols, vals, num_cone(transcription), num_variables(transcription)))*y
end

# We're ignoring the cone Hessian for now
function cone_dual_hessian!(transcription::Transcription, H, w, y)
end

function cone_dual_hessian_sparsity(transcription::Transcription)
  return Vector{Tuple{Int64, Int64}}(undef, 0)
end
