using Rotations
using LinearAlgebra
using StaticArrays

function skew(v)
  return SMatrix{3, 3}([0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0])
end

function alphaSkew(alpha, v)
  return skew(alpha*v)
end

# From pinocchio: https://github.com/stack-of-tasks/pinocchio/src/math/taylor-expansion.hpp
function taylor_precision(degree)
  return eps(Float64)^(1/(degree + 1))
end

function Exp3(r)
  return RotationVec(r[1], r[2], r[3])
end

function JExp3(r)
  n = norm(r)
  n2 = n*n
  n_inv = 1/n
  n2_inv = n_inv*n_inv
  cn = cos(n)
  sn = sin(n)
  a = 0.0
  b = 0.0
  c = 0.0
  if n < taylor_precision(3)
    a = 1 - n2/6
    b = -0.5 - n2/24
    c = 1/6 - n2/120
  else
    a = sn*n_inv
    b = -(1 - cn)*n2_inv
    c = n2_inv*(1 - a)
  end

  J = zeros(3, 3) + a*I(3)
  J[1, 2] = -b*r[3]
  J[2, 1] = -J[1, 2]
  J[1, 3] = b*r[2]
  J[3, 1] = -J[1, 3]
  J[2, 3] = -b*r[1]
  J[3, 2] = -J[2, 3]
  J += c*r*transpose(r)

  return J
end

function Log3(orientation)
  rotvec = RotationVec(orientation)
  return [rotvec.sx, rotvec.sy, rotvec.sz]
end

function JLog3(orientation)
  rotvec = RotationVec(orientation)
  r = [rotvec.sx, rotvec.sy, rotvec.sz]
  theta = norm(r)
  alpha = 0.0
  diag_value = 0.0
  if theta < taylor_precision(3)
    th2 = theta*theta
    alpha = 1/12 + th2/720
    diag_value = 0.5*(2 - th2/6)
  else
    ct = cos(theta)
    st = sin(theta)
    st_1mct = st/(1 - ct)
    alpha = 1/(theta*theta) - st_1mct/(2*theta)
    diag_value = 0.5*(theta*st_1mct)
  end

  J = alpha*r*transpose(r)
  J += diag_value*I
  J += skew(0.5*r)
  return J
end

function Exp6(twist)
  H = zeros(4, 4)
  H[4, 4] = 1

  # From pinocchio: https://github.com/stack-of-tasks/pinocchio/src/spatial/explog.hpp
  w = twist[1:3]
  v = twist[4:6]
  t = norm(w)
  t2 = t*t
  ct = cos(t)
  st = sin(t)
  inv_t2 = 1/t2

  alpha_wxv = 0.0
  alpha_v = 0.0
  alpha_w = 0.0
  diagonal_term = 0.0
  if t < taylor_precision(3)
    alpha_wxv = 0.5 - t2/24
    alpha_v = 1 - t2/6
    alpha_w = 1/6 - t2/120
    diagonal_term = 1 - t2/2
  else
    alpha_wxv = (1 - ct)*inv_t2
    alpha_v = st/t
    alpha_w = (1 - alpha_v)*inv_t2
    diagonal_term = ct
  end

  H[1:3, 4] = (alpha_v*v + (alpha_w*dot(w, v))*w + alpha_wxv*cross(w, v))
  H[1:3, 1:3] = alpha_wxv*w*transpose(w)
  H[1, 2] -= alpha_v*w[3]
  H[2, 1] += alpha_v*w[3]
  H[1, 3] += alpha_v*w[2]
  H[3, 1] -= alpha_v*w[2]
  H[2, 3] -= alpha_v*w[1]
  H[3, 2] += alpha_v*w[1]
  H += I*diagonal_term

  return H
end

function JExp6(twist)
  J = zeros(6, 6)
  v = twist[4:6]
  w = twist[1:3]
  t = norm(w)
  t2 = t*t
  tinv = 1/t
  t2inv = tinv*tinv
  st = sin(t)
  ct = cos(t)
  inv_2_2ct = 1/(2*(1 - ct))

  beta = 0.0
  beta_dot_over_theta = 0.0
  if t < taylor_precision(3)
    beta = 1/12 + t2/720
    beta_dot_over_theta = 1/360
  else
    beta = t2inv - st*tinv*inv_2_2ct
    beta_dot_over_theta = -2*t2inv*t2inv + (1 + st*tinv)*t2inv*inv_2_2ct
  end

  J[1:3, 1:3] = JExp3(w)
  J[4:6, 4:6] = J[1:3, 1:3]
  p = transpose(J[4:6, 4:6])*v
  wTp = dot(w, p)
  Jtmp = alphaSkew(0.5, p) + beta_dot_over_theta*wTp*w*transpose(w) - (t2*beta_dot_over_theta + 2*beta)*p*transpose(w) + wTp*beta*I(3) + beta*w*transpose(p)
  J[4:6, 1:3] = -J[4:6, 4:6]*Jtmp
  J[1:3, 4:6] .= 0
  return J
end

function Log6(H)
  p = @view H[1:3, 4]

  w = Log3(RotMatrix(SMatrix{3, 3}(H[1:3, 1:3])))
  t = norm(w)
  t2 = t*t
  alpha = 0.0
  beta = 0.0

  if t < taylor_precision(3)
    alpha = 1 - t2/12 - t2*t2/720
    beta = 1/12 + t2/720
  else
    st = sin(t)
    ct = cos(t)
    alpha = t*st/(2*(1 - ct))
    beta = 1/t2 - st/(2*t*(1 - ct))
  end

  ret = zeros(6)
  ret[4:6] = alpha*p - 0.5*cross(w, p) + (beta*dot(w, p)) * w
  ret[1:3] = w

  return ret
end

function JLog6(H)
  J = zeros(6, 6)

  R = RotMatrix(SMatrix{3, 3}(H[1:3, 1:3]))
  p = @view H[1:3, 4]

  w = Log3(R)
  t = norm(w)

  J[4:6, 4:6] = JLog3(R)
  J[1:3, 1:3] = J[4:6, 4:6]

  t2 = t*t
  beta = 0.0
  beta_dot_over_theta = 0.0
  if t < taylor_precision(3)
    beta = 1/12 + t2/720
    beta_dot_over_theta = 1/360
  else
    tinv = 1/t
    t2inv = tinv*tinv
    st = sin(t)
    ct = cos(t)
    inv_2_2ct = 1/(2*(1 - ct))
    beta = t2inv - st*tinv*inv_2_2ct
    beta_dot_over_theta = -2*t2inv*t2inv + (1 + st*tinv)*t2inv*inv_2_2ct
  end

  wTp = dot(w, p)
  v3_tmp = beta_dot_over_theta*wTp*w - (t2*beta_dot_over_theta + 2*beta)*p
  J[1:3, 4:6] = v3_tmp*transpose(w) + beta*w*transpose(p) + wTp*beta*I(3) + alphaSkew(0.5, p)
  J[4:6, 1:3] = J[1:3, 4:6]*J[4:6, 4:6]
  J[1:3, 4:6] .= 0

  return J
end

function Adj_se3(H)
  A = zeros(6, 6)
  A[1:3, 1:3] = H[1:3, 1:3]
  A[4:6, 4:6] = H[1:3, 1:3]
  A[4:6, 1:3] = skew(H[1:3, 4])*H[1:3, 1:3]
  return A
end
