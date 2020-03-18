from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def compute_dX_dt(X, t, alpha_x, sigma_x, k1, k2, a, d, kappa, gamma, kf, sigma, ptot):
  """
  Computes the time derivative of the state vector.

  X = [x, x_2, C, mY, Y]
  """
  x, x2, C, mY, Y = X

  alpha_x = params

  dx_dt = alpha_x - sigma_x*x - 2*k1*x**2 + 2*k2*x2
  dx2_dt = k1*x**2 - k2*x2 - a*x2*(ptot - C) + d*C
  dC_dt = a*x2*(ptot - C) - d*C
  dy_dt = kappa*mY - gamma*Y
  dmY_dt = kf*C - sigma*mY

  return np.array([dx_dt, dx2_dt, dC_dt, dy_dt, dmY_dt])


def simulate():
  """
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
  """
  t = np.linspace(0, 10, 101)

  alpha_x =
  sigma_x =
  k1 =
  k2 =
  a =
  d =
  kappa =
  gamma =
  kf =
  sigma =
  ptot =

  X_solution = odeint(
    compute_dX_dt,
    X_0,
    t,
    args=(alpha_x, sigma_x, k1, k2, a, d, kappa, gamma, kf, sigma, ptot))

  x_solution = X_solution[:,0]
  x2_solution = X_solution[:,1]

  plt.plot(t, x_solution, 'b', label='x(t)')
  plt.plot(t, x2_solution, 'g', label='x2(t)')
  plt.xlabel('t')
  plt.grid()
  plt.show()


if __name__ == "__main__":
  pass
