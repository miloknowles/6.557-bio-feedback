from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/tctianchi/pyvenn/issues/3


def compute_dX_dt(X, t, alpha_x, sigma_x, k1, k2, a, d, kappa, gamma, kf, sigma, ptot):
  """
  Computes the time derivative of the state vector.

  X = [x, x_2, C, mY, Y]
  """
  x, x2, C, mY, Y = X

  dx_dt = alpha_x - sigma_x*x - 2*k1*x**2 + 2*k2*x2
  dx2_dt = k1*x**2 - k2*x2 - a*x2*(ptot - C) + d*C
  dC_dt = a*x2*(ptot - C) - d*C
  dy_dt = kappa*mY - gamma*Y
  dmY_dt = kf*C - sigma*mY

  return np.array([dx_dt, dx2_dt, dC_dt, dmY_dt, dy_dt])


def compute_dX_dt_no_dimer(X, t, alpha_x, sigma_x, a, d, kappa, gamma, kf, sigma, ptot):
  """
  Computes the time derivative of the state vector. In this case there is no dimerization
  of x + x <=> x2, so our state vector only has four things.

  X = [x, C, mY, Y]
  """
  x, C, mY, Y = X

  dx_dt = alpha_x - sigma_x*x - a*x*(ptot - C) + d*C
  dC_dt = a*x*(ptot - C) - d*C
  dy_dt = kappa*mY - gamma*Y
  dmY_dt = kf*C - sigma*mY

  return np.array([dx_dt, dC_dt, dmY_dt, dy_dt])

def simulate(dimerization=True):
  """
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
  """
  t = np.linspace(0, 150, 1001)

  alpha_x = 0.8   # Basal rate of x production.
  sigma_x = 0.7   # Dilution rate of x.
  k1 = 10         # Forward constant for x + x <> x2
  k2 = 5          # Reverse constant for x + x <> x2
  a = 30          # Forward constant for x2 + p <> C
  d = 15          # Reverse constant for x2 + p <> C
  kappa = 0.5    # Rate at which protein Y is translated.
  gamma = 0.2    # Rate at which protein Y dilutes.
  kf = 0.01       # Rate at which mRNA for Y is transcribed.
  sigma = 0.005    # Rate at which mRNA for Y dilutes.
  ptot = 0.0      # Total concentration of RNAP.

  if dimerization:
    # x(0) = 0 (no transcription factor at start)
    X_0 = np.zeros(5)

    X_solution = odeint(
      compute_dX_dt,
      X_0,
      t,
      args=(alpha_x, sigma_x, k1, k2, a, d, kappa, gamma, kf, sigma, ptot))

    x_solution = X_solution[:,0]
    x2_solution = X_solution[:,1]

    plt.plot(t, x_solution, 'b', label='x(t)')
    plt.plot(t, x2_solution, 'g', label='x2(t)')
    plt.plot(t, X_solution[:,2], 'r', label='C(t)')
    plt.plot(t, X_solution[:,3], 'y', label='mY(t)')
    plt.title("x(t) and x2(t) (p_tot = 10.0)")

  else:
    X_0 = np.zeros(4)

    X_solution = odeint(
      compute_dX_dt_no_dimer,
      X_0,
      t,
      args=(alpha_x, sigma_x, a, d, kappa, gamma, kf, sigma, ptot))

    x_solution = X_solution[:,0]
    C_solution = X_solution[:,1]
    plt.plot(t, x_solution, 'b', label='x(t)')
    plt.plot(t, C_solution, 'r', label='C(t)')
    plt.title("x(t) and C(t) (p_tot = 0.0)")

  plt.xlabel('t (sec)')
  plt.ylabel('nM')
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  simulate(dimerization=False)
