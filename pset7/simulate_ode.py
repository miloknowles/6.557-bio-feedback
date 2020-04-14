from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def compute_time_derivatives(state, t, w, gamma, k_off, k_on, p_tot):
  """
  Computes time derivates of the state (X and C).

  dX/dt = k(t) - gamma*X + k_off*C - k_on*(p_tot - C)*X
  dC/dt = -k_off*C + k_on(p_tot - C)*X

  Where k(t) = gamma * (1 + sin(wt))
  """
  X, C = state

  k_of_t = gamma * (1 + np.sin(w * t))
  dX_dt = k_of_t - gamma*X + k_off*C - k_on*(p_tot - C)*X
  dC_dt = -k_off*C + k_on*(p_tot - C)*X

  return np.array([dX_dt, dC_dt])


def simulate():
  timesteps = np.linspace(0, 10000, 100000)
  initial_state = np.zeros(2)
  w = 0.005
  gamma = 0.01  # 1 / min
  k_on = 10     # 1 / (min * nM)
  k_off = 10    # 1 / min
  p_tot = 100   # nM

  state_trajectory = odeint(
    compute_time_derivatives,
    initial_state,
    timesteps,
    args=(w, gamma, k_off, k_on, p_tot))

  x_solution = state_trajectory[:,0]
  C_solution = state_trajectory[:,1]
  plt.plot(timesteps, x_solution, 'b', label='X(t)')
  plt.plot(timesteps, C_solution, 'r', label='C(t)')
  plt.title("X(t) and C(t)")

  plt.xlabel('Time (min)')
  plt.ylabel('Concentration (nM)')
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  simulate()
