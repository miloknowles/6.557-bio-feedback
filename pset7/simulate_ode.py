from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math


def dynamics_connected(state, t, w, gamma, k_off, k_on, p_tot):
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


def dynamics_isolated(state, t, w, gamma, k_off, k_on, p_tot):
  """
  Computes time derivates of the state (X and C).

  dX/dt = k(t) - gamma*X
  dC/dt = 0

  Where k(t) = gamma * (1 + sin(wt))
  """
  X, _ = state

  k_of_t = gamma * (1 + np.sin(w * t))
  dX_dt = k_of_t - gamma*X

  return np.array([dX_dt, 0])


def simulate():
  initial_state = np.zeros(2)
  w = 0.005

  timesteps = np.linspace(0, 10*2*math.pi / w, 100000)

  gamma = 0.01  # 1 / min
  k_on = 10     # 1 / (min * nM)
  k_off = 10    # 1 / min
  p_tot = 100   # nM

  # Simulate the isolated system dynamics.
  traj_isolated = odeint(
    dynamics_isolated,
    initial_state,
    timesteps,
    args=(w, gamma, k_off, k_on, p_tot))

  # Simulate the connected system dynamics.
  traj_connected = odeint(
    dynamics_connected,
    initial_state,
    timesteps,
    args=(w, gamma, k_off, k_on, p_tot))

  # Plot a comparison of isolated and connected.
  plt.plot(timesteps, traj_connected[:,0], 'blue', label='X(t) CONNECTED', linestyle="solid")
  # plt.plot(timesteps, traj_connected[:,1], 'r', label='C(t) CONNECTED', linestyle="dashed")
  plt.plot(timesteps, traj_isolated[:,0], 'black', label='X(t) ISOLATED', linestyle="dashed")
  # plt.plot(timesteps, traj_isolated[:,1], 'r', label='C(t) ISOLATED', linestyle="solid")
  plt.title("Comparison of Isolated and Connected Systems (w={})".format(w))

  plt.xlabel('Time (min)')
  plt.ylabel('Concentration (nM)')
  plt.legend()
  plt.grid()
  plt.show()


def frequency_amplitude_plot():
  initial_state = np.zeros(2)
  gamma = 1.   # 1 / min
  k_on = 10     # 1 / (min * nM)
  k_off = 10    # 1 / min
  p_tot = 100   # nM

  ws = []
  amps_iso = []
  amps_con = []

  for w in np.logspace(-5, -2, num=50):
    timesteps = np.linspace(0, 20*2*math.pi / w, 10000)

    # Simulate the isolated system dynamics.
    traj_isolated = odeint(
      dynamics_isolated,
      initial_state,
      timesteps,
      args=(w, gamma, k_off, k_on, p_tot))

    # Simulate the connected system dynamics.
    traj_connected = odeint(
      dynamics_connected,
      initial_state,
      timesteps,
      args=(w, gamma, k_off, k_on, p_tot))

    # Extract the amplitude: grab the max and min once trajectory has stabilized.
    offset = len(traj_isolated) // 10
    traj_isolated = traj_isolated[-offset:,0]
    traj_connected = traj_connected[-offset:,0]

    amp_iso = traj_isolated.max() - traj_isolated.min()
    amp_con = traj_connected.max() - traj_connected.min()

    ws.append(w)
    amps_iso.append(amp_iso)
    amps_con.append(amp_con)

    print("w={} ==> AMPLITUDES: isolated={} connected={}".format(w, amp_iso, amp_con))

    # Plot a comparison of isolated and connected.
    # plt.plot(timesteps[-offset:], traj_connected[:,0], 'blue', label='X(t) CONNECTED', linestyle="solid")
    # plt.plot(timesteps[-offset:], traj_isolated[:,0], 'black', label='X(t) ISOLATED', linestyle="dashed")
    # plt.title("Comparison of Isolated and Connected Systems (w={})".format(w))

    # plt.xlabel('Time (min)')
    # plt.ylabel('Concentration (nM)')
    # plt.legend()
    # plt.grid()
    # plt.show()

  plt.plot(ws, amps_iso, "black", label="Isolated", linestyle="dashed")
  plt.plot(ws, amps_con, "blue", label="Connected", linestyle="solid")
  plt.xlabel("Frequency (Hz)")
  plt.ylabel("Amplitude (nM)")
  plt.xscale("log")
  plt.legend()
  plt.title("Frequency-Amplitude Plot")
  plt.show()


if __name__ == "__main__":
  # simulate()
  frequency_amplitude_plot()
