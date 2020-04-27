from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math


def dynamics_connected(state, t, p_tot, downstream_protein):
  """
  Compute the time-derivatives of an acivator-repressor clock connected to a downstream system.

  State:
    mA, mB, A, B, C
  """
  assert(downstream_protein in ["A", "B"])

  # Define static parameters.
  sigma_A = sigma_B = 1.0 # 1 / hrs
  alpha_A = 250 # nM / hrs
  alpha_B = 30 # nM / hrs
  alpha_A0 = 0.04 # nM / hrs
  alpha_B0 = 0.004 # nM / hrs
  K_A = K_B = 1 # nM
  kappa_A = kappa_B = 1 # 1 / hrs
  gamma_A = 1 # 1 / hrs
  gamma_B = 0.5 # 1 / hrs
  n = 2
  m = 4

  # We know that k_off >> gamma_A and gamma_B.
  k_off = 10 * gamma_A
  k_on = 10 * gamma_A

  def F1(A, B):
    return (alpha_A * (A / K_A)**n + alpha_A0) / (1 + (A / K_A)**n + (B / K_B)**m)

  def F2(A):
    return (alpha_B * (A / K_A)**n + alpha_B0) / (1 + (A / K_A)**n)


  # Extract components of the state.
  mA, mB, A, B, C = state

  # Compute time-derivative of each component.
  dmA_dt = F1(A, B) - sigma_A*mA
  dmB_dt = F2(A) - sigma_B*mB

  # Conservation law: p_tot = p + C
  p = p_tot - C

  if downstream_protein == "A":
    dA_dt = kappa_A*mA - gamma_A*A - k_on*n*A*p + n*k_off*C
    dB_dt = kappa_B*mB - gamma_B*B
    dC_dt = k_on*n*A*p - k_off*C
  else:
    dA_dt = kappa_A*mA - gamma_A*A
    dB_dt = kappa_B*mB - gamma_B*B - k_on*n*B*p + n*k_off*C
    dC_dt = k_on*n*B*p - k_off*C

  return np.array([dmA_dt, dmB_dt, dA_dt, dB_dt, dC_dt])


def run_simulation():
  initial_state = np.zeros(5)

  timesteps = np.linspace(0, 50, 10000)

  p_tot = 0.05 # nM
  downstream_protein = "B"

  for p_tot in [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]:
    # Simulate the connected system dynamics.
    traj_connected = odeint(
      dynamics_connected,
      initial_state,
      timesteps,
      args=(p_tot, downstream_protein))

    # Plot a comparison of isolated and connected.
    plt.plot(timesteps, traj_connected[:,2], 'blue', label='A(t) CONNECTED', linestyle="solid")
    plt.plot(timesteps, traj_connected[:,3], 'red', label="B(t) CONNECTED", linestyle="solid")
    plt.title("Trajectory of Connected Activator-Repressor Clock (p_tot={})".format(p_tot))

    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (nM)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
  run_simulation()
