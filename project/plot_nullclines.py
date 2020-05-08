import numpy as np
import matplotlib.pyplot as plt


def plot_nullclines_original(K=6):
  beta = 10
  gamma = 1

  x1 = np.linspace(0.0, beta/gamma + 1, 1000)
  x2 = np.linspace(0.0, beta/gamma + 1, 1000)

  nullcline_x1_dot = (beta / gamma) / (1 + (x2 / K)**2)
  nullcline_x2_dot = (beta / gamma) / (1 + (x1 / K)**2)

  plt.plot(nullcline_x1_dot, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  plt.plot(x1, nullcline_x2_dot, color="blue", label="dX2/dt nullcline", linestyle="solid")
  plt.title("Nullclines for dX1/dt and dX2/dt (beta/gamma={}, K={})".format(beta/gamma, K))
  plt.legend()
  plt.xlabel("[X1]")
  plt.ylabel("[X2]")
  plt.show()


def plot_nullclines_protease(K):
  beta = 10
  gamma = 0.1

  P = 1 # Amount of the protease.
  alpha_B = 1.0 # Production rate of sspB
  DNA_tot = 1.0 # Concentration of DNA for sspB.
  RNAP = 1.0 # Concentration of RNA polymerase.
  Kd_RNAP_sspB = K # Dissociation constant for RNAP binding to the sspB promoter.
  B = (RNAP * DNA_tot * alpha_B / gamma) / (RNAP + Kd_RNAP_sspB) # Amount of sspB

  gamma_p = 0.2 # Degradation rate of the protease.
  Kd_sspB_X1_or_X2 = 1.0 # Dissociation constant for the sspB with either X1 or X2.

  # gamma_protease = k_f*(alpha / gamma)*P / (K + P)
  # The protease effectively functions as an extra "gamma" term that degrades X1 and X2.
  gamma_protease = (P * gamma_p * B) / Kd_sspB_X1_or_X2

  x1 = np.linspace(0.0, beta/(gamma + gamma_protease) + 1, 1000)
  x2 = np.linspace(0.0, beta/(gamma + gamma_protease) + 1, 1000)

  nullcline_x1_dot = (beta / (gamma + gamma_protease)) / (1 + (x2 / K)**2)
  nullcline_x2_dot = (beta / (gamma + gamma_protease)) / (1 + (x1 / K)**2)

  plt.plot(nullcline_x1_dot, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  plt.plot(x1, nullcline_x2_dot, color="blue", label="dX2/dt nullcline", linestyle="solid")
  plt.title("Nullclines (beta/gamma={:.02f}, K={})".format(beta/(gamma + gamma_protease), K))
  plt.legend()
  plt.xlabel("[X1]")
  plt.ylabel("[X2]")
  plt.show()


def plot_nullclines_additional_repressor(K=3):
  beta = 10
  gamma = 1
  R = 1
  Kd = 10*K

  x1 = np.linspace(0.0, beta/gamma + 1, 1000)
  x2 = np.linspace(0.0, beta/gamma + 1, 1000)

  nullcline_x1_dot = (beta / gamma) / (1 + (x2 / K)**2 + (R / Kd))
  nullcline_x2_dot = (beta / gamma) / (1 + (x1 / K)**2 + (R / Kd))

  plt.plot(nullcline_x1_dot, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  plt.plot(x1, nullcline_x2_dot, color="blue", label="dX2/dt nullcline", linestyle="solid")
  plt.title("Nullclines for dX1/dt and dX2/dt (beta/gamma={}, K={})".format(beta/gamma, K))
  plt.legend()
  plt.xlabel("[X1]")
  plt.ylabel("[X2]")
  plt.show()


if __name__ == "__main__":
  # plot_nullclines_original()

  plot_nullclines_protease(K=1)
  plot_nullclines_protease(K=3)
  plot_nullclines_protease(K=6)
  plot_nullclines_protease(K=9)
  plot_nullclines_protease(K=12)

  # plot_nullclines_additional_repressor(K=6)
