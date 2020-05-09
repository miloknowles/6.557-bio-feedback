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


# def plot_nullclines_protease(K):
#   beta = 10
#   gamma = 0.1

#   P = 1 # Amount of the protease.
#   alpha_B = 1.0 # Production rate of sspB
#   DNA_tot = 1.0 # Concentration of DNA for sspB.
#   RNAP = 1.0 # Concentration of RNA polymerase.
#   Kd_RNAP_sspB = K # Dissociation constant for RNAP binding to the sspB promoter.
#   B = (RNAP * DNA_tot * alpha_B / gamma) / (RNAP + Kd_RNAP_sspB) # Amount of sspB

#   gamma_p = 0.2 # Degradation rate of the protease.
#   Kd_sspB_X1_or_X2 = 1.0 # Dissociation constant for the sspB with either X1 or X2.

#   # gamma_protease = k_f*(alpha / gamma)*P / (K + P)
#   # The protease effectively functions as an extra "gamma" term that degrades X1 and X2.
#   gamma_protease = (P * gamma_p * B) / Kd_sspB_X1_or_X2

#   x1 = np.linspace(0.0, beta/(gamma + gamma_protease) + 1, 1000)
#   x2 = np.linspace(0.0, beta/(gamma + gamma_protease) + 1, 1000)

#   nullcline_x1_dot = (beta / (gamma + gamma_protease)) / (1 + (x2 / K)**2)
#   nullcline_x2_dot = (beta / (gamma + gamma_protease)) / (1 + (x1 / K)**2)

#   plt.plot(nullcline_x1_dot, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
#   plt.plot(x1, nullcline_x2_dot, color="blue", label="dX2/dt nullcline", linestyle="solid")
#   plt.title("Nullclines (beta/gamma={:.02f}, K={})".format(beta/(gamma + gamma_protease), K))
#   plt.legend()
#   plt.xlabel("[X1]")
#   plt.ylabel("[X2]")
#   plt.show()


def plot_nullclines_protease(Kd_repress):
  DNA_tot_rep = 1.0   # Total concentration of DNA for X1 and X2 (includes free DNA and complexes).
  DNA_tot_P = 1.0     # Total concentration of DNA for the protease activator.
  RNAP = 1.0          # (nM) Concentration of RNA polymerase.

  alpha_P = 2.0       # (nM / min) Transcription rate of protease activator P.
  alpha_X = 2.0       # (nM / min) Transription rate of proteins X1 and X2.

  gamma = 0.05         # Dilution rate.
  gamma_P = 0.2        # Degradation rate due to protease.

  Kd_dimer = 0.1                  # (nM) Dimerization reaction for X1 and X2.
  Kd_rnap = 1.0                   # (nM) Binding of the RNAP to promoter for X1 and X2.
  Kd_prot = 2.0                   # (nM) Binding of the sspB protease activator to X1 and X2.

  coeff = 5
  Kd_rnap_dsrA = coeff*Kd_repress     # Dissociation constant for RNAP binding to the dsrA promoter.

  # Production function of X1 and X2.
  K = (Kd_dimer * Kd_repress * (1 + RNAP/Kd_rnap))**0.5
  beta = alpha_X * (DNA_tot_rep * (RNAP / Kd_rnap)) / (1 + RNAP/Kd_rnap)

  # Production of protease activator P.
  P = RNAP * DNA_tot_P * (alpha_P / gamma) / (RNAP + Kd_rnap_dsrA)

  print("K:", K)
  print("beta:", beta)
  print("P:", P)

  # Kd_rnap_dsrA_values = coeff*np.linspace(0.25, 64, 100)
  # plt.plot(Kd_rnap_dsrA_values, RNAP * DNA_tot_P * (alpha_P / gamma) / (RNAP + Kd_rnap_dsrA_values), label="Steady State P")
  # plt.title("Steady State Amount of P")
  # plt.xlabel("Kd of RNAP binding to dsrA promoter")
  # plt.ylabel("Concentration (nM)")
  # plt.show()

  # Dilution AND degradation.
  gamma_total = (gamma + gamma_P*P/Kd_prot)

  x1 = np.linspace(0.0, 1.1 * beta/gamma_total, 1000)
  x2 = np.linspace(0.0, 1.1 * beta/gamma_total, 1000)

  nullcline_x1_dot = (beta / gamma_total) / (1 + (x2 / K)**2)
  nullcline_x2_dot = (beta / gamma_total) / (1 + (x1 / K)**2)

  plt.plot(nullcline_x1_dot, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  plt.plot(x1, nullcline_x2_dot, color="blue", label="dX2/dt nullcline", linestyle="solid")
  plt.title("Nullclines (beta/gamma={:.02f}, K={})".format(beta/gamma_total, K))
  plt.legend()
  plt.xlabel("[X1]")
  plt.ylabel("[X2]")
  plt.show()


if __name__ == "__main__":
  # plot_nullclines_original()

  plot_nullclines_protease(Kd_repress=0.25)
  plot_nullclines_protease(Kd_repress=0.5)
  plot_nullclines_protease(Kd_repress=1.0)
  plot_nullclines_protease(Kd_repress=2.0)
  plot_nullclines_protease(Kd_repress=4.0)
  plot_nullclines_protease(Kd_repress=8.0)
  plot_nullclines_protease(Kd_repress=16.0)
  plot_nullclines_protease(Kd_repress=32.0)
  plot_nullclines_protease(Kd_repress=64.0)

  # plot_nullclines_protease(K=2)
  # plot_nullclines_protease(K=4)
  # plot_nullclines_protease(K=8)
  # plot_nullclines_protease(K=16)
  # plot_nullclines_protease(K=32)

  # plot_nullclines_additional_repressor(K=6)
