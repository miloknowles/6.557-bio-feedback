import numpy as np
import matplotlib.pyplot as plt

def plot_nullclines_original(Kd_repress):
  DNA_tot_rep = 1.0   # Total concentration of DNA for X1 and X2 (includes free DNA and complexes).
  RNAP = 1.0          # (nM) Concentration of RNA polymerase.

  alpha_X = 1.0       # (nM / min) Transription rate of proteins X1 and X2.

  gamma = 0.1         # Dilution rate.

  Kd_dimer = 1.0                  # (nM) Dimerization reaction for X1 and X2.
  Kd_rnap = 1.0                   # (nM) Binding of the RNAP to promoter for X1 and X2.

  # Production function of X1 and X2.
  K = (Kd_dimer * Kd_repress * (1 + RNAP/Kd_rnap))**0.5
  beta = alpha_X * (DNA_tot_rep * (RNAP / Kd_rnap)) / (1 + RNAP/Kd_rnap)

  x1 = np.linspace(0.0, 1.1 * beta/gamma, 1000)
  x2 = np.linspace(0.0, 1.1 * beta/gamma, 1000)

  nullcline_x1_dot_orig = (beta / gamma) / (1 + (x2 / K)**2)
  nullcline_x2_dot_orig = (beta / gamma) / (1 + (x1 / K)**2)

  # Plot the original system nullclines.
  plt.plot(nullcline_x1_dot_orig, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  plt.plot(x1, nullcline_x2_dot_orig, color="blue", label="dX2/dt nullcline", linestyle="solid")
  plt.title("Nullclines: Toggle Switch (K={})".format(Kd_repress))
  plt.legend()
  plt.xlabel("[X1]")
  plt.ylabel("[X2]")
  plt.show()


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

  nullcline_x1_dot_orig = (beta / gamma) / (1 + (x2 / K)**2)
  nullcline_x2_dot_orig = (beta / gamma) / (1 + (x1 / K)**2)

  # Plot the IFFL system nullclines.
  plt.clf()
  plt.plot(nullcline_x1_dot, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  plt.plot(x1, nullcline_x2_dot, color="blue", label="dX2/dt nullcline", linestyle="solid")
  plt.title("Nullclines: Toggle Switch + IFFL (Kd={})".format(Kd_repress))
  plt.legend()
  plt.xlabel("[X1]")
  plt.ylabel("[X2]")
  plt.savefig("./output/nullclines_{}.png".format(Kd_repress))
  plt.show()

  # Plot the original system nullclines.
  # plt.plot(nullcline_x1_dot_orig, x2, color="red", label="dX1/dt nullcline", linestyle="dashed")
  # plt.plot(x1, nullcline_x2_dot_orig, color="blue", label="dX2/dt nullcline", linestyle="solid")
  # plt.title("Nullclines: Toggle Switch (K={})".format(K))
  # plt.legend()
  # plt.xlabel("[X1]")
  # plt.ylabel("[X2]")
  # plt.show()


if __name__ == "__main__":
  # Kd_repress_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
  # for Kd_repress in Kd_repress_values:
  plot_nullclines_protease(Kd_repress=512.0)

  # plot_nullclines_original(Kd_repress=1.0)
  # plot_nullclines_original(Kd_repress=2.0)
  # plot_nullclines_original(Kd_repress=4.0)
