import numpy as np
import math, random
from copy import deepcopy
import matplotlib.pyplot as plt


class ReactionConstants(object):
  def __init__(self, Kd_repress=1.0):
    self.alpha_X = 2.0
    self.alpha_P = 2.0

    self.gamma = 0.05
    self.gamma_P = 0.2

    Kd_dimer = 0.1
    self.Kf_dimer = 0.5
    self.Kr_dimer = Kd_dimer*self.Kf_dimer

    Kd_prot = 2.0
    self.Kf_prot = 0.5
    self.Kr_prot = Kd_prot*self.Kf_prot

    self.Kf_repress = 0.1
    self.Kr_repress = Kd_repress*self.Kf_repress

    Kd_rnap_dsrA = 5*Kd_repress
    self.Kf_rnap_dsrA = 0.5
    self.Kr_rnap_dsrA = Kd_rnap_dsrA*self.Kf_rnap_dsrA

  def copy(self):
    return deepcopy(self)


def SSA_with_protease(const, show_plots=False, N=1e4):
  Volume = 1

  # State: [X1, X2, X1d, X2d, C1, C2, D1, D2, P, Dp, DpA, Cp1, Cp2]
  x = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
  time_sequence = []
  state_sequence = []

  t = 0
  num_reactions = 0
  while num_reactions < N:
    time_sequence.append(t)
    state_sequence.append(np.expand_dims(x, 0).copy())

    # Parse the state vector.
    N_X1, N_X2, N_X1d, N_X2d, N_C1, N_C2, N_D1, N_D2, N_P, N_Dp, N_DpA, N_Cp1, N_Cp2 = x

    # Forward/reverse dimerization reactions (symmetric for X1 and X2).
    a_x1_dimer_f = const.Kf_dimer * N_X1 * (N_X1 - 1) / (2 * Volume)
    a_x2_dimer_f = const.Kf_dimer * N_X2 * (N_X2 - 1) / (2 * Volume)
    a_x1_dimer_r = const.Kr_dimer * N_X1d
    a_x2_dimer_r = const.Kr_dimer * N_X2d

    # Forward/reverse complex reactions (X1 and X2 repress each other by sequestering DNA).
    a_C1_f = const.Kf_repress * N_D1 * N_X2d / Volume
    a_C2_f = const.Kf_repress * N_D2 * N_X1d / Volume
    a_C1_r = const.Kr_repress * N_C1
    a_C2_r = const.Kr_repress * N_C2

    # Transcription of proteins X1 and X2.
    a_tx1 = const.alpha_X * N_D1
    a_tx2 = const.alpha_X * N_D2

    # Dilution of proteins X1 and X2.
    a_dilution1 = const.gamma * N_X1
    a_dilution2 = const.gamma * N_X2

    # Forward/reverse binding of RNAP to DNA for the protease P.
    a_DpA_f = const.Kf_rnap_dsrA * N_Dp   # RNAP + DNAp ==> DNApA (activated DNA)
    a_DpA_r = const.Kr_rnap_dsrA * N_DpA  # DNApA ==> RNAP + DNAp (activated DNA releases RNAP).

    # Transcription and dilution of the protease P.
    a_txP = const.alpha_P * N_DpA         # DNApA ==> P + DNApA (one step TX and TL).
    a_dilutionP = const.gamma * N_P       # P ==> 0 (dilution).

    # Forward/reverse binding of protease P to X1 and X2.
    a_Cp1_f = const.Kf_prot * N_X1 * N_P / Volume
    a_Cp2_f = const.Kf_prot * N_X2 * N_P / Volume
    a_Cp1_r = const.Kr_prot * N_Cp1
    a_Cp2_r = const.Kr_prot * N_Cp2

    # Degradation of X1 and X2 from the compex formed with protease P.
    a_degrade1_f = const.gamma_P * N_Cp1
    a_degrade2_f = const.gamma_P * N_Cp2

    rxn_propensities = np.array([
      a_x1_dimer_f,
      a_x2_dimer_f,
      a_x1_dimer_r,
      a_x2_dimer_r,
      a_C1_f,
      a_C2_f,
      a_C1_r,
      a_C2_r,
      a_tx1,
      a_tx2,
      a_dilution1,
      a_dilution2,
      a_DpA_f,
      a_DpA_r,
      a_txP,
      a_dilutionP,
      a_Cp1_f,
      a_Cp2_f,
      a_Cp1_r,
      a_Cp2_r,
      a_degrade1_f,
      a_degrade2_f
    ])

    # Indices: 0  1  2   3   4  5  6  7  8 9  10  11  12
    # Species: X1,X2,X1d,X2d,C1,C2,D1,D2,P,Dp,DpA,Cp1,Cp2
    rxn_stochiometry = [
      np.array([-2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # X1 dimer forward
      np.array([0, -2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # X2 dimer forward
      np.array([2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # X1 dimer reverse
      np.array([0, 2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # X2 dimer reverse
      np.array([0, 0, 0, -1, 1, 0, -1, 0, 0, 0, 0, 0, 0]), # C1 forward
      np.array([0, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0]), # C2 forward
      np.array([0, 0, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0]), # C1 reverse
      np.array([0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0]), # C2 reverse
      np.array([1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]), # Transcription 1
      np.array([0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]), # Transcription 2
      np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # dilution 1
      np.array([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # dilution 2
      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0]), # DpA forward
      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0]), # DpA reverse
      np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0]), # Protease transcription
      np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0]), # Protease dilution
      np.array([-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0]), # Protease complex 1 forward
      np.array([0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]), # Protease complex 2 forward
      np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0]), # Protease complex 1 reverse
      np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1]), # Protease complex 2 reverse
      np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0]), # X1 protease degradation
      np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1]), # X2 protease degradation
    ]

    assert(len(rxn_propensities) == len(rxn_stochiometry))

    # Compute the total probability that we transition out of the current state in the next dt.
    Ki_bar = rxn_propensities.sum()

    r = random.random()
    dt = -math.log(r) / Ki_bar

    # Determine which reaction will happen next.
    p = rxn_propensities / rxn_propensities.sum()
    index_of_next_rxn = np.random.choice(len(rxn_propensities), p=p)

    # Update the state based on the selected reaction.
    x += rxn_stochiometry[index_of_next_rxn]

    # Species may die and go below zero - don't allow to happen.
    x[x < 0] = 0

    t += dt
    num_reactions += 1

  print("Simulated {} reactions".format(len(time_sequence)))
  state_sequence = np.concatenate(state_sequence, axis=0)
  time_sequence = np.array(time_sequence)

  if show_plots:
    plt.plot(time_sequence, state_sequence[:,0], label="Protein X1", linestyle="solid", color="blue", alpha=0.5)
    plt.plot(time_sequence, state_sequence[:,1], label="Protein X2", linestyle="solid", color="red", alpha=0.5)
    plt.plot(time_sequence, state_sequence[:,8], label="Protease P", linestyle="dashed", color="green", alpha=0.5)
    # plt.plot(time_sequence, state_sequence[:,2], label="X1 dimer", linestyle="dashed", color="lightblue", alpha=0.5)
    # plt.plot(time_sequence, state_sequence[:,3], label="X2 dimer", linestyle="dashed", color="lightpink", alpha=0.5)
    plt.title("Stochastic Trajectories of Proteins X1 and X2 (Kd={:.03f})".format(const.Kr_repress / const.Kf_repress))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Molecule Count")
    plt.legend()
    plt.show()

  return time_sequence, state_sequence


def sample_pmf(const, num_reactions=1e5, num_trials=10, num_samples=10000):
  """
  Compute the steady state PMF by sampling from the trajectory at random points in time.
  """
  state_sequence = []
  for i in range(num_trials):
    _, state_sequence_i = SSA_with_protease(const, show_plots=False, N=num_reactions)
    state_sequence.append(state_sequence_i[len(state_sequence_i)//5:,:])

  state_sequence = np.concatenate(state_sequence, axis=0)

  print("Finished sampling, generating histogram...")

  random_indices = np.random.choice(len(state_sequence), size=num_samples)
  random_states = state_sequence[random_indices,:]

  Kd = const.Kr_repress / const.Kf_repress

  bins = [random_states[:,0].max() - random_states[:,0].min(), random_states[:,1].max() - random_states[:,1].min()]
  h = plt.hist2d(random_states[:,0], random_states[:,1], density=True, label="Joint Distribution of X1 and X2", bins=bins)
  plt.colorbar(h[3])
  plt.title("Joint Distribution of X1 and X2 (K = {:.03f})".format(Kd))
  plt.xlabel("X1 Count")
  plt.ylabel("X2 Count")
  plt.savefig("./output/pmf_{}.png".format(Kd))
  plt.show()


if __name__ == "__main__":
  for Kd_repress in [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]:
    const = ReactionConstants(Kd_repress=Kd_repress)
    # SSA_with_protease(const, show_plots=True, N=5e4)
    sample_pmf(const, num_reactions=5e4, num_trials=50, num_samples=20000)

  plt.clf()
