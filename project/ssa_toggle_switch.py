import numpy as np
import math, random
import matplotlib.pyplot as plt


def SSA(beta=1.0, gamma=0.1, show_plots=False, N=1e4):
  Ka_dimer = Kd_dimer = Ka_comp = Kd_comp = 1.0
  Volume = 1

  # State: [X1, X2, X1d, X2d, C1, C2, D1, D2]
  x = np.array([0, 0, 0, 0, 0, 0, 1, 1])
  time_sequence = []
  state_sequence = []

  t = 0
  num_reactions = 0
  while num_reactions < N:
    time_sequence.append(t)
    state_sequence.append(np.expand_dims(x, 0).copy())

    N_X1, N_X2, N_X1d, N_X2d, N_C1, N_C2, N_D1, N_D2 = x

    # Forward/reverse dimerization reactions (symmetric for X1 and X2).
    a_x1_dimer_f = Ka_dimer * N_X1 * (N_X1 - 1) / (2 * Volume)
    a_x2_dimer_f = Ka_dimer * N_X2 * (N_X2 - 1) / (2 * Volume)
    a_x1_dimer_r = Kd_dimer * N_X1d
    a_x2_dimer_r = Kd_dimer * N_X2d

    # Forward/reverse complex reactins (X1 and X2 repress each other by sequestering DNA).
    a_C1_f = Ka_comp * N_D1 * N_X2d / Volume
    a_C2_f = Ka_comp * N_D2 * N_X1d / Volume
    a_C1_r = Kd_comp * N_C1
    a_C2_r = Kd_comp * N_C2

    a_tx1 = beta * N_D1
    a_tx2 = beta * N_D2

    a_decay1 = gamma * N_X1
    a_decay2 = gamma * N_X2

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
      a_decay1,
      a_decay2
    ])

    # Species: X1,X2,X1d,X2d,C1,C2,D1,D2
    rxn_stochiometry = [
      np.array([-2, 0, 1, 0, 0, 0, 0, 0]), # X1 dimer forward
      np.array([0, -2, 0, 1, 0, 0, 0, 0]), # X2 dimer forward
      np.array([2, 0, -1, 0, 0, 0, 0, 0]), # X1 dimer reverse
      np.array([0, 2, 0, -1, 0, 0, 0, 0]), # X2 dimer reverse
      np.array([0, 0, 0, -1, 1, 0, -1, 0]), # C1 forward
      np.array([0, 0, -1, 0, 0, 1, 0, -1]), # C2 forward
      np.array([0, 0, 0, 1, -1, 0, 1, 0]), # C1 reverse
      np.array([0, 0, 1, 0, 0, -1, 0, 1]), # C2 reverse
      np.array([1, 0, 0, 0, 0, 0,  0, 0]), # Transcription 1
      np.array([0, 1, 0, 0, 0, 0,  0, 0]), # Transcription 2
      np.array([-1, 0, 0, 0, 0, 0, 0, 0]), # Decay 1
      np.array([0, -1, 0, 0, 0, 0, 0, 0]), # Decay 2
    ]

    assert(len(rxn_propensities) == len(rxn_stochiometry))

    # Compute the total probability that we transition out of the current state in the next dt.
    Ki_bar = rxn_propensities.sum()

    r = random.random()
    # dt = (1 / Ki_bar) * math.log(1 / (1 - r))
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
    # plt.plot(time_sequence, np.ones(len(state_sequence)) * Volume * beta / gamma, linestyle="dashed", color="black")
    plt.title("Stochastic Trajectories of Proteins X1 and X2 (beta/gamma={})".format(beta / gamma))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Molecule Count")
    plt.legend()
    plt.show()

  return time_sequence, state_sequence


def sample_pmf(beta=10, gamma=1, num_samples=1000, N=1e6):
  time_sequence, state_sequence = SSA(beta=beta, gamma=gamma, show_plots=False, N=N)

  random_indices = np.random.choice(len(state_sequence), size=num_samples)
  random_states = state_sequence[random_indices,:]

  bins = [random_states[:,0].max() - random_states[:,0].min(), random_states[:,1].max() - random_states[:,1].min()]
  h = plt.hist2d(random_states[:,0], random_states[:,1], density=True, label="Joint Distribution of X1 and X2", bins=bins)
  plt.colorbar(h[3])
  plt.title("Joint Distribution of X1 and X2 (beta/gamma={})".format(beta / gamma))
  plt.xlabel("X1 Count")
  plt.ylabel("X2 Count")
  plt.show()


if __name__ == "__main__":
  # beta / gamma << 1 ==> unimodal stationary distribution
  # beta / gamma >> 1 ==> bimodal stationary distribution
  # SSA(beta=10, gamma=1, show_plots=True, N=1e5)
  sample_pmf(beta=100, gamma=10, num_samples=10000, N=1e6)
