import numpy as np
import math, random
import matplotlib.pyplot as plt


def SSA(beta=0.9, gamma=0.1, show_plots=False):
  T = 100

  D1_tot = 100
  D2_tot = 100

  # Having some intial protein seems to help the trajectories "separate".
  X1_init = 1
  X2_init = 1

  Ka_dimer = 10.0
  Kd_dimer = 3.0
  Ka_comp = 5.0
  Kd_comp = 1.0
  # beta = 0.9
  # gamma = 0.1
  Volume = 1

  # State: [X1, X2, X1d, X2d, C1, C2]
  x = np.array([X1_init, X2_init, 0, 0, 0, 0])
  time_sequence = []
  state_sequence = []

  t = 0
  while t < T:
    time_sequence.append(t)
    state_sequence.append(np.expand_dims(x, 0).copy())

    N_X1, N_X2, N_X1d, N_X2d, N_C1, N_C2 = x
    N_D1 = D1_tot - N_C1
    N_D2 = D2_tot - N_C2

    a_x1_dimer_f = Ka_dimer * N_X1 * (N_X1 - 1) / (2 * Volume)
    a_x2_dimer_f = Ka_dimer * N_X2 * (N_X2 - 1) / (2 * Volume)

    # a_x1_dimer_f = max(0, a_x1_dimer_f)
    # a_x2_dimer_f = max(0, a_x2_dimer_f)

    a_x1_dimer_r = Kd_dimer * N_X1d
    a_x2_dimer_r = Kd_dimer * N_X2d

    a_C1_f = Ka_comp * N_D1 * N_X2d / Volume
    a_C2_f = Ka_comp * N_D2 * N_X1d / Volume

    a_C1_r = Kd_comp * N_C1
    a_C2_r = Kd_comp * N_C2

    a_trans1 = beta * N_D1
    a_trans2 = beta * N_D2

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
      a_trans1,
      a_trans2,
      a_decay1,
      a_decay2
    ])

    # print(rxn_propensities)
    # assert(False)

    rxn_stochiometry = [
      np.array([-2, 0, 1, 0, 0, 0]), # X1 dimer forward
      np.array([0, -2, 0, 1, 0, 0]), # X2 dimer forward
      np.array([2, 0, -1, 0, 0, 0]), # X1 dimer reverse
      np.array([0, 2, 0, -1, 0, 0]), # X2 dimer reverse
      np.array([0, 0, 0, -1, 1, 0]), # C1 forward
      np.array([0, 0, -1, 0, 0, 1]), # C2 forward
      np.array([0, 0, 0, 1, -1, 0]), # C1 reverse
      np.array([0, 0, 1, 0, 0, -1]), # C2 reverse
      np.array([1, 0, 0, 0, 0, 0]), # Transcription 1
      np.array([0, 1, 0, 0, 0, 0]), # Transcription 2
      np.array([-1, 0, 0, 0, 0, 0]), # Decay 1
      np.array([0, -1, 0, 0, 0, 0]), # Decay 2
    ]

    assert(len(rxn_propensities) == len(rxn_stochiometry))

    # Compute the total probability that we transition out of the current state in the next dt.
    Ki_bar = rxn_propensities.sum()

    r = random.random()
    dt = (1 / Ki_bar) * math.log(1 / (1 - r))

    # Determine which reaction will happen next.
    p = rxn_propensities / rxn_propensities.sum()
    # print(p)
    index_of_next_rxn = np.random.choice(len(rxn_propensities), p=p)

    # Update the state based on the selected reaction.
    x += rxn_stochiometry[index_of_next_rxn]
    x[x < 0] = 0

    t += dt

  print("Simulated {} reactions".format(len(time_sequence)))
  state_sequence = np.concatenate(state_sequence, axis=0)
  time_sequence = np.array(time_sequence)

  if show_plots:
    plt.plot(time_sequence, state_sequence[:,0], label="Protein A", linestyle="solid", color="blue")
    plt.plot(time_sequence, state_sequence[:,1], label="Protein B", linestyle="solid", color="red")
    # plt.plot(time_sequence, np.ones(len(state_sequence)) * Volume * beta / gamma, linestyle="dashed", color="black")
    plt.title("Stochastic Trajectories Protein A and B (beta/gamma={})".format(beta / gamma))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Molecule Count")
    plt.legend()
    plt.show()

  return time_sequence, state_sequence


def steady_state_distribution(beta, gamma):
  num_trials = 10

  steady_state = []

  for k in range(num_trials):
    print("Simulating SSA #{}".format(k+1))
    time_sequence, state_sequence = SSA(beta=beta, gamma=gamma, show_plots=True)

    # Only include the 2nd half of the states to throw away some "burn in" time.
    burn_in_index = int(len(state_sequence) * 0.8)
    steady_state.append(state_sequence[burn_in_index:])

  steady_state = np.concatenate(steady_state, axis=0)
  plt.hist(steady_state[:,0], bins=20, label="Protein A", density=True, facecolor="red", alpha=0.5)
  plt.hist(steady_state[:,1], bins=20, label="Protein B", density=True, facecolor="blue", alpha=0.5)
  plt.title("Steady State Distribution (beta/gamma={})".format(beta / gamma))
  plt.legend()
  plt.xlabel("Molecule Count")
  plt.ylabel("Probability")
  plt.show()

if __name__ == "__main__":
  # SSA()
  # steady_state_distribution(0.1, 0.1)
  # steady_state_distribution(0.5, 0.1)
  # steady_state_distribution(0.9, 0.1)
  # steady_state_distribution(1.0, 0.1)
  steady_state_distribution(0.9, 0.05)
