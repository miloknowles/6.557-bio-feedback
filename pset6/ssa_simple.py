import numpy as np
import math, random
import matplotlib.pyplot as plt

# Compare the
# steady state value of C obtained from the deterministic model to the mean value of
# C obtained from the stochastic model as the volume is changed in the stochastic
# model. What do you observe? You can perform this investigation through numeri-
# cal simulation.


def SSA():
  T = 10

  A_tot = 100
  B_tot = 100
  Ka = 0.1
  Kd = 0.5
  Volume = 2.0

  # Start out with A and B but not C.
  x = np.array([A_tot, B_tot, 0])
  time_sequence = []
  state_sequence = []

  t = 0
  while t < T:
    time_sequence.append(t)
    state_sequence.append(np.expand_dims(x, 0).copy())

    Na, Nb, Nc = x
    forw_propensity = Ka*Na*Nb / Volume
    back_propensity = Kd*Nc

    # Compute the total probability that we transition out of the current state in the next dt.
    # Two possible next states, from forward or backward reactions.
    Ki_bar = (forw_propensity + back_propensity)

    r = random.random()
    dt = (1 / Ki_bar) * math.log(1 / (1 - r))

    # Determine which reaction will happen next.
    forward_reaction_is_next = random.random() < (forw_propensity / Ki_bar)

    if forward_reaction_is_next:
      assert(Na > 0 and Nb > 0)
      x[0] -= 1
      x[1] -= 1
      x[2] += 1
    else:
      x[0] += 1
      x[1] += 1
      x[2] -= 1

    t += dt

  # Compute the deterministic concentrations for A, B, and C.
  a = Ka / Volume
  d = Kd / Volume
  Qb = (a*A_tot/Volume + a*B_tot/Volume + d)
  C_ss_conc = (Qb + math.sqrt(Qb**2 - 4*a*a*A_tot*B_tot/(Volume*Volume))) / 2
  C_ss_mole = Volume * C_ss_conc
  print("Expect {} molecules C".format(C_ss_mole))

  print("Simulated {} reactions".format(len(time_sequence)))
  state_sequence = np.concatenate(state_sequence, axis=0)
  time_sequence = np.array(time_sequence)
  plt.plot(time_sequence, state_sequence[:,0], label="A and B", linestyle="solid", color="blue")
  plt.plot(time_sequence, state_sequence[:,2], label="C", linestyle="solid", color="red")
  # plt.plot(time_sequence, np.ones(len(time_sequence))*C_ss_mole, linestyle="dashed", color="black", label="Deterministic Steady State")
  plt.title("Stochastic Trajectories for A, B, and C (Volume={})".format(Volume))
  plt.xlabel("Time (seconds)")
  plt.ylabel("Molecule Count")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  SSA()
