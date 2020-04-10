import numpy as np
import math, random
import matplotlib.pyplot as plt

# Compare the
# steady state value of C obtained from the deterministic model to the mean value of
# C obtained from the stochastic model as the volume is changed in the stochastic
# model. What do you observe? You can perform this investigation through numeri-
# cal simulation.


def SSA():
  T = 100

  A_tot = 40
  B_tot = 40
  Ka = 0.1
  Kd = 3.0
  Volume = 1.0

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

  print("Simulated {} reactions".format(len(time_sequence)))
  state_sequence = np.concatenate(state_sequence, axis=0)
  time_sequence = np.array(time_sequence)
  plt.plot(time_sequence, state_sequence[:,0], label="A and B", linestyle="solid", color="blue")
  plt.plot(time_sequence, state_sequence[:,2], label="C", linestyle="solid", color="red")
  plt.title("Stochastic Trajectories for A, B, and C")
  plt.xlabel("Time (seconds)")
  plt.ylabel("Molecule Count")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  SSA()
