import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """
  R_z_pos = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
  R_z_neg = R_z_pos.T
  [U, S, Vh] = np.linalg.svd(E)

  t_r_pair = {}
  t_r_pair["T"] = U[:,-1]
  t_r_pair["R"] = U @ R_z_pos.T @ Vh
  transform_candidates.append(t_r_pair)

  t_r_pair = {}
  t_r_pair["T"] = U[:,-1]
  t_r_pair["R"] = U @ R_z_neg.T @ Vh
  transform_candidates.append(t_r_pair)

  t_r_pair = {}
  t_r_pair["T"] = -U[:,-1]
  t_r_pair["R"] = U @ R_z_pos.T @ Vh
  transform_candidates.append(t_r_pair)

  t_r_pair = {}
  t_r_pair["T"] = -U[:,-1]
  t_r_pair["R"] = U @ R_z_neg.T @ Vh
  transform_candidates.append(t_r_pair)

  """ END YOUR CODE
  """
  return transform_candidates