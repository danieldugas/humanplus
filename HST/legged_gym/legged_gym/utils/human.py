import numpy as np
import torch

def load_single_target_jt(device, file, offset):
    one_target_jt = np.load(f"/home/daniel/Code/humanplus/HST/legged_gym/data/{file}").astype(np.float32)
    one_target_jt = torch.from_numpy(one_target_jt).to(device) # (T, 19)
    target_jt = one_target_jt.unsqueeze(0) # (1, T, 19)
    target_jt += offset

    size = torch.tensor([one_target_jt.shape[0]]).to(device) # [1]
    return target_jt, size

def load_target_jt(device, file, offset):
    import glob
    paths = glob.glob("/home/daniel/Insync/daniel@dugas.ch/Google Drive/AI/HumaNoid/ACCAD_H1_PARTIAL/*.npy")
    sequences = []
    sequence_lengths = []
    _19 = 19 # number of dofs
    for path in paths:
        print("{}/{}".format(len(sequences), len(paths)))
        sequence = np.load(path).astype(np.float32)
        sequences.append(sequence)
        sequence_lengths.append(len(sequence))
    # to fixed size array
    max_seq_len = np.max(sequence_lengths)
    n_sequences = len(sequences)
    np_sequences = np.zeros((n_sequences, max_seq_len, _19)).astype(np.float32)
    for i, sequence in enumerate(sequences):
        np_sequences[i, :len(sequence), :] = sequence
    np_sequence_lengths = np.array(sequence_lengths)
    # to tensors
    target_jt = torch.from_numpy(np_sequences).to(device)
    target_jt += offset
    size = torch.from_numpy(np_sequence_lengths).to(device)
    return target_jt, size

if __name__ == "__main__":
    device = 'cuda:0' # 'cpu'
    offset = torch.tensor([[ 0.0000,  0.0000, -0.3490,  0.6980, -0.3490,  0.0000,  0.0000, -0.3490,
          0.6980, -0.3490,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000,  0.0000]]).to(device)
    file = 'ACCAD_walk_10fps.npy'
    target_jt, size = load_target_jt(device, file, offset)
    print("Done.")
