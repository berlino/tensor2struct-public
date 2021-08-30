import scipy.special as misc
import edit_distance
import numpy as np
import copy


def compute_hamming_distance(l1, l2):
    assert len(l1) == len(l2)
    dist = 0
    ops = []
    for i, (i1, i2) in enumerate(zip(l1, l2)):
        if i1 != i2:
            dist += 1
            ops.append(("replace", i, i + 1, i, i + 1))
        else:
            ops.append(("equal", i, i + 1, i, i + 1))
    return dist, ops


def compute_levenshtein_distance(l1, l2):
    sm = edit_distance.SequenceMatcher(a=l1, b=l2)
    return sm.distance(), sm.get_opcodes()


def ComputeHammingCDF(len_target, temprature, vocab):
    max_edits = len_target + 1  # we allow between 0 and len_target subs
    a = np.zeros(max_edits)
    for n_subs in range(max_edits):
        count_n_subs = []
        tot_edits = misc.comb(len_target, n_subs)
        a[n_subs] = np.log(tot_edits) + n_subs * np.log(
            len(vocab) - 1
        )  # number of sequences: tot_edits * (N-1) ^ n_subs
        a[n_subs] += -n_subs / float(temprature) * np.log(
            len(vocab) - 1
        ) - n_subs / float(
            temprature
        )  # tot_edits * (N-1) ^ n_subs * ((N-1)e) ^ (-n_subs / T)
    p_subs = a - np.max(a)
    p_subs = np.exp(p_subs)
    p_subs /= np.sum(p_subs)
    p_hamming_cdf = np.cumsum(p_subs)
    return p_hamming_cdf


def SubstitutionSampling(s, temprature, hamming_cdf, vocab):
    """
    Sample one sequence from the vicinity of a given target sequence s.
        A string t is sampled proportionally to exp{-hamming_distance(t, s) / temprature}

    Args:
        s: numpy array of a sequence which is output of a seq2seq/crf model e.g. POS tag sequence
        temprature: temprature of sampling
        hamming_cdf: precomputed edit CDF
        vocab: the vocabulary elements that are allowed for substitution

        Returns:
        numpy array of a sampled sequence t
    """
    # assert min(vocab) >= 0
    len_target = len(s) - 1
    p_hamming_cdf = hamming_cdf[len_target]
    # sample
    rand_n_subs = np.sum(np.random.rand() >= p_hamming_cdf)
    # apply changes
    t = copy.copy(s)
    perm = np.random.permutation(len_target)
    subs = perm[:rand_n_subs]
    for i in subs:
        while True:
            rand_char = vocab[np.random.randint(len(vocab))]
            if not t[i] == rand_char:
                break
        t[i] = rand_char
    return t


if __name__ == "__main__":
    vocab = ["walk", "jump", "turn_left", "turn_right"]

    # one can precompute hamming CDFs for different sequence lengths
    subs_cdf = []
    for len_target in range(10):  # assuming maximum length is 10
        p_subs_cdf = ComputeHammingCDF(len_target, temprature=0.9, vocab=vocab)
        subs_cdf.append(p_subs_cdf)

    for _ in range(10):
        sample = SubstitutionSampling(
            ["walk", "turn_left", "turn_right", "jump"], 0.9, subs_cdf, vocab
        )
        print(sample)
