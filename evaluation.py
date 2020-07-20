import numpy as np


def dcg(scores):
	"""
		Returns the DCG value of the list of scores.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		
		Returns
		-------
		DCG_val: int
			This is the value of the DCG on the given scores
	"""
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in range(len(scores))
					])


def dcg_k(scores, k):
	"""
		Returns the DCG value of the list of scores and truncates to k values.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		k : int
			In the amount of values you want to only look at for computing DCG
		
		Returns
		-------
		DCG_val: int
			This is the value of the DCG on the given scores
	"""
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in range(len(scores[:k]))
					])


def ideal_dcg(scores):
	"""
		Returns the Ideal DCG value of the list of scores.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		
		Returns
		-------
		Ideal_DCG_val: int
			This is the value of the Ideal DCG on the given scores
	"""
	scores = [score for score in sorted(scores)[::-1]]
	return dcg(scores)


def ideal_dcg_k(scores, k):
    """
        Returns the Ideal DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG
        
        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


if __name__ == "__main__":
    # test functions
    order = np.array([2, 3, 3, 1, 2])

    assert round(dcg(order), 2) == 12.51
    assert round(ideal_dcg(order), 2) == 14.60
