""" Placeholder function. """

def test(filepath):
    """ TODO: Replace with SynthID Bayesian/Weighted checker. """
    with open(filepath, "r") as f:
        content = f.read()

    if "integer" in content:
        return False
    return True
