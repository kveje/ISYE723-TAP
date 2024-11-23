import random


def evolve_preference(current_preference, phi, sigma_eta):
    eta = random.gauss(0, sigma_eta)
    new_preference = max(0, min(1, phi * current_preference + eta))
    return new_preference
