import random


def generate_feedback(true_preference, sigma_feedback):
    noise = random.gauss(0, sigma_feedback)
    feedback = max(0, min(1, true_preference + noise))
    return feedback
