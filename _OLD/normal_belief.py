import numpy as np


class NormalBelief:
    def __init__(self, num_individuals: int):
        # Number of individuals
        self.num_individuals = num_individuals

        # Initialize mean and variance matrices
        self.mean = np.zeros((num_individuals, num_individuals))
        self.previous_mean = np.zeros((num_individuals, num_individuals))
        self.variance = np.ones((num_individuals, num_individuals))

        # Guessed initial estimates for sigma_w and sigma_f
        self.estimated_sigma_w = 1
        self.estimated_sigma_f = 1

        # Feedback count for each pair
        self.num_feedback = np.zeros((num_individuals, num_individuals))

        # Variables for estimating sigma_f
        self.sum_residuals = np.zeros((num_individuals, num_individuals))
        self.sum_squared_residuals = np.zeros((num_individuals, num_individuals))

        # Variables for estimating sigma_w
        self.sum_squared_deltas = np.zeros((num_individuals, num_individuals))

        # Forgetting factor for variance estimation
        self.alpha = 0.9

    def update(self, feedback_matrix: np.ndarray):
        """
        Update mean and variance based on feedback using Kalman Filter.
        """
        # Make feedback filter
        feedback_filter = ~np.isnan(feedback_matrix)
        np.fill_diagonal(feedback_filter, False)
        self.update_feedback_count(feedback_filter)

        # Update estimated values for sigma_f and sigma_w
        sigma_f = self.update_sigma_f(feedback_matrix, feedback_filter)
        sigma_w = self.update_sigma_w()

        # Phase 1: Prediction
        predicted_mean = self.mean
        predicted_variance = self.variance + (sigma_w**2 * feedback_filter)

        # Phase 2: Update
        # Calculate innovation and Kalman gain
        innovation = (feedback_matrix - predicted_mean) * feedback_filter
        innovation = np.nan_to_num(innovation, nan=0)
        kalman_gain = predicted_variance / (predicted_variance + sigma_f**2)

        # Update mean and variance
        self.mean = predicted_mean + kalman_gain * innovation
        self.variance = (
            self.variance * (1 - feedback_filter)
            + (1 - kalman_gain) * predicted_variance
        )
        # Avoid updating variance for self-feedback
        np.fill_diagonal(self.variance, 0)
        np.fill_diagonal(self.mean, 0)

        # Step 6: Update sum_squared_deltas for sigma_w estimation
        delta_means = (self.mean - predicted_mean) * feedback_filter
        self.sum_squared_deltas += delta_means**2

        # Update previous mean
        self.previous_mean = self.mean.copy()

    def update_feedback_count(self, feedback_filter: np.ndarray):
        """
        Update the feedback count for each pair.
        """
        self.num_feedback += feedback_filter

    def update_sigma_f(self, feedback_matrix: np.ndarray, feedback_filter: np.ndarray):
        # Compute residuals where feedback is observed
        residuals = feedback_filter * (feedback_matrix - self.mean)
        residuals = np.nan_to_num(residuals, nan=0)
        # Update sums
        self.sum_residuals += residuals
        self.sum_squared_residuals += residuals**2

        # Calculate variance of residuals
        valid_counts = self.num_feedback > 1

        if np.any(valid_counts):
            variance_residuals = np.zeros_like(self.sum_squared_residuals)
            variance_residuals[valid_counts] = (
                self.sum_squared_residuals[valid_counts]
                - (self.sum_residuals[valid_counts] ** 2)
                / self.num_feedback[valid_counts]
            ) / (self.num_feedback[valid_counts] - 1)

            # Ensure variances are non-negative
            variance_residuals = np.maximum(variance_residuals, 0)

            # Update estimated_sigma_f as the mean of variance_residuals
            self.estimated_sigma_f = np.mean(variance_residuals[valid_counts])

            return self.estimated_sigma_f
        else:
            # Use initial guess if no valid data
            return self.estimated_sigma_f

    def update_sigma_w(self):
        # Use accumulated sum_squared_deltas and num_feedback
        valid_counts = self.num_feedback > 1

        if np.any(valid_counts):
            # Compute deltas where feedback is observed
            variance_deltas = np.zeros_like(self.sum_squared_deltas)
            variance_deltas[valid_counts] = self.sum_squared_deltas[valid_counts] / (
                self.num_feedback[valid_counts] - 1
            )

            # Update estimated_sigma_w as the mean of variance_deltas
            self.estimated_sigma_w = np.mean(variance_deltas[valid_counts])

            return self.estimated_sigma_w
        else:
            # Use initial guess if no valid data
            return self.estimated_sigma_w

    def get_means(self):
        """
        Retrieve the current mean estimates.
        """
        return self.mean

    def get_variances(self):
        """
        Retrieve the current variance estimates.
        """
        return self.variance


if __name__ == "__main__":
    # Test the NormalBelief class
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming the NormalBelief class has been defined as per your latest code
    # from normal_belief import NormalBelief  # Uncomment if NormalBelief is in a separate file

    # Parameters
    num_individuals = 100  # Small number of individuals for testing
    num_periods = 1000  # Number of time periods to simulate
    true_sigma_w = 0.3  # True process noise standard deviation
    true_sigma_f = 0.3  # True observation noise standard deviation

    # Initialize the NormalBelief object without knowing true sigma_w and sigma_f
    belief = NormalBelief(num_individuals=num_individuals)

    # Initialize true preferences
    true_preferences = np.random.uniform(-1, 1, size=(num_individuals, num_individuals))

    # For simplicity, set diagonal to zero (no self-preference)
    np.fill_diagonal(true_preferences, 0)

    # Store estimated and true preferences over time for plotting
    estimated_means_over_time = []
    true_preferences_over_time = []
    variances_over_time = []
    estimated_sigma_f = []
    estimated_sigma_w = []

    # Simulate over multiple periods
    for period in range(num_periods):
        # Store the true preferences
        true_preferences_over_time.append(true_preferences.copy())
        # For this test, assume we receive feedback for all pairs
        # In practice, feedback may be sparse
        # Optionally, create a feedback filter to simulate missing feedback
        feedback_filter = np.random.choice(
            [0, 1], size=(num_individuals, num_individuals), p=[0.9, 0.1]
        )
        feedback_filter = feedback_filter.astype(bool)
        # Simulate the evolution of true preferences
        # Let's assume that individuals interact with all others in every period
        # Process noise affects all pairs
        process_noise = np.random.normal(
            0, true_sigma_w, size=(num_individuals, num_individuals)
        )
        true_preferences += process_noise * feedback_filter
        np.fill_diagonal(true_preferences, 0)  # No self-preference

        # Simulate feedback observations with observation noise
        observation_noise = np.random.normal(
            0, true_sigma_f, size=(num_individuals, num_individuals)
        )
        feedback_matrix = true_preferences + observation_noise

        np.fill_diagonal(feedback_filter, False)  # No self-feedback
        feedback_matrix *= feedback_filter

        # Update the belief estimates
        belief.update(feedback_matrix)

        estimated_sigma_w.append(belief.estimated_sigma_w)
        estimated_sigma_f.append(belief.estimated_sigma_f)

        # Store the estimated means
        estimated_means_over_time.append(belief.get_means().copy())
        variances_over_time.append(belief.get_variances().copy())

    # Convert stored data to numpy arrays for easier indexing
    estimated_means_over_time = np.array(estimated_means_over_time)
    true_preferences_over_time = np.array(true_preferences_over_time)
    variances_over_time = np.array(variances_over_time)

    # Plot the true and estimated preferences over time for a specific pair
    pair_indices = [(0, 1)]  # Pairs to plot

    for i, j in pair_indices:
        plt.figure()
        plt.plot(
            range(num_periods),
            true_preferences_over_time[:, i, j],
            label=f"True Preference",
            linestyle="--",
        )
        # Add the variance as a shaded region
        plt.fill_between(
            range(num_periods),
            estimated_means_over_time[:, i, j] - variances_over_time[:, i, j],
            estimated_means_over_time[:, i, j] + variances_over_time[:, i, j],
            alpha=0.2,
        )
        plt.plot(
            range(num_periods),
            estimated_means_over_time[:, i, j],
            label=f"Estimated Preference",
        )
        plt.xlabel("Time Period")
        plt.ylabel("Preference")
        plt.title(f"Preference Estimates for Pair ({i},{j})")
        plt.legend()
        plt.show()

    # Print final estimated variances and noise estimates
    print("Final Estimated Variances:")
    print(belief.get_variances())

    print(f"Estimated sigma_f: {belief.estimated_sigma_f}")
    print(f"Estimated sigma_f^2: {belief.estimated_sigma_f**2}")
    print(f"Estimated sigma_w: {belief.estimated_sigma_w}")
    print(f"Estimated sigma_w^2: {belief.estimated_sigma_w**2}")

    print(f"True sigma_f^2: {true_sigma_f**2}")
    print(f"True sigma_w^2: {true_sigma_w**2}")

    # Print estimated feedback counts
    print("Feedback Counts:")
    print(belief.num_feedback)

    # Plot the estimated sigma_f and sigma_w over time
    plt.figure()
    plt.plot(
        range(num_periods),
        estimated_sigma_f,
        label="Estimated sigma_f",
    )
    plt.plot(
        range(num_periods),
        estimated_sigma_w,
        label="Estimated sigma_w",
    )
    plt.xlabel("Time Period")
    plt.ylabel("Value")
    plt.title("Estimated Noise Parameters over Time")
    plt.legend()
    plt.show()
