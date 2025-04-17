import numpy as np

def validate_positive_parameters(arc_length, avg_speed):
    if np.any(arc_length < 0) or np.any(avg_speed <= 0):
        raise ValueError(
            "Both arc_length and avg_speed must be greater than zero.")


def lognormal_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mu_x = avg_speed
    sigma_x = 6

    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1))
    # Mean of 1 / X is exp(-mu + sigma^2 / 2)
    mean = np.exp(-mu_speed + (sigma_speed ** 2) / 2)
    time = arc_length * mean
    return time


def lognormal_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mu_x = avg_speed
    sigma_x = 6

    mu_speed = np.log((mu_x ** 2) / np.sqrt((mu_x ** 2 + sigma_x ** 2)))
    sigma_speed = np.sqrt(np.log((sigma_x ** 2 / (mu_x ** 2)) + 1))
    # To sample 1 / X, where X is lognormal
    # it is equivalent to sample X with mean -mu and sigma
    # or equivalently sample X with mean mu and sigma
    # and then take the inverse
    random_speed_inverse = np.random.lognormal(-mu_speed, sigma_speed)
    time = arc_length * random_speed_inverse
    return time


def uniform_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    low = max(avg_speed - 3, 0.001)
    high = avg_speed + 3

    inverse_mean = (np.log(high) - np.log(low)) / (high-low)
    time = arc_length * inverse_mean
    return time


def uniform_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    low = max(avg_speed - 3, 0.001)
    high = avg_speed + 3

    random_speed = np.random.uniform(low, high)
    time = arc_length / random_speed
    return time


def normal_expected_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mean = avg_speed
    std = 6
    # the next step is false, the mean of 1 / X is not 1 / mean
    time = arc_length / mean
    return time


def normal_random_time(arc_length, avg_speed=25):
    validate_positive_parameters(arc_length, avg_speed)

    mean = avg_speed
    std = 6

    random_speed = np.random.normal(mean, std)
    time = arc_length / random_speed
    return time


def expected_time(arc_length, avg_speed=25, distribution="lognormal"):
    validate_positive_parameters(arc_length, avg_speed)

    if distribution == "lognormal":
        return lognormal_expected_time(arc_length, avg_speed) * 3.6
    elif distribution == "normal":
        return normal_expected_time(arc_length, avg_speed) * 3.6
    elif distribution == "uniform":
        return uniform_expected_time(arc_length, avg_speed) * 3.6
    else:
        raise ValueError("Unsupported distribution type")


def random_time(arc_length, avg_speed=25, distribution="lognormal"):
    """Returns a random travel time given an arc length and average speed. The distribution type can be lognormal, normal, or uniform. Arc length is assumed to be in meters and average speed in km/h. """
    validate_positive_parameters(arc_length, avg_speed)
    if distribution == "lognormal":
        return lognormal_random_time(arc_length, avg_speed) * 3.6
    elif distribution == "normal":
        return normal_random_time(arc_length, avg_speed) * 3.6
    elif distribution == "uniform":
        return uniform_random_time(arc_length, avg_speed) * 3.6
    else:
        raise ValueError("Unsupported distribution type")


if __name__ == "__main__":
    arc_length = 100  # km
    avg_speed = 25  # km/h

    # Test lognormal distribution
    print("Lognormal Expected Time:", expected_time(
        arc_length, avg_speed, "lognormal"))
    print("Lognormal Random Time:", random_time(
        arc_length, avg_speed, "lognormal"))

    # Test normal distribution
    print("Normal Expected Time:", expected_time(
        arc_length, avg_speed, "normal"))
    print("Normal Random Time:", random_time(arc_length, avg_speed, "normal"))

    # Test uniform distribution
    print("Uniform Expected Time:", expected_time(
        arc_length, avg_speed, "uniform"))
    print("Uniform Random Time:", random_time(
        arc_length, avg_speed, "uniform"))
