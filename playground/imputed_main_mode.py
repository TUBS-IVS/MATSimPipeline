
import random
def main_mode_imputed(row):
    """
    Impute main mode based on the distance of the leg, the availability of a car and statistical probabilities.
    Note: Imputed main mode is available in MiD, so this isn't needed, but it's here for reference.
    :param row:
    :return: mode
    """
    r = random.random()

    boundaries_never_or_no_license = [
        (0.5, [0.89], ["walk", "bike"]),
        (1, [0.74], ["walk", "bike"]),
        (2, [0.48, 0.71], ["walk", "bike", "ride"]),
        (5, [0.25, 0.52, 0.81], ["walk", "bike", "ride", "pt"]),
        (float('inf'), [0.53], ["ride", "pt"])
    ]

    boundaries_otherwise = [
        (0.5, [0.89], ["walk", "bike"]),
        (1, [0.56, 0.76], ["walk", "bike", "car"]),
        (2, [0.31, 0.52, 0.65], ["walk", "bike", "ride", "car"]),
        (5, [0.14, 0.29, 0.45, 0.90], ["walk", "bike", "ride", "car", "pt"]),
        (float('inf'), [0.26, 0.77], ["ride", "car", "pt"])
    ]

    boundaries = boundaries_never_or_no_license if row[s.CAR_AVAIL_COL] == s.CAR_NEVER or row[
        "imputed_license"] == s.LICENSE_NO else boundaries_otherwise

    for distance, probabilities, modes in boundaries:
        if row[s.LEG_DISTANCE_KM_COL] < distance:
            for prob, mode in zip(probabilities, modes):
                if r <= prob:
                    return mode
            return modes[-1]  # If the random number is greater than all probabilities, return the last mode in the list

    return None
