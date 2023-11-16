import numpy as np
import pandas as pd


def inversion_detector(
    df: pd.DataFrame, landmark_name: str = "ANKLE"
) -> np.ndarray:
    left_landmark = df.filter(
        regex=f".*LEFT.*{landmark_name}.*|.*{landmark_name}.*LEFT.*"
    )
    right_landmark = df.filter(
        regex=f".*RIGHT.*{landmark_name}.*|.*{landmark_name}.*RIGHT.*"
    )

    inversion = np.where(right_landmark["x"] > left_landmark["x"], True, False)
    return inversion
