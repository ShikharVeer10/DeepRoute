import numpy as np
import pandas as pd
from typing import Tuple
from app.data_pipeline.synthetic_data import generate_training_data
from app.models.modular.config import FEATURE_NAMES, N_SAMPLES, RANDOM_STATE


def load_dataset(n_samples: int = N_SAMPLES) -> Tuple[pd.DataFrame, pd.Series, list]:
    df = generate_training_data(n_samples=n_samples, seed=RANDOM_STATE)
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    
    X = df[available_features]
    y = df["travel_time_factor"]
    
    return X, y, available_features
