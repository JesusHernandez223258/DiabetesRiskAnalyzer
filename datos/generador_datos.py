import numpy as np
import pandas as pd

def generar_datos_ejemplo():
    np.random.seed(42)
    n_samples = 1000
    sex = np.random.choice(['Male', 'Female'], n_samples)
    ethnicity = np.random.choice(['Caucasian', 'Hispanic', 'African American', 'Asian'], n_samples)
    age = np.clip(np.random.normal(50, 15, n_samples), 18, 80)
    bmi = np.clip(np.random.normal(26, 4, n_samples), 18, 45)
    glucose = np.clip(np.random.normal(100, 25, n_samples), 70, 200)
    
    return pd.DataFrame({
        'Sex': sex,
        'Ethnicity': ethnicity,
        'Age': age,
        'BMI': bmi,
        'Glucose': glucose
    })
