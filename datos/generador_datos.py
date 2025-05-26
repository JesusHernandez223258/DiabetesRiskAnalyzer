import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generar_datos_ejemplo(n_samples=1000):
    np.random.seed(42)
    
    sex = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
    ethnicity = np.random.choice(['Caucasian', 'Hispanic', 'African American', 'Asian', 'Other'], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    age = np.clip(np.random.normal(50, 15, n_samples), 18, 90).astype(int)
    bmi = np.clip(np.random.normal(28, 6, n_samples), 15, 50)
    glucose = np.clip(np.random.normal(110, 30, n_samples), 60, 250)
    cholesterol_hdl = np.clip(np.random.normal(50, 12, n_samples), 20, 100)
    smoking_status = np.random.choice(['Never', 'Former', 'Current', np.nan], n_samples, p=[0.5, 0.25, 0.2, 0.05]) # Con NaNs
    physical_activity_level = np.random.choice(['Low', 'Moderate', 'High'], n_samples)
    
    # Generar algunas fechas
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365*2)) for _ in range(n_samples)]

    # Introducir algunos outliers y NaNs adicionales
    bmi[np.random.choice(n_samples, size=int(n_samples*0.02), replace=False)] = np.random.uniform(55, 70, size=int(n_samples*0.02)) # Outliers altos
    glucose[np.random.choice(n_samples, size=int(n_samples*0.03), replace=False)] = np.nan # NaNs

    return pd.DataFrame({
        'PatientID': [f'P{i:04d}' for i in range(n_samples)],
        'EnrollmentDate': dates,
        'Sex': sex,
        'Ethnicity': ethnicity,
        'Age': age,
        'BMI': bmi,
        'Glucose': glucose,
        'Cholesterol_HDL': cholesterol_hdl,
        'Smoking_Status': smoking_status,
        'Physical_Activity_Level': physical_activity_level,
        'Has_Comorbidity': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # Binaria numérica
    })

if __name__ == '__main__':
    df_ejemplo = generar_datos_ejemplo(100)
    print(df_ejemplo.head())
    print(df_ejemplo.info())
    print(df_ejemplo.isnull().sum())