import numpy as np
import pandas as pd

CONDITIONS = ["Healthy", "Diabetes", "Heart_Disease", "Respiratory_Infection", "Migraine"]


def generate_synthetic_health_data(n_samples=6000, random_state=42):
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // len(CONDITIONS)

    rows = []
    for condition in CONDITIONS:
        for _ in range(n_per_class):
            age = int(np.clip(rng.normal(45, 15), 5, 90))
            sex = rng.choice([0, 1])  # 0 = female, 1 = male
            bmi = np.clip(rng.normal(26, 5), 15, 50)

            # Baseline vitals
            bp_systolic = rng.normal(120, 12)
            cholesterol = rng.normal(190, 30)
            glucose = rng.normal(95, 15)
            heart_rate = rng.normal(75, 10)

            # Baseline symptom probabilities (0 = absent, 1 = present)
            fever = 0
            cough = 0
            fatigue = 0
            chest_pain = 0
            shortness_of_breath = 0
            headache = 0
            nausea = 0
            joint_pain = 0

            if condition == "Healthy":
                fatigue = rng.binomial(1, 0.05)

            elif condition == "Diabetes":
                glucose = rng.normal(170, 25)
                bmi = np.clip(rng.normal(31, 5), 18, 55)
                fatigue = rng.binomial(1, 0.7)
                nausea = rng.binomial(1, 0.2)

            elif condition == "Heart_Disease":
                bp_systolic = rng.normal(150, 15)
                cholesterol = rng.normal(250, 25)
                heart_rate = rng.normal(95, 12)
                chest_pain = rng.binomial(1, 0.8)
                shortness_of_breath = rng.binomial(1, 0.65)
                fatigue = rng.binomial(1, 0.5)

            elif condition == "Respiratory_Infection":
                fever = rng.binomial(1, 0.75)
                cough = rng.binomial(1, 0.85)
                shortness_of_breath = rng.binomial(1, 0.4)
                fatigue = rng.binomial(1, 0.6)
                heart_rate = rng.normal(88, 10)

            elif condition == "Migraine":
                headache = rng.binomial(1, 0.9)
                nausea = rng.binomial(1, 0.55)
                fatigue = rng.binomial(1, 0.3)

            rows.append({
                "age": age,
                "sex": sex,
                "bmi": round(bmi, 1),
                "bp_systolic": round(max(bp_systolic, 70), 1),
                "cholesterol": round(max(cholesterol, 100), 1),
                "glucose": round(max(glucose, 50), 1),
                "heart_rate": round(max(heart_rate, 40), 1),
                "fever": fever,
                "cough": cough,
                "fatigue": fatigue,
                "chest_pain": chest_pain,
                "shortness_of_breath": shortness_of_breath,
                "headache": headache,
                "nausea": nausea,
                "joint_pain": joint_pain,
                "diagnosis": condition,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_synthetic_health_data()
    df.to_csv("patient_data.csv", index=False)
    print("Synthetic dataset saved as patient_data.csv")
    print(f"Shape: {df.shape}")
    print(df["diagnosis"].value_counts())