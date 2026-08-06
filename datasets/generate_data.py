import os
import random
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    random.seed(42)
    np.random.seed(42)
    
    features_config = {
    "Age": {
        "name": "Age",
        "type": "int",
        "label": "Age",
        "min_val": 20,
        "max_val": 90,
        "default": 55,
        "description": "Patient age in years"
    },
    "Sex": {
        "name": "Sex",
        "type": "str",
        "label": "Sex",
        "default": "Male",
        "options": [
            "Male",
            "Female"
        ],
        "description": "Biological sex"
    },
    "ChestPainType": {
        "name": "ChestPainType",
        "type": "str",
        "label": "Chest Pain Type",
        "default": "Typical Angina",
        "options": [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal",
            "Asymptomatic"
        ],
        "description": "Sensation category"
    },
    "MaxHeartRate": {
        "name": "MaxHeartRate",
        "type": "int",
        "label": "Max Heart Rate Achieved (bpm)",
        "min_val": 80,
        "max_val": 220,
        "default": 140,
        "description": "Peak rate during exercise stress test"
    },
    "RestBP": {
        "name": "RestBP",
        "type": "int",
        "label": "Resting Blood Pressure (mmHg)",
        "min_val": 80,
        "max_val": 200,
        "default": 120,
        "description": "Resting systolic pressure"
    },
    "Cholesterol": {
        "name": "Cholesterol",
        "type": "int",
        "label": "Serum Cholesterol (mg/dL)",
        "min_val": 100,
        "max_val": 500,
        "default": 220,
        "description": "Total cholesterol count"
    },
    "ST_Depression": {
        "name": "ST_Depression",
        "type": "float",
        "label": "ST Depression (ECG)",
        "min_val": 0.0,
        "max_val": 6.0,
        "default": 1.2,
        "description": "ECG depression relative to rest"
    }
}
    
    data = []
    for _ in range(num_samples):
        row = {}
        for name, spec in features_config.items():
            if spec.get("options"):
                val = random.choice(spec["options"])
            elif spec["type"] == "int":
                val = random.randint(spec["min_val"], spec["max_val"])
            elif spec["type"] == "float":
                val = round(random.uniform(spec["min_val"], spec["max_val"]), 2)
            else:
                val = "Sample Text"
            
            # Inject 3% random missing values to satisfy preprocessing/imputer checks
            if random.random() < 0.03:
                val = None
                
            row[name] = val
        data.append(row)
        
    df = pd.DataFrame(data)
    target_col = "Target"
    proj_id = "06_Heart_Disease_Prediction"
    
    if proj_id == "01_House_Price_Prediction":
        sq = df["SquareFootage"].fillna(2500)
        bd = df["Bedrooms"].fillna(3)
        bt = df["Bathrooms"].fillna(2.5)
        nq = df["NeighborhoodQuality"].fillna(7)
        dt = df["DistanceToCenter"].fillna(10.0)
        noise = np.random.normal(0, 15000, num_samples)
        df[target_col] = sq * 150 + bd * 15000 + bt * 20000 + nq * 25000 - dt * 3000 + noise
        df[target_col] = df[target_col].clip(lower=50000)
        
    elif proj_id == "02_Student_Performance_Prediction":
        sh = df["StudyHoursPerWeek"].fillna(15)
        at = df["AttendanceRate"].fillna(90)
        sl = df["SleepHours"].fillna(7)
        pg = df["PriorGrade"].fillna(75)
        noise = np.random.normal(0, 3, num_samples)
        df[target_col] = sh * 0.8 + at * 0.4 + sl * 1.2 + pg * 0.3 + noise
        df[target_col] = df[target_col].clip(0, 100)
        
    elif proj_id == "03_Salary_Prediction":
        ye = df["YearsExperience"].fillna(5)
        pr = df["PerformanceRating"].fillna(3)
        el = df["EducationLevel"].map({"Bachelor": 1, "Master": 2, "PhD": 3}).fillna(1)
        noise = np.random.normal(0, 5000, num_samples)
        df[target_col] = 35000 + ye * 6000 + el * 18000 + pr * 8000 + noise
        
    elif proj_id == "04_Loan_Approval_Prediction":
        inc = df["Income"].fillna(75000)
        cr = df["CreditScore"].fillna(700)
        la = df["LoanAmount"].fillna(150000)
        dti = df["DebtToIncomeRatio"].fillna(0.35)
        score = (cr - 500) / 350 + (inc / la) - dti * 2 + np.random.normal(0, 0.5, num_samples)
        df[target_col] = (score > 0.5).astype(int)
        
    elif proj_id == "05_Diabetes_Prediction":
        gl = df["GlucoseLevel"].fillna(110)
        bmi = df["BMI"].fillna(26.5)
        age = df["Age"].fillna(45)
        score = (gl - 100)/50 + (bmi - 25)/10 + (age - 35)/30 + np.random.normal(0, 0.6, num_samples)
        df[target_col] = (score > 1.2).astype(int)
        
    elif proj_id == "06_Heart_Disease_Prediction":
        hr = df["MaxHeartRate"].fillna(140)
        dep = df["ST_Depression"].fillna(1.2)
        chol = df["RestBP"].fillna(120)
        score = (dep * 1.5) - (hr - 120)/30 + (chol - 120)/80 + np.random.normal(0, 0.5, num_samples)
        df[target_col] = (score > 0.8).astype(int)
        
    elif proj_id == "07_Breast_Cancer_Detection":
        rad = df["MeanRadius"].fillna(14.0)
        area = df["MeanArea"].fillna(650.0)
        text = df["MeanTexture"].fillna(19.0)
        score = (rad - 12)*0.5 + (area - 500)/250 + (text - 15)*0.1 + np.random.normal(0, 0.4, num_samples)
        df[target_col] = (score > 1.0).astype(int)
        
    elif proj_id == "08_Employee_Attrition_Prediction":
        sat = df["JobSatisfaction"].fillna(3)
        inc = df["MonthlyIncome"].fillna(6500)
        ot = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)
        score = ot * 2.0 - sat * 0.8 - (inc / 5000) + np.random.normal(0, 0.5, num_samples)
        df[target_col] = (score > -1.5).astype(int)
        
    elif proj_id == "09_Customer_Churn_Prediction":
        ten = df["TenureMonths"].fillna(12)
        chg = df["MonthlyCharges"].fillna(75)
        con = df["ContractType"].map({"Month-to-month": 2, "One Year": 1, "Two Year": 0}).fillna(2)
        score = con * 1.5 + (chg / 50) - (ten / 10) + np.random.normal(0, 0.5, num_samples)
        df[target_col] = (score > 2.0).astype(int)
        
    elif proj_id == "10_Insurance_Premium_Prediction":
        age = df["Age"].fillna(35)
        bmi = df["BMI"].fillna(27)
        sm = df["Smoker"].map({"Yes": 1, "No": 0}).fillna(0)
        df[target_col] = 1200 + age * 280 + bmi * 120 + sm * 16000 + np.random.normal(0, 500, num_samples)
        
    elif proj_id == "11_Weather_Prediction":
        hum = df["Humidity"].fillna(0.65)
        ws = df["WindSpeed"].fillna(15.0)
        pr = df["Pressure"].fillna(1013.0)
        cc = df["CloudCover"].fillna(0.40)
        df[target_col] = 30 - hum * 15 - cc * 10 + (pr - 1000)*0.1 - (ws / 10) + np.random.normal(0, 2, num_samples)
        
    elif proj_id == "12_Movie_Recommendation_System":
        movies = [
            ("Toy Story", "Animation|Comedy", "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him."),
            ("Jumanji", "Adventure|Children", "When two kids find and play a magical board game, they release a man trapped for decades."),
            ("Heat", "Action|Crime", "A group of high-end professional thieves start to feel the heat from the LAPD."),
            ("Sabrina", "Comedy|Romance", "An ugly duckling goes to Paris and returns as a beautiful, sophisticated woman."),
            ("GoldenEye", "Action|Adventure", "James Bond teams up with the lone survivor of a destroyed Russian research center."),
            ("The Matrix", "Sci-Fi|Action", "A computer hacker learns from mysterious rebels about the true nature of his reality."),
            ("Inception", "Sci-Fi|Thriller", "A thief who steals corporate secrets through the use of dream-sharing technology."),
            ("Gladiator", "Action|Drama", "A former Roman General sets out to exact vengeance against the corrupt emperor."),
            ("Interstellar", "Sci-Fi|Drama", "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."),
            ("The Dark Knight", "Action|Crime", "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham."),
            ("Pulp Fiction", "Crime|Drama", "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits."),
            ("Forrest Gump", "Drama|Romance", "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal unfold.")
        ]
        rows = []
        for i in range(num_samples):
            mv = movies[i % len(movies)]
            rows.append({
                "Title": mv[0],
                "Genres": mv[1],
                "Overview": mv[2],
                "Popularity": round(random.uniform(5.0, 150.0), 2),
                "VoteAverage": round(random.uniform(5.0, 9.0), 1)
            })
        df = pd.DataFrame(rows)
        
    elif proj_id == "13_Fake_News_Detection":
        real_phrases = [
            "Congress passes new infrastructure funding program",
            "Astronomers identify new exoplanet in nearby star cluster",
            "Tech firm launches new AI microprocessors for server farms",
            "Health agency updates safety rules for food production"
        ]
        fake_phrases = [
            "SECRET SPACE LASER system activated over city! Citizens warned!",
            "Cure for aging discovered in standard backyard lemon trees!",
            "Breaking: Global leaders meet in secret underground bunker!",
            "Alien technology retrieved from ocean floor, government admits!"
        ]
        rows = []
        for i in range(num_samples):
            is_fake = random.choice([0, 1])
            txt = random.choice(fake_phrases if is_fake else real_phrases)
            subj = random.choice(["Politics", "World News", "Technology", "Government"])
            rows.append({"Text": txt, "Subject": subj, "Label": is_fake})
        df = pd.DataFrame(rows)
        
    elif proj_id == "14_Spam_Email_Classifier":
        spam_phrases = [
            "URGENT: Click here to claim your $1000 cash prize instantly!",
            "Get cheap pills online fast shipping guaranteed no prescription required!",
            "Make money sitting at home! Build residual streams within weeks!"
        ]
        ham_phrases = [
            "Let's schedule our project sync meeting for Thursday morning.",
            "Can you review the attached reports and send me your notes?",
            "Thanks for the coffee earlier, let's keep in touch."
        ]
        rows = []
        for i in range(num_samples):
            is_spam = random.choice([0, 1])
            body = random.choice(spam_phrases if is_spam else ham_phrases)
            rows.append({"EmailBody": body, "HasLinks": random.choice(["Yes", "No"]), "Label": is_spam})
        df = pd.DataFrame(rows)
        
    elif proj_id == "15_Sentiment_Analysis":
        pos_phrases = ["Amazing product!", "Absolutely loved using this", "Superb service and quality", "Highly recommend this build"]
        neg_phrases = ["Horrible waste of money", "Terrible customer experience", "Broke immediately", "Poor materials used"]
        neu_phrases = ["It is okay, nothing special", "Works as expected", "Average product", "Standard experience"]
        rows = []
        for i in range(num_samples):
            sent = random.choice(["Positive", "Negative", "Neutral"])
            if sent == "Positive":
                txt = random.choice(pos_phrases)
            elif sent == "Negative":
                txt = random.choice(neg_phrases)
            else:
                txt = random.choice(neu_phrases)
            rows.append({"ReviewText": txt, "Platform": random.choice(["Reddit", "LinkedIn", "TikTok"]), "Sentiment": sent})
        df = pd.DataFrame(rows)
        
    elif proj_id == "16_Language_Detection":
        langs = {
            "English": ["This is a beautiful day to write code", "The project is proceeding according to plan", "I need to go shopping"],
            "Spanish": ["Este es un hermoso día para escribir código", "El proyecto avanza según lo planeado", "Necesito ir de compras"],
            "French": ["C'est une belle journée pour écrire du code", "Le projet se déroule comme prévu", "Je dois faire des courses"],
            "German": ["Dies ist ein schöner Tag, um Code zu schreiben", "Das Projekt verläuft nach Plan", "Ich muss einkaufen gehen"],
            "Italian": ["Questo è un bellissimo giorno per scrivere codice", "Il progetto sta procedendo come previsto", "Devo andare a fare la spesa"],
            "Portuguese": ["Este é um belo dia para escrever código", "O projeto está ocorrendo conforme o planejado", "Eu preciso ir às compras"]
        }
        rows = []
        for i in range(num_samples):
            lang = random.choice(list(langs.keys()))
            txt = random.choice(langs[lang])
            rows.append({"Text": txt, "Language": lang})
        df = pd.DataFrame(rows)
        
    elif proj_id == "17_Resume_Screening":
        profiles = {
            "Software Engineering": "Software developer skilled in JavaScript, Python, Django, database performance tuning, React, and Git.",
            "HR": "Experienced HR consultant with a track record in employee relations, recruitment, and onboarding strategies.",
            "Finance": "Financial analyst with focus on corporate portfolios, balance sheet planning, audits, and investment metrics.",
            "Marketing": "SEO expert and digital content marketer with experience driving conversion rates on SaaS platforms.",
            "Sales": "Sales representative skilled in client acquisition, CRM administration, cold calling, and negotiation."
        }
        rows = []
        for i in range(num_samples):
            role = random.choice(list(profiles.keys()))
            txt = profiles[role] + " " + " ".join(random.sample(["experience", "skills", "team", "results", "growth", "proven"], 3))
            rows.append({"ResumeText": txt, "EducationLevel": random.choice(["Bachelor", "Master", "PhD"]), "JobRole": role})
        df = pd.DataFrame(rows)
        
    elif proj_id == "18_Stock_Price_Trend_Prediction":
        open_p = df["Open"].fillna(150)
        high_p = df["High"].fillna(155)
        low_p = df["Low"].fillna(148)
        close_p = df["Close"].fillna(152)
        noise = np.random.normal(0, 1.5, num_samples)
        df[target_col] = close_p * 1.01 + (high_p - low_p)*0.2 - (open_p - close_p)*0.5 + noise
        
    elif proj_id == "19_Credit_Card_Fraud_Detection":
        v1 = df["V1"].fillna(0)
        v2 = df["V2"].fillna(0)
        v3 = df["V3"].fillna(0)
        amount = df["Amount"].fillna(100)
        score = v1 * -1.5 + v2 * 2.0 - v3 * 1.8 + (amount / 1000)
        prob = 1 / (1 + np.exp(-score))
        df[target_col] = (prob > 0.85).astype(int)
        
    elif proj_id == "20_Customer_Segmentation":
        inc = df["AnnualIncome"].fillna(60)
        ss = df["SpendingScore"].fillna(50)
        df[target_col] = (inc > 70).astype(int) * 2 + (ss > 55).astype(int)
        
    elif proj_id == "21_Traffic_Flow_Prediction":
        hr = df["Hour"].fillna(12)
        hol = df["IsHoliday"].map({"Yes": 1, "No": 0}).fillna(0)
        weather = df["WeatherCondition"].map({"Clear": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}).fillna(0)
        base = 200 + np.sin((hr - 8) * np.pi / 12) * 150 - hol * 120 - weather * 40
        df[target_col] = np.round(base + np.random.normal(0, 20, num_samples)).astype(int).clip(10)
        
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(1200)
    train_df = df.iloc[:1000]
    test_df = df.iloc[1000:]
    
    os.makedirs("datasets", exist_ok=True)
    train_df.to_csv("datasets/train.csv", index=False)
    test_df.to_csv("datasets/test.csv", index=False)
    print("Synthetic datasets generated!")
