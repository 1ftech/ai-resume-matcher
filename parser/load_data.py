import pandas as pd

def load_resume_data(path='data/resume.csv'):
    df = pd.read_csv(path)
    print("✅ Loaded dataset. Shape:", df.shape)
    print("📋 Full Column Names:", df.columns.tolist())
    print("🧾 Sample rows:")
    print(df.head())

    if 'Category' in df.columns:
        print("\nCategory distribution:")
        print(df['Category'].value_counts())
    else:
        print("⚠️ 'Category' column not found. Please inspect your dataset.")
    
    return df

if __name__ == "__main__":
    load_resume_data()
