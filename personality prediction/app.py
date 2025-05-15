import os
import re
import pandas as pd
from tkinter import *
from tkinter import filedialog, messagebox
import tkinter.font as font
from sklearn.linear_model import LogisticRegression
import spacy
from docx import Document
from pdfminer.high_level import extract_text

# Load spaCy English model for NLP tasks
nlp = spacy.load('en_core_web_sm')

# Define common skill keywords
COMMON_SKILLS = ['python', 'java', 'sql', 'c++', 'machine learning', 'excel', 'communication', 'leadership', 'management', 'teamwork']

# Define common degree keywords (can expand)
DEGREES = ['bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.tech', 'm.tech', 'mba', 'associate']

# Regex for email extraction
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Regex for phone number (simple version)
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')

# Resume parser function improved
def parse_resume(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""

    # Extract text from PDF or DOCX
    if ext == ".pdf":
        try:
            text = extract_text(filepath)
        except Exception as e:
            raise ValueError(f"PDF reading error: {e}")
    elif ext == ".docx":
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"DOCX reading error: {e}")
    else:
        raise ValueError("Unsupported file format. Use .pdf or .docx only.")

    # Lowercase text for easier matching
    text_lower = text.lower()

    # Extract email
    emails = re.findall(EMAIL_REGEX, text)
    email = emails[0] if emails else "Not found"

    # Extract phone number
    phones = re.findall(PHONE_REGEX, text)
    phone = ''.join(phones[0]) if phones else "Not found"

    # Extract candidate name: use heuristics - first line or from doc entities
    doc = nlp(text)
    name = None
    # Look for PERSON entities in the first 300 characters
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.start_char < 300:
            name = ent.text
            break
    if not name:
        # fallback: first line of the resume text
        first_line = text.strip().split('\n')[0]
        name = first_line.strip()

    # Extract skills by keyword matching
    skills_found = []
    for skill in COMMON_SKILLS:
        if skill in text_lower:
            skills_found.append(skill.title())

    # Extract education degrees found in text
    degrees_found = []
    for degree in DEGREES:
        if degree in text_lower:
            degrees_found.append(degree.upper())

    # Extract companies: use ORG entities in text
    companies = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))

    # Extract possible designations: look for typical job titles
    designations_keywords = ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'intern', 'director', 'officer', 'specialist']
    designations_found = []
    for word in designations_keywords:
        if word in text_lower:
            designations_found.append(word.title())

    return {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Skills": skills_found,
        "Degrees": degrees_found,
        "Designations": designations_found,
        "Companies": companies
    }


class TrainModel:
    def train(self):
        data = pd.read_csv('training_dataset.csv')

        # Drop rows with missing data
        data = data.dropna()

        # Map gender: Male=1, Female=0 (assumes first column is gender)
        gender_col = data.columns[0]
        data[gender_col] = data[gender_col].map({'Male': 1, 'Female': 0})

        # Select features and target
        X = data.iloc[:, 0:7].apply(pd.to_numeric, errors='coerce')
        y = data.iloc[:, 7]

        # Drop rows with NaNs after conversion
        valid_idx = X.notnull().all(axis=1) & y.notnull()
        X = X[valid_idx]
        y = y[valid_idx]

        self.model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.model.fit(X, y)

        print(f"Training complete. Model accuracy on train set: {self.model.score(X, y)*100:.2f}%")

    def test(self, test_data):
        try:
            test_data_floats = [float(i) for i in test_data]
            prediction = self.model.predict([test_data_floats])
            return prediction[0]
        except Exception as e:
            print("Prediction error:", e)
            return "Unknown"


def check_type(data):
    if isinstance(data, str):
        return data.title()
    elif isinstance(data, (list, tuple)):
        return ", ".join(str(i) for i in data)
    else:
        return str(data)


def prediction_result(top, applicant_name_entry, cv_path, personality_values):
    top.withdraw()

    try:
        personality = model.test(personality_values)
    except Exception as e:
        personality = "Unknown"
        print("Prediction failed:", e)

    try:
        resume_data = parse_resume(cv_path)
    except Exception as e:
        messagebox.showerror("Error", f"Resume parsing failed: {e}")
        resume_data = {}

    result_win = Tk()
    result_win.title("Personality Prediction Result")
    result_win.geometry(f"{result_win.winfo_screenwidth()}x{result_win.winfo_screenheight()}+0+0")
    result_win.configure(background='white')

    titleFont = font.Font(family='Arial', size=30, weight='bold')
    Label(result_win, text="Personality Prediction Result", fg='green', bg='white', font=titleFont, pady=15).pack()

    Label(result_win, text=f"Candidate Name: {applicant_name_entry.get().title()}", bg='white', fg='black', anchor='w').pack(fill=X, padx=20)
    Label(result_win, text=f"Predicted Personality: {personality.title()}", bg='white', fg='black', anchor='w').pack(fill=X, padx=20, pady=(0,10))

    # Show resume extracted data
    Label(result_win, text="Parsed Resume Details:", bg='white', fg='blue', anchor='w', font=('Arial', 16, 'bold')).pack(fill=X, padx=20, pady=5)

    for key, val in resume_data.items():
        Label(result_win, text=f"{key}: {check_type(val)}", bg='white', fg='black', anchor='w').pack(fill=X, padx=40)

    # Explanation text for personality traits
    explanation_text = """
    Personality Traits Meaning:
    ---------------------------
    Openness: Curious, imaginative, open to new experiences.
    Conscientiousness: Organized, reliable, thorough.
    Extraversion: Outgoing, energetic, social.
    Agreeableness: Friendly, compassionate, cooperative.
    Neuroticism: Emotional instability, prone to stress.
    """
    Label(result_win, text=explanation_text, bg='white', fg='green', justify=LEFT, font=('Arial', 12), anchor='w').pack(fill=X, padx=20, pady=20)

    Button(result_win, text="Close", command=result_win.destroy, bg='red', fg='white').pack(pady=10)

    result_win.mainloop()


def open_file(button):
    global loc
    file_path = filedialog.askopenfilename(
        initialdir=os.path.expanduser("~"),
        filetypes=[("Documents", "*.docx *.pdf"), ("All Files", "*.*")]
    )
    if file_path:
        loc = file_path
        filename = os.path.basename(file_path)
        button.config(text=filename)
    else:
        button.config(text="Select File")


def predict_person():
    global loc
    root.withdraw()
    top = Toplevel()
    top.title("Personality Prediction - Enter Details")
    top.geometry("700x500")
    top.configure(background='black')

    titleFont = font.Font(family='Helvetica', size=20, weight='bold')
    Label(top, text="Enter Your Details for Personality Prediction", bg='black', fg='red', font=titleFont, pady=20).pack()

    Label(top, text="Applicant Name", bg='black', fg='white').place(x=70, y=130)
    Label(top, text="Age", bg='black', fg='white').place(x=70, y=160)
    Label(top, text="Gender", bg='black', fg='white').place(x=70, y=190)
    Label(top, text="Upload Resume", bg='black', fg='white').place(x=70, y=220)
    Label(top, text="Openness (1-10)", bg='black', fg='white').place(x=70, y=250)
    Label(top, text="Neuroticism (1-10)", bg='black', fg='white').place(x=70, y=280)
    Label(top, text="Conscientiousness (1-10)", bg='black', fg='white').place(x=70, y=310)
    Label(top, text="Agreeableness (1-10)", bg='black', fg='white').place(x=70, y=340)
    Label(top, text="Extraversion (1-10)", bg='black', fg='white').place(x=70, y=370)

    entry_name = Entry(top)
    entry_name.place(x=450, y=130, width=180)

    entry_age = Entry(top)
    entry_age.place(x=450, y=160, width=180)

    gender_var = IntVar()
    Radiobutton(top, text="Male", variable=gender_var, value=1, bg='black', fg='white').place(x=450, y=190)
    Radiobutton(top, text="Female", variable=gender_var, value=0, bg='black', fg='white').place(x=530, y=190)

    btn_upload = Button(top, text="Select File", command=lambda: open_file(btn_upload))
    btn_upload.place(x=450, y=220, width=180)

    entry_openness = Entry(top)
    entry_openness.insert(0, "5")
    entry_openness.place(x=450, y=250, width=180)

    entry_neuroticism = Entry(top)
    entry_neuroticism.insert(0, "5")
    entry_neuroticism.place(x=450, y=280, width=180)

    entry_conscientiousness = Entry(top)
    entry_conscientiousness.insert(0, "5")
    entry_conscientiousness.place(x=450, y=310, width=180)

    entry_agreeableness = Entry(top)
    entry_agreeableness.insert(0, "5")
    entry_agreeableness.place(x=450, y=340, width=180)

    entry_extraversion = Entry(top)
    entry_extraversion.insert(0, "5")
    entry_extraversion.place(x=450, y=370, width=180)

    btn_submit = Button(
        top,
        text="Submit",
        bg='red',
        fg='white',
        command=lambda: prediction_result(
            top,
            entry_name,
            loc,
            (
                gender_var.get(),
                entry_age.get(),
                entry_openness.get(),
                entry_neuroticism.get(),
                entry_conscientiousness.get(),
                entry_agreeableness.get(),
                entry_extraversion.get(),
            )
        )
    )
    btn_submit.place(x=350, y=420, width=200)


if __name__ == "__main__":
    loc = ""
    model = TrainModel()
    model.train()

    root = Tk()
    root.title("Personality Prediction System")
    root.geometry("700x500")
    root.configure(background='white')

    titleFont = font.Font(family='Helvetica', size=25, weight='bold')
    Label(root, text="Personality Prediction System", bg='white', font=titleFont, pady=50).pack()

    Button(
        root,
        text="Predict Personality",
        bg='black',
        fg='white',
        font=('Arial', 14, 'bold'),
        command=predict_person
    ).place(relx=0.5, rely=0.5, anchor=CENTER)

    root.mainloop()
