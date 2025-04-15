from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load ML Models
tech_model = joblib.load("models/tech_model.pkl")
medical_model = joblib.load("models/med_model.pkl")
creative_model = joblib.load("models/creative_model.pkl")

# Load Label Encoders & Scalers
tech_encoder = joblib.load("models/tech_label_encoders.pkl")
medical_encoder = joblib.load("models/med_label_encoders.pkl")
creative_encoder = joblib.load("models/creative_label_encoders.pkl")

tech_scaler = joblib.load("models/tech_scaler.pkl")
medical_scaler = joblib.load("models/med_scaler.pkl")
creative_scaler = joblib.load("models/creative_scaler.pkl")

tech_target_encoder = joblib.load("models/tech_target_label_encoder.pkl")
medical_target_encoder = joblib.load("models/med_target_label_encoder.pkl")
creative_target_encoder = joblib.load("models/creative_target_label_encoder.pkl")


# Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/contact')
def contact():
    return render_template('Contact.html')

# Domain Selection
@app.route('/domain')
def domain():
    return render_template('Career_Domains.html')

# Technical Form Page
@app.route('/technical', methods=['GET', 'POST'])
def technical():
    if request.method == 'POST':
        # Get form data
        data = {
            "Education Level": request.form["education"],
            "Degree": request.form["degree"],
            "CGPA": float(request.form["cgpa"]),
            "Coding Skill Level": int(request.form["codingSkill"]),
            "Num Programming Languages": int(request.form["numLanguages"]),
            "Primary Programming Language": request.form["primaryLanguage"],
            "Num Projects": int(request.form["numProjects"]),
            "Project Domains": request.form["projectDomain"],
            "Certifications Completed": int(request.form["certifications"]),
            "Internship Experience (Months)": int(request.form["internshipExp"]),
            "Previous Job Experience (Years)": int(request.form["jobExp"]),
            "Machine Learning Knowledge": request.form["machinelearning"],
            "Cloud Computing Knowledge": request.form["cloudcomputing"],
            "Cybersecurity Knowledge": request.form["cybersecurity"],
            "Databases Knowledge": request.form["database"],
            "Problem-Solving Skill Score": int(request.form["problemSolvingSkill"]),
            "Communication Skill Score": int(request.form["communicationSkill"]),
            "Preferred Work Environment": request.form["workenv"],
            "Preferred Job Role": request.form["jobRole"],
            "Soft Skills Rating": int(request.form["softSkill"]),
            "Hackathon Participation": request.form["hackathon"],
            "Open-Source Contributions": request.form["opensource"],
        }

        # Convert categorical variables using label encoder
        for key in ["Education Level", "Degree", "Primary Programming Language", "Project Domains", 
                    "Machine Learning Knowledge", "Cloud Computing Knowledge", "Cybersecurity Knowledge", 
                    "Databases Knowledge", "Preferred Work Environment", "Preferred Job Role", "Hackathon Participation",
                    "Open-Source Contributions"]:
            data[key] = tech_encoder[key].transform([data[key]])[0]

        # Convert data to DataFrame and scale
        df = pd.DataFrame([data])
        df_scaled = tech_scaler.transform(df)

        # Make prediction
        prediction = tech_model.predict(df_scaled)
        predicted_career = tech_target_encoder.inverse_transform(prediction)[0]

        return render_template('Technical.html', prediction=predicted_career)

    return render_template('Technical.html', prediction=None)


# Medical Form Page
@app.route('/medical', methods=['GET', 'POST'])
def medical():
    if request.method == 'POST':
        data = {
            "Region": request.form["region"],
            "High School Science Score": float(request.form["highSchoolScore"]),
            "Biology Score": float(request.form["bScore"]),
            "Chemistry Score": float(request.form["cScore"]),
            "Physics Score": float(request.form["pScore"]),
            "Undergraduate Degree": request.form["ugDegree"],
            "Medical School Ranking": request.form["medSchoolRanking"],
            "Institution Type": request.form["institution"],
            "Medical Entrance Exam Score": float(request.form["entranceScore"]),
            "Ability to diagnose & treat patients": int(request.form["abilityToDiagnose"]),
            "Doctor-patient interaction, explaining procedures": int(request.form["doctorPatientInteraction"]),
            "Hand-Eye Coordination": int(request.form["handEyeCoordination"]),
            "Empathy": int(request.form["empathy"]),
            "Analytical Thinking": int(request.form["analyticalThinking"]),
            "Years of Medical Education": int(request.form["yearsMedEdu"]),
            "Clinical Internship Experience (Months)": int(request.form["cliInternExp"]),
            "Research Experience (Months)": int(request.form["resExp"]),
            "Medical Certifications": int(request.form["medCert"]),
            "Specialization Courses": int(request.form["speCourse"]),
            "Interest in Surgery": int(request.form["surgery"]),
            "Interest in Patient Care": int(request.form["patientCare"]),
            "Interest in Laboratory & Research": int(request.form["labReasearch"]),
            "Interest in Emergency Medicine": int(request.form["emergencyMedicine"]),
            "Interest in Mental Health & Psychology": int(request.form["mentalHealth"]),
            "Preferred Work Environment": request.form["workenv"],
            "Comfort with Long Working Hours": int(request.form["workingHrs"]),
            "Willingness to Work in Rural Areas": request.form["work"],
            "Teamwork Preference": int(request.form["teamwork"])
        }

        # Convert categorical variables using label encoder
        for key in ["Region", "Undergraduate Degree", "Medical School Ranking", "Institution Type", 
                    "Preferred Work Environment", "Willingness to Work in Rural Areas"]:
            data[key] = medical_encoder[key].transform([data[key]])[0]

        # Convert data to DataFrame and scale
        df = pd.DataFrame([data])
        df_scaled = medical_scaler.transform(df)

        # Make prediction
        prediction = medical_model.predict(df_scaled)
        predicted_career = medical_target_encoder.inverse_transform(prediction)[0]

        return render_template('Medical.html', prediction=predicted_career)

    return render_template('Medical.html', prediction=None)


# Creative Form Page
@app.route('/creative', methods=['GET', 'POST'])
def creative():
    if request.method == 'POST':
        data = {
            "Creativity Score": int(request.form["creativity"]),
            "Adobe Photoshop Proficiency": int(request.form["adobePhotoshop"]),
            "Adobe Illustrator Proficiency": int(request.form["adobeIllustrator"]),
            "Adobe After Effects Proficiency": int(request.form["adobeAfterEffects"]),
            "3D Modeling Skill": int(request.form["3DModeling"]),
            "UI/UX Knowledge": int(request.form["UI/UXKnowledge"]),
            "Video Editing Skill": int(request.form["videoEditing"]),
            "Game Design Knowledge": int(request.form["gameDesign"]),
            "Sketching/Drawing Skill": int(request.form["sketching"]),
            "Typography Skill": int(request.form["typography"]),
            "Color Theory Knowledge": int(request.form["colorTheory"]),
            "Branding Knowledge": int(request.form["branding"]),
            "Web Development Knowledge": int(request.form["web"]),
            "Fashion Design Knowledge": int(request.form["fashion"]),
            "Interior Design Knowledge": int(request.form["interior"]),
            "Product Design Knowledge": int(request.form["productDesign"]),
            "Illustration Skill": int(request.form["illustration"]),
            "Animation Skill": int(request.form["animation"]),
            "Years of Experience": int(request.form["yrsExp"]),
            "Preferred Design Style": request.form["designStyle"]
        }

        # Convert categorical variables using label encoder
        for key in ["Preferred Design Style"]:
            data[key] = creative_encoder[key].transform([data[key]])[0]

        # Convert data to DataFrame and scale
        df = pd.DataFrame([data])
        df_scaled = creative_scaler.transform(df)

        # Make prediction
        prediction = creative_model.predict(df_scaled)
        predicted_career = creative_target_encoder.inverse_transform(prediction)[0]
        print("Predicted Career:", predicted_career)

        return render_template('Creative.html', prediction=predicted_career)

    return render_template('Creative.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
