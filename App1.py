import streamlit as st
import pandas as pd
import pickle 
import os

# 1. Setup The Page
st.set_page_config(
    page_title="O-Level Predictor", 
    page_icon="🎓"
)

st.title("🎓 O-Level English Distinction Predictor")
st.write("Enter the Student's History to predict their English O-level result.")

# 2. Load the Brain
# We use a special command @st-cache_resource so it doesn't reload the brain every time you click a button
@st.cache_resource
def load_model():
    filename = 'distinction_brain.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return None

model = load_model()

if model is None:
    st.error ("❌ Error: 'distinction_brain.pkl' not found! Please run your training script file first.")
    st.stop()

# 3. The GUI (Inputs on the Main Page)
with st.form("student_input_form"):
    st.subheader("👨‍🎓 Student Details")
    name = st.text_input("Student Name (Optional)", placeholder="e.g. Zoe Tay")
    st.markdown("---")
    st.subheader("Past English Scores")
    
    # Create a list of scores from 0 to 100 for the dropdowns
    score_options = list(range(101))
    
    s1 = st.selectbox("Sec 1 English", options=score_options, index=70)
    s2 = st.selectbox("Sec 2 English", options=score_options, index=70)
    s3 = st.selectbox("Sec 3 English", options=score_options, index=70)
    s4 = st.selectbox("Sec 4 Mid-Year", options=score_options, index=70)
    prelim = st.selectbox("Prelim English Score", options=score_options, index=70)
    
    # The prediction button is now a form submit button
    submitted = st.form_submit_button("🔮 Predict Result")
# 4. The Prediction Logic & Display
if submitted:
    # Organize the data EXACTLY how the model expects it
    # Order: ['S1_Eng', S2_Eng', 'S3_Eng', 'S4_MY_Eng', 'Prelim_Eng']
    input_data = pd.DataFrame(
        [[s1, s2, s3, s4, prelim]],
        columns=['S1_Eng', 'S2_Eng', 'S3_Eng', 'S4_MY_Eng', 'Prelim_Eng']
    )
    
    # Get Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Result
    st.subheader("📊 Prediction Outcome")
    if prediction == 1:
        st.success("🌟 **Result: High Chance of Distinction!**")
        st.metric(label="Confidence", value=f"{probability:.1%}")
        st.balloons() #<-- A fun surprise effect
    else:
        st.warning("⚠️ **Result: Unlikely to be a Distinction.**")
        st.metric(label="Confidence in this Outcome", value=f"{1-probability:.1%}")
        st.info("Suggested Action: Focus on Prelim revision, as it has the highest weight.")
 