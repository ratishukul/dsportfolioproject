import streamlit as st


st.title('UserDetails')

user_input = st.text_input("Enter your details")
st.write('The user entered:', user_input)


import streamlit as st
import datetime

# st.title("User Input Demo")

# Text input
name = st.text_input("Age:")

# Text area
Diabetes = st.number_input("Diabetes:")

# Number input
BloodPressure = st.number_input("BloodPressure:")

# Checkbox
AnyTransplants = st.number_input("AnyTransplants:")

# Radio buttons
AnyChronicDiseases = st.number_input("AnyChronicDiseases:")

# Select box
Height = st.number_input("Height:")

# Multiselect
Weight = st.number_input("Weight:")

# Slider
KnownAllergies = st.number_input("KnownAllergies:")

# Date input
HistoryofCancerInFamily = st.number_input("HistoryofCancerInFamily:")

# Time input
NumberOfMajorSurgeries = st.number_input("NumberOfMajorSurgeries:")

# File uploader
PremiumPrice = st.number_input("PremiumPrice:")

# Submit Button
if st.button("Submit"):
    st.success("Form submitted successfully!")
    st.write("### Your Input Summary:")
    st.write(f"Name: {name}")
    st.write(f"Diabetes: {Diabetes}")
    st.write(f"BloodPressure: {BloodPressure}")
    st.write(f"AnyTransplants: {AnyTransplants}")
    st.write(f"AnyChronicDiseases: {AnyChronicDiseases}")
    st.write(f"Height: {Height}")
    st.write(f"Weight: {Weight}")
    st.write(f"KnownAllergies: {KnownAllergies}")
    st.write(f"HistoryofCancerInFamily: {HistoryofCancerInFamily}")
    st.write(f"NumberOfMajorSurgeries: {NumberOfMajorSurgeries}")
    st.write ( f"PremiumPrice: {PremiumPrice}" )
    # if uploaded_file is not None:
    #     st.write(f"**File Name:** {uploaded_file.name}")




