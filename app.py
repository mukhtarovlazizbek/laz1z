import streamlit as st
import joblib
import pandas as pd

st.title("Dori turini bashorat qilish")

# Foydalanuvchidan ma'lumotlarni kiritishni so'rash
condition = st.text_input("Kasallik holatini kiriting")
drug = st.text_input("Dori nomini kiriting")
ease_of_use = st.number_input("Foydalanish qulayligini kiriting (1-10)", min_value=1, max_value=10, step=1)
effective = st.number_input("Samaradorlik bahosini kiriting (1-10)", min_value=1, max_value=10, step=1)
form = st.text_input("Dori shaklini kiriting")
indication = st.text_input("Indikatsiyani kiriting")
price = st.number_input("Narxni kiriting", min_value=0.0, step=0.01)
reviews = st.number_input("Sharhlar sonini kiriting", min_value=0, step=1)
satisfaction = st.number_input("Qoniqish darajasini kiriting (1-10)", min_value=1, max_value=10, step=1)

# Modelni yuklash va bashorat qilish
if st.button("Dori turini bashorat qilish"):
    # Kiritilgan ma'lumotlarni DataFrame ga o'tkazish
    input_data = {
        "Condition": [condition],
        "Drug": [drug],
        "EaseOfUse": [ease_of_use],
        "Effective": [effective],
        "Form": [form],
        "Indication": [indication],
        "Price": [price],
        "Reviews": [reviews],
        "Satisfaction": [satisfaction]
    }
    
    df = pd.DataFrame(input_data)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['Condition', 'Drug', 'Form', 'Indication'])

    # Modelni yuklash
    model = joblib.load('decision_tree_model.pkl')  # Model faylingiz nomini mos ravishda kiriting

    # Bashorat qilish
    outcome = model.predict(df_encoded)

    # Natijani ko'rsatish
    st.write(f"Bashorat qilingan dori turi: {outcome[0]}")
