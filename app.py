import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/best_car_insurance_model.pkl")

st.set_page_config(page_title="Car Insurance Claim Predictor", layout="centered")
st.title("🚗 Car Insurance Claim Prediction")
st.markdown("---")

def yesno(x):
    return 1 if x == "Yes" else 0

def clean_num(x):
    try:
        return float(x)
    except:
        return 0.0



st.sidebar.title("🚗 Car Insurance Claim")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Data Explorer", "Model Monitor", "Admin"])

if page == "Home":

    st.write("""
    ### About This App
    This AI system predicts the probability that a customer will file a car insurance claim.

    **It uses a production-grade machine learning pipeline trained on 300,000+ policy records.**

    ### Features Used
    - Vehicle specs  
    - Safety features  
    - Policy tenure  
    - Demographics  
    - Location cluster  

    ### Output
    - Claim probability  
    - Risk classification  
    - Helps insurance companies prevent fraud and reduce losses.
    """)

    st.info("👈 Go to Predict page from sidebar to check risk")


# -------------------- PREDICTION --------------------

if page== "Prediction":
    
    # -------------------- INPUT UI --------------------
    
    make = st.selectbox("Make", ["Maruti","Hyundai","Tata","Mahindra","Toyota","Honda","Ford","Renault","Kia",
                                "Volkswagen","Skoda","Nissan","Chevrolet","Fiat","Datsun","MG","Jeep","Isuzu"])

    model_name = st.selectbox("Model", [f"M{i}" for i in range(1,13)])

    fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","Electric"])
    segment = st.selectbox("Segment", ["A","B","C","D","E"])
    transmission_type = st.selectbox("Transmission", ["Manual","Automatic"])
    steering_type = st.selectbox("Steering Type", ["Power","Manual"])
    rear_brakes_type = st.selectbox("Rear Brakes Type", ["Drum","Disc"])
    engine_type = st.selectbox("Engine Type", ["1.0L","1.2L","1.5L","2.0L"])
    area_cluster = st.selectbox("Area Cluster", [f"C{i}" for i in range(1,26)])

    # -------------------- NUMERIC --------------------

    age_of_policyholder = clean_num(st.text_input("Policyholder Age", "35"))
    age_of_car = clean_num(st.text_input("Car Age", "2"))
    policy_tenure = clean_num(st.text_input("Policy Tenure", "1"))
    population_density = clean_num(st.text_input("Population Density", "500"))
    gross_weight = clean_num(st.text_input("Gross Weight", "1000"))
    displacement = clean_num(st.text_input("Displacement", "1200"))
    cylinder = clean_num(st.text_input("Cylinder", "4"))
    gear_box = clean_num(st.text_input("Gear Box", "5"))
    turning_radius = clean_num(st.text_input("Turning Radius", "5"))
    length = clean_num(st.text_input("Length", "3500"))
    width = clean_num(st.text_input("Width", "1500"))
    height = clean_num(st.text_input("Height", "1500"))
    airbags = clean_num(st.text_input("Airbags", "2"))
    ncap_rating = clean_num(st.text_input("NCAP Rating", "4"))
    max_torque = clean_num(st.text_input("Max Torque", "100"))
    max_power = clean_num(st.text_input("Max Power", "80"))

    # -------------------- BINARY --------------------

    is_esc = yesno(st.selectbox("ESC", ["Yes","No"]))
    is_adjustable_steering = yesno(st.selectbox("Adjustable Steering", ["Yes","No"]))
    is_tpms = yesno(st.selectbox("TPMS", ["Yes","No"]))
    is_parking_sensors = yesno(st.selectbox("Parking Sensors", ["Yes","No"]))
    is_parking_camera = yesno(st.selectbox("Parking Camera", ["Yes","No"]))
    is_front_fog_lights = yesno(st.selectbox("Fog Lights", ["Yes","No"]))
    is_rear_window_wiper = yesno(st.selectbox("Rear Wiper", ["Yes","No"]))
    is_rear_window_washer = yesno(st.selectbox("Rear Washer", ["Yes","No"]))
    is_rear_window_defogger = yesno(st.selectbox("Defogger", ["Yes","No"]))
    is_brake_assist = yesno(st.selectbox("Brake Assist", ["Yes","No"]))
    is_power_door_locks = yesno(st.selectbox("Power Door Locks", ["Yes","No"]))
    is_central_locking = yesno(st.selectbox("Central Locking", ["Yes","No"]))
    is_power_steering = yesno(st.selectbox("Power Steering", ["Yes","No"]))
    is_driver_seat_height_adjustable = yesno(st.selectbox("Driver Seat Adjustable", ["Yes","No"]))
    is_day_night_rear_view_mirror = yesno(st.selectbox("Day/Night Mirror", ["Yes","No"]))
    is_ecw = yesno(st.selectbox("ECW", ["Yes","No"]))
    is_speed_alert = yesno(st.selectbox("Speed Alert", ["Yes","No"]))
    
    
    # -------------------- BUILD INPUT DF --------------------

    input_df = pd.DataFrame([{
        "make": make, "model": model_name, "fuel_type": fuel_type, "segment": segment,
        "transmission_type": transmission_type, "steering_type": steering_type,
        "rear_brakes_type": rear_brakes_type, "engine_type": engine_type, "area_cluster": area_cluster,
        "age_of_policyholder": age_of_policyholder, "age_of_car": age_of_car, "policy_tenure": policy_tenure,
        "population_density": population_density, "gross_weight": gross_weight, "displacement": displacement,
        "cylinder": cylinder, "gear_box": gear_box, "turning_radius": turning_radius, "length": length,
        "width": width, "height": height, "airbags": airbags, "ncap_rating": ncap_rating,
        "max_torque": max_torque, "max_power": max_power,
        "is_esc": is_esc, "is_adjustable_steering": is_adjustable_steering, "is_tpms": is_tpms,
        "is_parking_sensors": is_parking_sensors, "is_parking_camera": is_parking_camera,
        "is_front_fog_lights": is_front_fog_lights, "is_rear_window_wiper": is_rear_window_wiper,
        "is_rear_window_washer": is_rear_window_washer, "is_rear_window_defogger": is_rear_window_defogger,
        "is_brake_assist": is_brake_assist, "is_power_door_locks": is_power_door_locks,
        "is_central_locking": is_central_locking, "is_power_steering": is_power_steering,
        "is_driver_seat_height_adjustable": is_driver_seat_height_adjustable,
        "is_day_night_rear_view_mirror": is_day_night_rear_view_mirror,
        "is_ecw": is_ecw, "is_speed_alert": is_speed_alert
    }])
    
    
    if st.button("Predict Claim Probability"):
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"Claim Probability: {prob*100:.2f}%")
        st.balloons()



# -------------------Data Explorer --------------------------------------
import matplotlib.pyplot as plt

if page == "Data Explorer":
    st.title("📊 Dataset Explorer")

    df = pd.read_csv("train.csv")

    st.subheader("🔢 Dataset Shape")
    st.write(df.shape)

    st.subheader("📄 Sample Data")
    st.dataframe(df.head(50))

    # Target Distribution
    st.subheader("🎯 Claim Distribution")
    claim_counts = df["is_claim"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(["No Claim", "Claim"], claim_counts.values)
    st.pyplot(fig)

    st.markdown("**⚠ Dataset is highly imbalanced → justifies imbalance handling in model**")

    # Feature Analysis
    st.subheader("🚘 Claim Rate by Vehicle Segment")
    seg_claim = df.groupby("segment")["is_claim"].mean().sort_values()

    fig2, ax2 = plt.subplots(figsize=(10,4))
    seg_claim.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Claim Probability")
    st.pyplot(fig2)

    st.subheader("⛽ Claim Rate by Fuel Type")
    fuel_claim = df.groupby("fuel_type")["is_claim"].mean()

    fig3, ax3 = plt.subplots()
    fuel_claim.plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    st.subheader("📍 Claim Rate by Area Cluster")
    area_claim = df.groupby("area_cluster")["is_claim"].mean()

    fig4, ax4 = plt.subplots(figsize=(10,4))
    area_claim.plot(kind="bar", ax=ax4)
    st.pyplot(fig4)

    st.success("This explorer proves why class imbalance handling is mandatory.")

from mlflow.tracking import MlflowClient
import plotly.express as px


if page == "Model Monitor":

    st.title("📊 Model Monitor")

    client = MlflowClient()
    
    st.subheader("🎯 Claim Classification Model Performance")

    exp_cls = client.get_experiment_by_name("Car_Insurance_Claim_Model")

    if exp_cls is not None:
        runs_cls = client.search_runs(exp_cls.experiment_id)

        if runs_cls:
            df_cls = pd.DataFrame([{
                "Model": r.data.tags.get("mlflow.runName"),
                "Run ID": r.info.run_id,
                "ROC-AUC": r.data.metrics.get("roc_auc"),
                "F1 Score": r.data.metrics.get("f1_score"),
                "Accuracy": r.data.metrics.get("accuracy")
            } for r in runs_cls])

            st.dataframe(df_cls, use_container_width=True)
            
            
    
            st.subheader("📈 F1 Score vs Models")

            fig = px.line(
                df_cls,
                x="Model",
                y="F1 Score",
                title="F1 Score Across Claim Models",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            
            
            st.subheader("📊 ROC-AUC vs Accuracy")

            fig2 = px.bar(
                df_cls,
                x="Model",
                y="ROC-AUC",
                color="Accuracy",
                text_auto=".3f",
                title="Claim Models: ROC-AUC vs Accuracy"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No MLflow runs found.")
    else:
        st.error("MLflow experiment not found.")



            
