from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"logistics_predictive_dataset.csv"
MODEL=ROOT/"models"/"delivery_time_model.joblib"
st.set_page_config(page_title="Logistics Predictive Analytics",page_icon="🚚",layout="wide")
st.title("🚚 Logistics Predictive Modeling & Optimization")
st.caption("Week 4 project | Synthetic data for educational demonstration")
df=pd.read_csv(DATA)
model=None
if MODEL.exists():
    try: model=joblib.load(MODEL)
    except Exception: model=None
with st.sidebar:
    st.header("Explore the dataset")
    regions=st.multiselect("Region",sorted(df.region.unique()),default=sorted(df.region.unique()))
    modes=st.multiselect("Transport mode",sorted(df.transport_mode.unique()),default=sorted(df.transport_mode.unique()))
    types=st.multiselect("Shipment type",sorted(df.shipment_type.unique()),default=sorted(df.shipment_type.unique()))
f=df[df.region.isin(regions)&df.transport_mode.isin(modes)&df.shipment_type.isin(types)]
if f.empty:
    st.warning("No records match these filters.");st.stop()
a,b,c,d=st.columns(4)
a.metric("Shipments selected",f"{len(f):,}")
b.metric("Average delivery time",f"{f.actual_delivery_days.mean():.2f} days")
c.metric("Average transport cost",f"₹{f.transportation_cost_inr.mean():,.0f}")
d.metric("Average shipment distance",f"{f.distance_km.mean():.0f} km")
left,right=st.columns(2)
with left:
    fig=px.histogram(f,x="actual_delivery_days",nbins=20,title="Observed Delivery Time Distribution",labels={"actual_delivery_days":"Actual delivery (days)"})
    st.plotly_chart(fig,use_container_width=True)
with right:
    summary=f.groupby("transport_mode",as_index=False).agg(avg_delivery_days=("actual_delivery_days","mean"),avg_cost=("transportation_cost_inr","mean"))
    fig=px.scatter(summary,x="avg_delivery_days",y="avg_cost",size="avg_cost",color="transport_mode",text="transport_mode",title="Mode Trade-off: Average Time vs Cost",labels={"avg_delivery_days":"Average delivery (days)","avg_cost":"Average cost (INR)"})
    st.plotly_chart(fig,use_container_width=True)
st.subheader("Predict delivery time for a shipment")
st.write("Enter planning-time features below. The saved model is created by running `python src/train_model.py`.")
with st.form("predict_form"):
    c1,c2,c3=st.columns(3)
    region= c1.selectbox("Region",sorted(df.region.unique()))
    mode= c1.selectbox("Transport mode",sorted(df.transport_mode.unique()))
    stype= c1.selectbox("Shipment type",sorted(df.shipment_type.unique()))
    distance=c2.number_input("Distance (km)",min_value=1.0,max_value=3000.0,value=250.0,step=10.0)
    volume=c2.number_input("Shipment volume (units)",min_value=1,max_value=1000,value=30,step=1)
    weight=c2.number_input("Weight (kg)",min_value=1.0,max_value=5000.0,value=100.0,step=5.0)
    traffic=c3.slider("Traffic level (1=low, 5=high)",1,5,3)
    weather=c3.selectbox("Weather condition",["Clear","Rain","Severe"])
    submitted=st.form_submit_button("Predict delivery time")
if submitted:
    if model is None:
        st.error("Model file not found or could not be loaded. Run `python src/train_model.py` in the project terminal, then refresh.")
    else:
        row=pd.DataFrame([{"region":region,"transport_mode":mode,"shipment_type":stype,"distance_km":distance,"shipment_volume_units":volume,"weight_kg":weight,"traffic_level_1_to_5":traffic,"weather_condition":weather}])
        prediction=float(model.predict(row)[0])
        st.success(f"Estimated delivery time: {prediction:.2f} days")
        if prediction>=df.actual_delivery_days.quantile(.75):
            st.warning("Higher-duration risk signal: consider carrier confirmation, contingency planning, and proactive customer updates.")
        else:
            st.info("This estimate is below the dataset's 75th-percentile delivery-time threshold. Continue normal monitoring.")
st.subheader("Mode trade-off summary")
st.dataframe(f.groupby("transport_mode").agg(Shipments=("shipment_id","count"),Average_delivery_days=("actual_delivery_days","mean"),Average_cost_INR=("transportation_cost_inr","mean")).round(2),use_container_width=True)
st.download_button("Download filtered dataset",f.to_csv(index=False).encode(),file_name="filtered_logistics_data.csv",mime="text/csv")
st.info("Predictions are estimates from synthetic data, not guaranteed delivery times. Validate with real historical records before operational use.")
