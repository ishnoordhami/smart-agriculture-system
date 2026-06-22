import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sqlite3

st.set_page_config(
    page_title="AgriVision",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():

    crop_rec_model = pickle.load(
        open("crop_recommendation_model.pkl", "rb")
    )

    crop_encoder = pickle.load(
        open("label_encoder.pkl", "rb")
    )

    crop_yield_model = pickle.load(
        open("crop_yield_model.pkl", "rb")
    )

    crop_price_model = pickle.load(
        open("crop_price_model.pkl", "rb")
    )

    price_encoders = pickle.load(
        open("label_encoder_price.pkl", "rb")
    )

    disease_model = pickle.load(
        open("crop_disease_model.pkl", "rb")
    )

    disease_encoder = pickle.load(
        open("label_encoder_disease.pkl", "rb")
    )
    return (
        crop_rec_model,
        crop_encoder,
        crop_yield_model,
        crop_price_model,
        price_encoders,
        disease_model,
        disease_encoder
    )


(
    crop_rec_model,
    crop_encoder,
    crop_yield_model,
    crop_price_model,
    price_encoders,
    disease_model,
    disease_encoder
) = load_models()

conn = sqlite3.connect(
    "equipment.db",
    check_same_thread=False
)

c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS equipment(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_name TEXT,
    location TEXT,
    equipment_type TEXT,
    rent_per_day REAL,
    contact TEXT
)
""")

conn.commit()

st.markdown("""
<style>

/* Remove Streamlit top spacing */
.block-container{
    padding-top: 0.5rem !important;
}

/* Hide Streamlit header completely */
header[data-testid="stHeader"]{
    display:none !important;
}

/* Remove empty top toolbar space */
[data-testid="stToolbar"]{
    display:none !important;
}

/* Remove top blank container */
div[data-testid="stDecoration"]{
    display:none !important;
}

/* Remove extra top padding */
.main .block-container{
    padding-top: 0rem !important;
    margin-top: -20px !important;
}

/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Background */
.stApp{
    background: linear-gradient(
        135deg,
        #f4fff4 0%,
        #e8f5e9 50%,
        #d7f5dd 100%
    );
}
/* Page Width */
.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #1B5E20,
        #2E7D32,
        #43A047
    );
}

[data-testid="stSidebar"] {
    color:white !important;
}
/* Hero Section */
.hero-card{
    background: linear-gradient(
        135deg,
        #0B6623,
        #2E8B57,
        #4CAF50
    );
    border-radius:25px;
    padding:40px;
    text-align:center;
    color:white;
    box-shadow:0 10px 30px rgba(0,0,0,0.25);
    margin-bottom:20px;
}
.hero-card h1,
.hero-card h2,
.hero-card h3,
.hero-card p {
    color: white !important;
}
.hero-card *{
    color:white !important;
}
/* Glass Cards */
.metric-card{
    background:white;
    border-radius:20px;
    padding:25px;
    text-align:center;
    box-shadow:0 8px 25px rgba(0,0,0,0.12);
    transition:0.3s;
    cursor:pointer;
}

.metric-card:hover{
    transform:translateY(-5px);
}

.metric-card h2{
    color:#1B5E20 !important;
    font-weight:700;
}

.metric-card h3{
    color:#444 !important;
}
.metric-card:hover{
    transform:translateY(-5px);
}

/* Prediction Cards */
.prediction-box{
    background:white;
    border-radius:20px;
    padding:25px;
    box-shadow:0 8px 20px rgba(0,0,0,0.10);
    border-left:8px solid #2E7D32;
}
/* Inputs */
.stNumberInput,
.stSelectbox,
.stTextInput{
    border-radius:12px;
}

/* Buttons */
.stButton > button{
    width:100%;
    height:58px;
    border:none;
    border-radius:15px;
    background:linear-gradient(
        135deg,
        #1B5E20,
        #43A047
    );
    color:white;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
    box-shadow:0 6px 15px rgba(0,0,0,0.15);
}
.stButton > button:hover{
    transform:translateY(-2px);
    background:linear-gradient(
        135deg,
        #0B6623,
        #2E7D32
    );
}

/* Success Box */
.success-card{
    background:linear-gradient(
        135deg,
        #43A047,
        #66BB6A
    );
    color:white;
    padding:20px;
    border-radius:18px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    box-shadow:0 8px 20px rgba(0,0,0,0.15);
}
/* Section Headings */
h1{
    color:#1B5E20;
    font-weight:800;
}

h2{
    color:#2E7D32;
    font-weight:700;
}

h3{
    font-weight:600;
}

/* Radio Buttons */
div[role="radiogroup"] label{
    background:rgba(255,255,255,0.12);
    padding:10px;
    border-radius:10px;
    margin-bottom:5px;
}
/* Mobile Optimization */
@media (max-width:768px){

.hero-card{
padding:25px;
}

.metric-card{
padding:15px;
}

h1{
font-size:30px;
}

h2{
font-size:22px;
}

.stButton > button{
height:50px;
font-size:16px;
}

}
/* ===== FIX LABELS & HEADINGS ===== */

h1, h2, h3, h4, h5, h6,
label,
p,
span,
div {
    color: #1B1B1B !important;
}

/* Page Titles */
.stTitle,
[data-testid="stMarkdownContainer"] h1 {
    color: #1B5E20 !important;
    font-weight: 800 !important;
}

/* Input Labels */
.stNumberInput label,
.stSelectbox label,
.stTextInput label,
.stDateInput label {
    color: #1B1B1B !important;
    font-weight: 600 !important;
}

/* ===== WHITE INPUT BOXES ===== */

.stNumberInput input,
.stTextInput input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #dfe6e9 !important;
    border-radius: 10px !important;
}

/* Selectboxes */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

/* Dropdown text */
.stSelectbox * {
    color: black !important;
}

/* Number Input +/- buttons */
button[kind="secondary"] {
    background-color: white !important;
    color: black !important;
}

/* Input containers */
[data-testid="stNumberInput"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 5px !important;
}

/* ===== PREDICTION PAGE HEADINGS ===== */

.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #1B5E20 !important;
}

/* ===== SUCCESS CARDS ===== */

.success-card h1,
.success-card h2,
.success-card h3,
.success-card {
    color: white !important;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: white !important;
}
/* ===== SELECTBOX FIX ===== */

/* Main dropdown box */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border: 2px solid #dfe6e9 !important;
    border-radius: 10px !important;
}

/* Selected value */
div[data-baseweb="select"] span {
    color: black !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: black !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: white !important;
}

/* Dropdown options */
li[role="option"] {
    color: black !important;
    background-color: white !important;
}

/* Hover effect */
li[role="option"]:hover {
    background-color: #e8f5e9 !important;
    color: black !important;
}
.sidebar-logo{
    color:white !important;
    font-size:22px;
    font-weight:800;
    margin-bottom:20px;
}
/* =========================
   MOBILE RESPONSIVE FIXES
   ========================= */

@media (max-width: 768px) {

    /* Sidebar width */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 260px !important;
    }

    /* Sidebar title */
    .sidebar-logo {
        font-size: 28px !important;
        text-align: center;
    }

    /* Sidebar subtitle */
    [data-testid="stSidebar"] p {
        font-size: 14px !important;
        text-align: center;
    }
     [data-testid="stSidebar"]{
    width:260px !important;
    min-width:260px !important;
    }

    section[data-testid="stSidebar"]{
    width:260px !important;
    min-width:260px !important;
    }

    /* Navigation options */
    div[role="radiogroup"] label {
        width: 100% !important;
        padding: 8px !important;
        margin-bottom: 8px !important;
        border-radius: 12px !important;
        font-size: 15px !important;
    }

    /* Hero card */
    .hero-card {
        padding: 20px !important;
        border-radius: 18px !important;
    }

    .hero-card h1 {
        font-size: 32px !important;
    }

    .hero-card h3 {
        font-size: 18px !important;
    }

    .hero-card p {
        font-size: 14px !important;
    }

    /* Feature cards */
    .metric-card {
        padding: 15px !important;
    }

    .metric-card h2 {
        font-size: 18px !important;
    }

    .metric-card h3 {
        font-size: 14px !important;
    }

    /* Buttons */
    .stButton > button {
        font-size: 15px !important;
        height: 50px !important;
    }

    /* Inputs */
    .stNumberInput,
    .stSelectbox,
    .stTextInput {
        width: 100% !important;
    }
}

/* Top Navigation Buttons */
.stButton > button{
    background:#2E7D32 !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    height:55px !important;
    font-weight:700 !important;
}

/* Dropdown */
.stSelectbox div[data-baseweb="select"]{
    background:white !important;
    border-radius:12px !important;
    color:black !important;
}

/* Dropdown text */
.stSelectbox *{
    color:black !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:white;
padding:15px;
border-radius:15px;
margin-bottom:20px;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
border:1px solid #e0e0e0;
">
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,3])

with col1:
    home_btn = st.button(
        "🏠 Home",
        use_container_width=True
    )

with col2:
    about_btn = st.button(
        "ℹ️ About",
        use_container_width=True
    )

with col3:
    model = st.selectbox(
        "Models",
        [
            "Select Feature",
            "🌱 Crop Recommendation",
            "🌾 Crop Yield Prediction",
            "💰 Crop Price Prediction",
            "🩺 Plant Health Prediction",
            "📦 Post-Harvest Storage",
            "🚜 Equipment Sharing & Rental"
        ]
    )
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if home_btn:
    st.session_state.page = "🏠 Home"

if about_btn:
    st.session_state.page = "ℹ️ About"

if model != "Select Feature":
    st.session_state.page = model

page = st.session_state.page

if page == "🏠 Home":

    st.markdown("""
    <div class='hero-card'>
    <h1>🌾 AgriVision</h1>
    <h3>Smart Agriculture Analytics Platform</h3>
    <p>
    Transforming agriculture through intelligent predictions, resource sharing, and smart storage management.
    </p>
    </div>
    """, unsafe_allow_html=True) 

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h2>🌱 Crop Recommendation</h2>
        <h3>97% Accuracy</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h2>🩺 Crop Health</h2>
        <h3>99% Accuracy</h3>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class='metric-card'>
        <h2>🌾 Yield Prediction</h2>
        <h3>High R² Score</h3>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
        <h2>💰 Price Prediction</h2>
        <h3>High R² Score</h3>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    st.write("")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("""
        <div class='metric-card'>
        <h2>🚜 Equipment Sharing</h2>
        <h3>Rent Farm Machinery</h3>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class='metric-card'>
        <h2>📦 Post-Harvest Storage</h2>
        <h3>Shelf Life & Spoilage Risk</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="color:#1B5E20;">🚀 Features</h2>

    <div style="
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 12px rgba(0,0,0,0.1);
    color:black;
    ">

    ✅ Smart Crop Recommendation<br><br>
    ✅ Crop Yield Prediction<br><br>
    ✅ Crop Price Prediction<br><br>
    ✅ Crop Health Prediction<br><br>
    ✅ Equipment Sharing & Rental<br><br>
    ✅ Post-Harvest Storage Advisor<br><br>
    ✅ Mobile Friendly Interface<br><br>
    ✅ AI Powered Agricultural Insights

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    
elif page == "🌱 Crop Recommendation":
    st.title("🌱 Crop Recommendation")
    st.write("Enter soil and climate information.")
    col1, col2 = st.columns(2)
    with col1:
      N = st.number_input("Nitrogen (N)", min_value=0.0) 
      P = st.number_input("Phosphorus (P)", min_value=0.0) 
      K = st.number_input("Potassium (K)", min_value=0.0)
      temperature = st.number_input("Temperature (°C)")
    with col2:
      humidity = st.number_input("Humidity (%)") 
      ph = st.number_input("Soil pH") 
      rainfall = st.number_input("Rainfall (mm)")
    if st.button("🌱 Recommend Crop"):
      data = np.array([[ N, P, K, temperature, humidity, ph, rainfall ]])
      prediction = crop_rec_model.predict(data)
      crop_name = crop_encoder.inverse_transform(prediction)
      st.markdown(f"""
      <div class="success-card">
      Recommended Crop<br><br>
      {crop_name[0]}
      </div>
      """, unsafe_allow_html=True)

elif page == "🌾 Crop Yield Prediction":
    st.title("🌾 Crop Yield Prediction")
    crop = st.selectbox( "Crop", ['Arecanut','Arhar/Tur','Bajra','Banana','Barley','Black pepper',
                                  'Cardamom','Cashewnut','Castor seed','Coconut ','Coriander',
                                  'Cotton(lint)','Cowpea(Lobia)','Dry chillies','Garlic','Ginger',
                                  'Gram','Groundnut','Guar seed','Horse-gram','Jowar','Jute',
                                  'Khesari','Linseed','Maize','Masoor','Mesta',
                                  'Moong(Green Gram)','Moth','Niger seed','Oilseeds total',
                                  'Onion','Other Rabi pulses','Other Cereals',
                                  'Other Kharif pulses','Other Summer Pulses',
                                  'Peas & beans (Pulses)','Potato','Ragi',
                                  'Rapeseed &Mustard','Rice','Safflower','Sannhamp',
                                  'Sesamum','Small millets','Soyabean','Sugarcane',
                                  'Sunflower','Sweet potato','Tapioca','Tobacco',
                                  'Turmeric','Urad','Wheat','other oilseeds'] )
    season = st.selectbox( "Season", ['Autumn ','Kharif ','Rabi ', 'Summer ','Whole Year ','Winter '] )
    state = st.selectbox( "State", ['Andhra Pradesh','Arunachal Pradesh','Assam','Bihar',
                                    'Chhattisgarh','Delhi','Goa','Gujarat','Haryana',
                                    'Himachal Pradesh','Jammu and Kashmir','Jharkhand',
                                    'Karnataka','Kerala','Madhya Pradesh','Maharashtra',
                                    'Manipur','Meghalaya','Mizoram','Nagaland','Odisha',
                                    'Puducherry','Punjab','Sikkim','Tamil Nadu',
                                    'Telangana','Tripura','Uttar Pradesh',
                                    'Uttarakhand','West Bengal'] )
    crop_year = st.number_input("Crop Year", value=2024)
    area = st.number_input("Area")
    rainfall = st.number_input("Annual Rainfall")
    fertilizer = st.number_input("Fertilizer") 
    pesticide = st.number_input("Pesticide")
    if st.button("🌾 Predict Yield"):
      input_df = pd.DataFrame({ 'Crop':[crop],
                               'Crop_Year':[crop_year],
                               'Season':[season],
                               'State':[state],
                               'Area':[area],
                               'Production':[production], 
                               'Annual_Rainfall':[rainfall], 
                               'Fertilizer':[fertilizer], 
                               'Pesticide':[pesticide] 
      })
      prediction = crop_yield_model.predict(input_df)
      st.markdown(f"""
      <div class="success-card">
      <h2>Predicted Yield</h2>
      <h1>{prediction[0]:,.2f}</h1>
      </div>
      """, unsafe_allow_html=True)
        
elif page == "💰 Crop Price Prediction":
    st.title("💰 Crop Price Prediction")
    state = st.selectbox( "State", list(price_encoders["STATE"].classes_) )
    district = st.selectbox( "District", list(price_encoders["District Name"].classes_) )
    market = st.selectbox( "Market", list(price_encoders["Market Name"].classes_) )
    commodity = st.selectbox( "Commodity", list(price_encoders["Commodity"].classes_) )
    variety = st.selectbox( "Variety", list(price_encoders["Variety"].classes_) )
    grade = st.selectbox( "Grade", list(price_encoders["Grade"].classes_) )
    min_price = st.number_input("Minimum Price")
    max_price = st.number_input("Maximum Price")
    year = st.number_input("Year", value=2025) 
    month = st.number_input("Month", value=1) 
    day = st.number_input("Day", value=1)
    if st.button("💰 Predict Price"):
      input_df = pd.DataFrame({ "STATE":[ price_encoders["STATE"].transform([state])[0] ],
                               "District Name":[ price_encoders["District Name"].transform([district])[0] ],
                               "Market Name":[ price_encoders["Market Name"].transform([market])[0] ],
                               "Commodity":[ price_encoders["Commodity"].transform([commodity])[0] ],
                               "Variety":[ price_encoders["Variety"].transform([variety])[0] ],
                               "Grade":[ price_encoders["Grade"].transform([grade])[0] ],
                               "Min_Price":[min_price], 
                               "Max_Price":[max_price], 
                               "Year":[year], 
                               "Month":[month],
                               "Day":[day] 
      })
      prediction = crop_price_model.predict(input_df)
      st.markdown(f"""
      <div class="success-card">
      <h2>Predicted Crop Price</h2>
      <h1>₹ {prediction[0]:,.2f}</h1>
      </div>
      """, unsafe_allow_html=True)

elif page == "🩺 Crop Health Prediction":
    st.title("🩺 Crop Health Prediction")
    col1, col2 = st.columns(2)
    with col1:
      Soil_Moisture = st.number_input("Soil Moisture")
      Ambient_Temperature = st.number_input("Ambient Temperature")
      Soil_Temperature = st.number_input("Soil Temperature") 
      Humidity = st.number_input("Humidity") 
      Light_Intensity = st.number_input("Light Intensity")
      Soil_pH = st.number_input("Soil pH")
      Nitrogen_Level = st.number_input("Nitrogen Level") 
      Phosphorus_Level = st.number_input("Phosphorus Level")
    with col2:
      Potassium_Level = st.number_input("Potassium Level") 
      Chlorophyll_Content = st.number_input("Chlorophyll Content")
      Electrochemical_Signal = st.number_input("Electrochemical Signal")
      current = datetime.now()
      Year = st.number_input("Year", value=current.year)
      Month = st.number_input("Month", value=current.month)
      Day = st.number_input("Day", value=current.day) 
      Hour = st.number_input("Hour", value=current.hour)
    if st.button("🩺 Predict Crop Health"):
      input_df = pd.DataFrame([[ Soil_Moisture,
                                Ambient_Temperature,
                                Soil_Temperature, 
                                Humidity, 
                                Light_Intensity, 
                                Soil_pH,
                                Nitrogen_Level, 
                                Phosphorus_Level,
                                Potassium_Level,
                                Chlorophyll_Content, 
                                Electrochemical_Signal,
                                Year,
                                Month, 
                                Day,
                                Hour 
                               ]],columns=[ 'Soil_Moisture', 
                                           'Ambient_Temperature', 
                                           'Soil_Temperature',
                                           'Humidity',
                                           'Light_Intensity', 
                                           'Soil_pH',
                                           'Nitrogen_Level', 
                                           'Phosphorus_Level', 
                                           'Potassium_Level', 
                                           'Chlorophyll_Content', 
                                           'Electrochemical_Signal',
                                           'Year',
                                           'Month', 
                                           'Day', 
                                           'Hour' ])
      prediction = disease_model.predict(input_df)
      result = disease_encoder.inverse_transform(prediction)
      st.markdown(f"""
      <div class="success-card">
      <h2>Plant Health Status</h2>
      <h1>{result[0]}</h1>
      </div>
      """, unsafe_allow_html=True)

elif page == "🚜 Equipment Sharing & Rental":

    st.title("🚜 Equipment Sharing & Rental")

    menu = st.radio(
        "Select Option",
        [
            "List Equipment",
            "Search Equipment",
            "Equipment Recommendation"
        ]
    )

    # LIST EQUIPMENT
    if menu == "List Equipment":

        owner = st.text_input("Owner Name")
        location = st.text_input("Village / Location")

        equipment = st.selectbox(
            "Equipment Type",
            [
                "Tractor",
                "Harvester",
                "Rotavator",
                "Seeder",
                "Cultivator",
                "Sprayer"
            ]
        )

        rent = st.number_input("Rent Per Day (₹)")
        contact = st.text_input("Contact Number")

        if st.button("Add Equipment"):

            c.execute(
                """
                INSERT INTO equipment
                (
                owner_name,
                location,
                equipment_type,
                rent_per_day,
                contact
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    owner,
                    location,
                    equipment,
                    rent,
                    contact
                )
            )

            conn.commit()
            st.success("Equipment Listed Successfully")

    # SEARCH EQUIPMENT
    elif menu == "Search Equipment":

        equipment = st.selectbox(
            "Equipment",
            [
                "Tractor",
                "Harvester",
                "Rotavator",
                "Seeder",
                "Cultivator",
                "Sprayer"
            ]
        )

        if st.button("Search"):

            df = pd.read_sql_query(
                """
                SELECT *
                FROM equipment
                WHERE equipment_type = ?
                """,
                conn,
                params=(equipment,)
            )

            if len(df) > 0:
                st.dataframe(df)
            else:
                st.warning("No Equipment Available")

    # RECOMMENDATION
    elif menu == "Equipment Recommendation":

        crop = st.selectbox(
            "Crop",
            [
                "Wheat",
                "Rice",
                "Maize",
                "Cotton"
            ]
        )

        farm_size = st.number_input(
            "Farm Size (Acres)",
            min_value=1
        )

        if st.button("Recommend"):

            if farm_size <= 2:
                rec = "Power Tiller"
            elif farm_size <= 5:
                rec = "Tractor + Seeder"
            else:
                rec = "Tractor + Rotavator + Harvester"

            st.markdown(f"""
            <div class="success-card">
            Recommended Equipment<br><br>
            {rec}
            </div>
            """, unsafe_allow_html=True)

elif page == "ℹ️ About":

    st.title("ℹ️ About AgriVision")
    st.markdown("""
    AgriVision is an AI-powered agriculture
    platform developed using Machine Learning.

    Modules:
    - Crop Recommendation
    - Crop Yield Prediction
    - Crop Price Prediction
    - Crop Health Prediction
    - Equipment Sharing & Rental

    Developed using:
    - Python
    - Streamlit
    - Scikit-Learn
    - Machine Learning
    """)
  
  
