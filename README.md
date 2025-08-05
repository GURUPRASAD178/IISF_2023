# Space-Hackathon

# 🛰️ NLP-Driven Geospatial Processing Platform

An AI-powered platform that interprets natural language queries and visualizes relevant geospatial data using satellite imagery and open data sources like **ISRO Bhuvan**.

> 🚀 Developed as part of **IISF 2023 Space Hackathon – Problem Statement 3**:  
> *"Geo Processing Platform with NLP for Bhuvan"*

---

## 📌 Problem Statement

> Build a secure, lightweight, and intelligent geospatial data processing platform that allows users to query spatial datasets using **natural language**, and get actionable insights and visualizations—without knowing GIS.

Example query:  
🗣️ *“Show me NDVI in Karnataka between 2018 and 2022”*  
🗺️ → Returns map and charts based on satellite imagery.

---

## 🎯 Key Features

- ✅ Natural Language Query Parser (NLP)
- ✅ Query-to-Action JSON Converter
- ✅ Integration with geospatial datasets (Bhuvan/ISRO/Open Data)
- ✅ Dynamic Map and Graph Generation
- ✅ User-Friendly Interface (Streamlit or Flask)

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|-------------|
| NLP | spaCy, Regex, Transformers (optional) |
| Backend | Python (Flask / Streamlit) |
| Geospatial | GeoPandas, Rasterio, Folium, Shapely, GDAL |
| Visualization | Plotly, Matplotlib |
| Data Sources | ISRO Bhuvan, Open Government Data, Kaggle |
| UI | Streamlit (demo) or HTML+JS |

---

## 🧠 How It Works

1. **User Input**:  
   > “Show rainfall in Hassan from 2019 to 2022”

2. **NLP Parser**:  
   Extracts →  
   ```json
   {
     "region": "Hassan",
     "parameter": "rainfall",
     "start_year": 2019,
     "end_year": 2022
   }
