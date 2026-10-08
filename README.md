# 🌊 KKU Smart Flood AI

AI Prototype สำหรับประเมินความเสี่ยงน้ำท่วมจาก Rainfall, Water Level, Humidity และ Temperature

## ⚠️ Disclaimer
โปรเจกต์นี้ใช้ **Synthetic Dataset** เพื่อการศึกษาและสาธิต ไม่ใช่ข้อมูลน้ำท่วมจริง ดังนั้น Accuracy ไม่ควรนำไปอ้างว่าเป็นความแม่นยำของระบบจริง

## Features
- Random Forest Classification
- LOW / MEDIUM / HIGH risk
- Probability ของแต่ละระดับ
- Feature Importance
- Data Analysis
- Prediction History
- Streamlit Dashboard

## Run
```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
python src/data_analysis.py
streamlit run app.py
```

## Future Development
- ข้อมูลจริงจากสถานีตรวจวัด
- Real-time weather/water API
- แผนที่พื้นที่เสี่ยง
- ระบบแจ้งเตือน
- เปรียบเทียบโมเดลหลายชนิด
