from pathlib import Path
import joblib,pandas as pd,streamlit as st
from src.prediction import predict_risk
from src.data_analysis import load_data,feature_importance,create_feature_importance_chart
st.set_page_config(page_title='KKU Smart Flood AI',page_icon='🌊',layout='wide')
st.title('🌊 KKU Smart Flood AI'); st.subheader('ระบบ AI สำหรับประเมินความเสี่ยงน้ำท่วมจากข้อมูลสภาพแวดล้อม')
st.info('Prototype เพื่อการศึกษา ใช้ Synthetic Dataset ไม่ใช่ระบบพยากรณ์น้ำท่วมจริง')
if not Path('models/flood_model.pkl').exists(): st.error('ไม่พบโมเดล กรุณารัน generate_data.py และ train_model.py ก่อน'); st.stop()
b=joblib.load('models/flood_model.pkl'); df=load_data()
if 'history' not in st.session_state: st.session_state.history=[]
st.header('1. ป้อนข้อมูล'); c1,c2,c3,c4=st.columns(4)
with c1: rainfall=st.number_input('ปริมาณฝน (mm)',0.,250.,50.,1.)
with c2: water=st.number_input('ระดับน้ำ (m)',.05,2.5,.8,.05)
with c3: humidity=st.number_input('ความชื้น (%)',35.,100.,78.,1.)
with c4: temperature=st.number_input('อุณหภูมิ (°C)',18.,40.,29.,.5)
if st.button('🔎 วิเคราะห์ความเสี่ยง',type='primary',use_container_width=True):
 p,q=predict_risk(rainfall,water,humidity,temperature); st.session_state.prediction=p; st.session_state.probabilities=q
 st.session_state.history.insert(0,{'Rainfall (mm)':rainfall,'Water Level (m)':water,'Humidity (%)':humidity,'Temperature (°C)':temperature,'Prediction':p,'Confidence (%)':round(q[p]*100,2)})
if 'prediction' in st.session_state:
 p=st.session_state.prediction; q=st.session_state.probabilities; labels={'LOW':'🟢 ความเสี่ยงต่ำ','MEDIUM':'🟡 ความเสี่ยงปานกลาง','HIGH':'🔴 ความเสี่ยงสูง'}
 st.header('2. ผลการวิเคราะห์ AI'); st.success(f'ผลการประเมิน: {labels[p]}'); a,c,d=st.columns(3); a.metric('LOW',f"{q.get('LOW',0)*100:.2f}%"); c.metric('MEDIUM',f"{q.get('MEDIUM',0)*100:.2f}%"); d.metric('HIGH',f"{q.get('HIGH',0)*100:.2f}%")
st.header('3. ประสิทธิภาพโมเดล'); a,c,d=st.columns(3); a.metric('Test Accuracy',f"{b['accuracy']*100:.2f}%"); c.metric('จำนวนต้นไม้',b['model'].n_estimators); d.metric('จำนวนข้อมูล',len(df)); st.caption('Accuracy นี้มาจาก Synthetic Dataset จึงไม่ใช่ความแม่นยำของระบบจริง')
st.header('4. AI ให้ความสำคัญกับปัจจัยใดมากที่สุด?'); imp=feature_importance(); st.dataframe(imp.style.format({'Importance':'{:.4f}'}),use_container_width=True,hide_index=True)
if not Path('images/feature_importance.png').exists(): create_feature_importance_chart()
st.image('images/feature_importance.png',use_container_width=True)
st.header('5. วิเคราะห์ข้อมูล'); x,y=st.columns(2)
with x: st.subheader('จำนวนข้อมูลแต่ละระดับ'); st.bar_chart(df['Risk'].value_counts())
with y: st.subheader('ตัวอย่างข้อมูล'); st.dataframe(df.head(10),use_container_width=True)
st.header('6. ประวัติการวิเคราะห์')
if st.session_state.history:
 st.dataframe(pd.DataFrame(st.session_state.history),use_container_width=True)
 if st.button('ล้างประวัติ'): st.session_state.history=[]; st.rerun()
else: st.write('ยังไม่มีประวัติ')
st.header('7. เกี่ยวกับโครงงาน'); st.markdown('''**KKU Smart Flood AI** เป็น AI Prototype ที่นำ Machine Learning มาใช้จำแนกความเสี่ยงน้ำท่วมเป็น LOW / MEDIUM / HIGH จากปัจจัยด้านสิ่งแวดล้อม\n\n**เทคโนโลยี:** Python, Streamlit, Pandas, NumPy, Scikit-learn, Random Forest, Matplotlib\n\n**การพัฒนาต่อ:** ใช้ข้อมูลจริงจากสถานีวัดฝน/ระดับน้ำ, Real-time API, แผนที่พื้นที่เสี่ยง และระบบแจ้งเตือน'''); st.warning('Prototype นี้ไม่ได้เชื่อมต่อข้อมูล Real-time และไม่ควรใช้ตัดสินใจด้านความปลอดภัยจริง')
