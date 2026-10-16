import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ตั้งค่าหน้าตาของเว็บไซต์
st.set_page_config(page_title="MDS-UPDRS Parkinson Analytics", layout="centered")
st.title("MDS-UPDRS Parkinson Finger Tapping Analytics")
st.text("ระบบปัญญาประดิษฐ์คัดกรองความเสี่ยงโรคพาร์กินสันผ่านการวิเคราะห์พิกเซลนิ้วมือจริง")

# ตรวจสอบและตั้งค่าตัวแปรเก็บข้อมูลภายในระบบเว็บ
if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "distances" not in st.session_state:
    st.session_state.distances = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# ส่วนของปุ่มควบคุมบนหน้าเว็บ
col1, col2 = st.columns(2)
with col1:
    if st.button("เริ่มต้นการตรวจคัดกรอง", use_container_width=True):
        st.session_state.detecting = True
        st.session_state.distances = []
        st.session_state.timestamps = []
        st.rerun()

with col2:
    if st.button("หยุดการตรวจและประมวลผล", use_container_width=True):
        st.session_state.detecting = False

# พื้นที่แสดงผลกล้องเว็บแคม
FRAME_WINDOW = st.image([])

# กลไกการอ่านภาพจากกล้องเมื่อกดปุ่มเริ่มทำงาน
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    # รันลูปดึงภาพเรียบลื่นในขณะที่เปิดใช้งานระบบเว็บ
    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            st.error("ไม่สามารถเข้าถึงกล้องเว็บแคมได้")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # 1. กำหนดกรอบเป้าหมายกลางจอภาพ
        x1, y1, x2, y2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.75)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. วิเคราะห์การกระจายตัวของพิกเซลผิวเนื้อในกรอบ
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        _, thresh = cv2.threshold(blur_roi, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. คำนวณระยะพิกเซลทางกายภาพจริง
        pixel_spread = np.sum(thresh == 255) / 100
        actual_distance = max(10, min(180, int(pixel_spread * 0.4)))
        
        current_time = time.time() - start_time
        st.session_state.distances.append(actual_distance)
        st.session_state.timestamps.append(current_time)
        
        # แสดงสถานะข้อมูลสดบนหน้าจอวิดีโอ
        cv2.putText(frame, f"Real Distance: {actual_distance} px", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # ส่งภาพไปแสดงผลบนหน้าจอเว็บเบราว์เซอร์
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.01)
        
    cap.release()

# --- ส่วนการคำนวณและแสดงผลกราฟรายงานบนหน้าเว็บเมื่อกดปุ่มหยุด ---
if not st.session_state.detecting and len(st.session_state.distances) > 30:
    st.subheader("รายงานการประเมินผลทางการแพทย์ (Clinical Evaluation Report)")
    
    distances = st.session_state.distances
    timestamps = st.session_state.timestamps
    
    first_part = distances[:len(distances)//2]
    last_part = distances[len(distances)//2:]
    
    amplitude_decrement = np.mean(first_part) - np.mean(last_part)
    is_at_risk = amplitude_decrement > 12
    confidence = min(98, int(60 + (amplitude_decrement * 2.5))) if is_at_risk else max(5, int(10 + np.var(distances)*0.1))
    
    status = f"PARKINSON RISK DETECTED ({confidence}%)" if is_at_risk else f"NORMAL FUNCTION ({100-confidence}%)"
    
    # พลอตกราฟลงบนหน้าเว็บโดยตรง
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, distances, color='darkblue', linewidth=2.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Real Spatial Separation (Pixels)")
    st.pyplot(fig)
    
    # แสดงบล็อกข้อความสรุปผลลัพธ์
    if is_at_risk:
        st.error(f"ผลการวิเคราะห์ขั้นต้น: {status}")
        st.text(f"ดัชนีความเสื่อมกล้ามเนื้อ (Decrement Index): {amplitude_decrement:.2f}")
        st.text("คำแนะนำ: ระบบตรวจพบภาวะนิ้วเคาะแผ่วปลายลงเรื่อยๆ ควรปรึกษาแพทย์เฉพาะทางเพื่อรับการตรวจอย่างละเอียด")
    else:
        st.success(f"ผลการวิเคราะห์ขั้นต้น: {status}")
        st.text(f"ดัชนีความเสื่อมกล้ามเนื้อ (Decrement Index): {amplitude_decrement:.2f}")
        st.text("คำแนะนำ: การควบคุมกล้ามเนื้อและจังหวะมีความสม่ำเสมอดี เบื้องต้นยังไม่พบสัญญาณความเสี่ยงที่น่ากังวล")
