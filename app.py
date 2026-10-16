import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. ตั้งค่าโครงสร้างหน้าเว็บแบบขยายเต็มจอ
st.set_page_config(
    page_title="Parkinson Motion Analytics Tool",
    layout="wide"
)

# 2. ปรับตัวสะกดเป็น unsafe_allow_html=True (ไม่มีตัว e) ถูกต้องตาม Logs แจ้งเตือน
st.markdown("""
    <style>
        .main {
            background-color: #fcfcfd;
        }
        .header-box {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .header-box h1 {
            color: #f8fafc !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem;
        }
        .result-card-normal {
            background-color: #f0fdf4;
            border-left: 6px solid #16a34a;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .result-card-risk {
            background-color: #fef2f2;
            border-left: 6px solid #dc2626;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            color: #0f172a !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. ส่วนหัวเว็บไล่เฉดสี ปรับตัวสะกดให้ถูกต้อง
st.markdown("""
    <div class="header-box">
        <h1>Parkinson Motion Analytics</h1>
        <p style="font-size: 1.1rem; opacity: 0.85; margin: 0;">
            นวัตกรรมคัดกรองและบันทึกแนวโน้มสุขภาพของระบบสั่งการกล้ามเนื้อมือผ่านกล้องวิดีโอทั่วไป
        </p>
    </div>
""", unsafe_allow_html=True)

# สร้างพื้นที่แบ่งฝั่ง ซ้าย-ขวา
col_left, col_right = st.columns([1, 1.1], gap="large")

with col_left:
    st.markdown("### 📸 หน้าต่างควบคุมระบบกล้อง")
    
    if "detecting" not in st.session_state:
        st.session_state.detecting = False
    if "distances" not in st.session_state:
        st.session_state.distances = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []

    # ปุ่มควบคุมหน้าเว็บขนาดใหญ่
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("เปิดกล้องเริ่มทดสอบ", type="primary", use_container_width=True):
            st.session_state.detecting = True
            st.session_state.distances = []
            st.session_state.timestamps = []
            st.rerun()
    with btn_col2:
        if st.button("กดเพื่อหยุดและประมวลผล", type="secondary", use_container_width=True):
            st.session_state.detecting = False

    # พื้นที่แสดงผลวิดีโอ
    FRAME_WINDOW = st.image([])
    
    if not st.session_state.detecting and len(st.session_state.distances) == 0:
        st.info("ระบบพร้อมใช้งาน กรุณากดปุ่มสีน้ำเงินเพื่อเปิดระบบกล้องบันทึกภาพ")

# ทำงานดึงภาพจากกล้อง
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # วาดกรอบเป้าหมายสีเขียวนีออน
        x1, y1, x2, y2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.75)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        _, thresh = cv2.threshold(blur_roi, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        pixel_spread = np.sum(thresh == 255) / 100
        actual_distance = max(10, min(180, int(pixel_spread * 0.4)))
        
        current_time = time.time() - start_time
        st.session_state.distances.append(actual_distance)
        st.session_state.timestamps.append(current_time)
        
        cv2.putText(frame, f"Tracking... {actual_distance} px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (57, 255, 20), 2)
        
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.01)
        
    cap.release()

# ฝั่งขวา: แสดงผลรายงานและการวิเคราะห์แบบ Dashboard
with col_right:
    st.markdown("### 📈 สรุปรายงานการวิเคราะห์พฤติกรรม")
    
    if not st.session_state.detecting and len(st.session_state.distances) > 30:
        distances = st.session_state.distances
        timestamps = st.session_state.timestamps
        
        first_part = distances[:len(distances)//2]
        last_part = distances[len(distances)//2:]
        
        amplitude_decrement = np.mean(first_part) - np.mean(last_part)
        rhythm_variance = np.var(distances)
        
        is_at_risk = amplitude_decrement > 12
        confidence = min(98, int(60 + (amplitude_decrement * 2.5))) if is_at_risk else max(5, int(10 + rhythm_variance*0.1))
        
        # แสดงค่าดัชนีคณิตศาสตร์ผ่านโมดูล Metric ดีไซน์สวยงาม
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="ดัชนีการอ่อนแรงของนิ้วมือ (Decrement Index)", value=f"{amplitude_decrement:.2f}")
        with m_col2:
            st.metric(label="ค่าความนิ่งของจังหวะเคาะ (Rhythm Variance)", value=f"{rhythm_variance:.2f}")
            
        st.write("")
        
        # แสดงผลลัพธ์ผ่านบล็อกกล่อง HTML ที่ปรับแต่งตัวสะกดเรียบร้อย
        if is_at_risk:
            st.markdown(f"""
                <div class="result-card-risk">
                    <h4 style="color: #dc2626; margin-top: 0;">พบความเปลี่ยนแปลงในการเคลื่อนไหว ({confidence}%)</h4>
                    <p style="color: #7f1d1d; margin-bottom: 0;">
                        ระบบตรวจพบแนวโน้มที่ความกว้างในการกางนิ้วลดลงเรื่อยๆ ในช่วงท้ายของการทดสอบ 
                        ซึ่งเป็นสัญญาณสะท้อนของความเหนื่อยล้าของกลุ่มกล้ามเนื้อสั่งการ แนะนำให้เก็บบันทึกผลนี้เพื่อใช้ปรึกษาแพทย์ร่วมด้วย
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-normal">
                    <h4 style="color: #16a34a; margin-top: 0;">การเคลื่อนไหวอยู่ในเกณฑ์ปกติ ({100-confidence}%)</h4>
                    <p style="color: #14532d; margin-bottom: 0;">
                        ความกว้างของนิ้วมือและรอบจังหวะในการทดสอบมีความสม่ำเสมอดีตลอดรอดฝั่ง 
                        ไม่พบบ่งชี้ของการช้าลงหรือระยะกางที่หดแคบลงที่เป็นนัยสำคัญ
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("**กราฟแสดงทิศทางการกางนิ้วตามจริง (Movement Trace Graph)**")
        
        # พลอตกราฟ
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(timestamps, distances, color='#2563eb', linewidth=2.5)
        ax.set_xlabel("Time (Seconds)", fontsize=9, color="#64748b")
        ax.set_ylabel("Distance (Pixels)", fontsize=9, color="#64748b")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        st.pyplot(fig)
        
    elif not st.session_state.detecting:
        st.caption("หลังจากทำการทดสอบและกดปุ่มประมวลผลเสร็จสิ้น หน้าต่างกราฟและสถิติประเมินแบบละเอียดจะดีดขึ้นตรงนี้ทันที")
