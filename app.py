import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# 1. ตั้งค่าโครงสร้างหน้าเว็บแบบขยายเต็มจอ
st.set_page_config(
    page_title="Parkinson Motion Analytics Tool",
    layout="wide"
)

# แก้ไขรูปแบบ ICE Servers ให้ถูกต้องตามมาตรฐานเบราว์เซอร์สากล (ข้าม Error Invalid hostname)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 2. ฝังโค้ด CSS สำหรับดีไซน์หน้าเว็บให้สวยงาม โค้งมน มีมิติ
st.markdown("""
    <style>
        .main { background-color: #fcfcfd; }
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

# 3. ส่วนหัวเว็บไล่เฉดสีพรีเมียม
st.markdown("""
    <div class="header-box">
        <h1>Parkinson Motion Analytics</h1>
        <p style="font-size: 1.1rem; opacity: 0.85; margin: 0;">
            นวัตกรรมคัดกรองและบันทึกแนวโน้มสุขภาพของระบบสั่งการกล้ามเนื้อมือผ่านกล้องวิดีโอทั่วไปออนไลน์ 24 ชม.
        </p>
    </div>
""", unsafe_allow_html=True)

# สร้างพื้นที่แบ่งฝั่ง ซ้าย-ขวา
col_left, col_right = st.columns([1, 1.1], gap="large")

# ตัวเก็บข้อมูลระยะห่างที่เกิดขึ้นระหว่างรันระบบเว็บ
if "distances" not in st.session_state:
    st.session_state.distances = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# คลาสประมวลผลวิดีโอสดในระดับพิกเซลบนระบบอินเทอร์เน็ต (WebRTC Video Processor)
class ParkinsonVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # 1. กำหนดกรอบเป้าหมายสีเขียวนีออน
        x1, y1, x2, y2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.75)
        cv2.rectangle(img, (x1, y1), (x2, y2), (57, 255, 20), 2)
        
        # 2. วิเคราะห์โครงสร้างความหนาแน่นพิกเซลผิวเนื้อจริง
        roi = img[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        _, thresh = cv2.threshold(blur_roi, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        pixel_spread = np.sum(thresh == 255) / 100
        actual_distance = max(10, min(180, int(pixel_spread * 0.4)))
        
        current_time = time.time() - self.start_time
        
        # ส่งค่ากลับไปเก็บในระบบส่วนกลางแบบปลอดภัย
        st.session_state.distances.append(actual_distance)
        st.session_state.timestamps.append(current_time)
        
        cv2.putText(img, f"Tracking: {actual_distance} px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (57, 255, 20), 2)
        
        return frame.from_ndarray(img, format="bgr24")

with col_left:
    st.markdown("### 📸 ระบบกล้องอัจฉริยะผ่านเว็บ (Webcam Terminal)")
    st.info("กรุณากดปุ่ม START ด้านล่าง และกดยอมรับ (Allow) เพื่อให้เบราว์เซอร์เปิดกล้องเว็บแคมของคุณ")
    
    # ดึงค่าโครงสร้างแบบคลีนส่งเข้าระบบออนไลน์
    ctx = webrtc_streamer(
        key="parkinson-screening",
        video_processor_factory=ParkinsonVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

# ฝั่งขวา: แสดงผลรายงานและการวิเคราะห์แบบ Dashboard
with col_right:
    st.markdown("### 📈 สรุปรายงานการวิเคราะห์พฤติกรรม")
    
    # ตรวจสอบปุ่มหยุดทำงานของกล้องเพื่อทำการวิเคราะห์
    if ctx and not ctx.state.playing and len(st.session_state.distances) > 30:
        distances = st.session_state.distances
        timestamps = st.session_state.timestamps
        
        first_part = distances[:len(distances)//2]
        last_part = distances[len(distances)//2:]
        
        amplitude_decrement = np.mean(first_part) - np.mean(last_part)
        rhythm_variance = np.var(distances)
        
        is_at_risk = amplitude_decrement > 12
        confidence = min(98, int(60 + (amplitude_decrement * 2.5))) if is_at_risk else max(5, int(10 + rhythm_variance*0.1))
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="ดัชนีการอ่อนแรงของนิ้วมือ (Decrement Index)", value=f"{amplitude_decrement:.2f}")
        with m_col2:
            st.metric(label="ค่าความนิ่งของจังหวะเคาะ (Rhythm Variance)", value=f"{rhythm_variance:.2f}")
            
        st.write("")
        
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
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(timestamps, distances, color='#2563eb', linewidth=2.5)
        ax.set_xlabel("Time (Seconds)", fontsize=9, color="#64748b")
        ax.set_ylabel("Distance (Pixels)", fontsize=9, color="#64748b")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        
    else:
        st.caption("ระบบพร้อมประมวลผล เมื่อคุณกดปุ่ม STOP ที่หน้าต่างกล้องเพื่อสิ้นสุดการทดสอบ รายงานและกราฟวิเคราะห์โรคจะดีดขึ้นแสดงผลตรงนี้ทันที")
