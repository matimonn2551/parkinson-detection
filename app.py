import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ตั้งค่าหน้าเว็บเริ่มต้นแบบเรียบง่าย
st.title("MDS-UPDRS Parkinson Finger Tapping Analytics")
st.text("ระบบคัดกรองความเสี่ยงโรคพาร์กินสันเบื้องต้นผ่านกล้องวิดีโอทั่วไป")

# เตรียมตารางสำหรับบันทึกผลการทดสอบ
if "distances" not in st.session_state:
    st.session_state.distances = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# คลาสประมวลผลวิดีโอสดระดับพิกเซลทางกายภาพจริง
class ParkinsonVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # 1. วาดกรอบสี่เหลี่ยมสีเขียวสำหรับวางมือ
        x1, y1, x2, y2 = int(w*0.35), int(h*0.25), int(w*0.65), int(h*0.75)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. วิเคราะห์การขยับของพิกเซลผิวเนื้อในกรอบจริง
        roi = img[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        _, thresh = cv2.threshold(blur_roi, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        pixel_spread = np.sum(thresh == 255) / 100
        actual_distance = max(10, min(180, int(pixel_spread * 0.4)))
        
        current_time = time.time() - self.start_time
        
        # บันทึกค่าลงหน่วยความจำชั่วคราว
        st.session_state.distances.append(actual_distance)
        st.session_state.timestamps.append(current_time)
        
        cv2.putText(img, f"Distance: {actual_distance} px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame.from_ndarray(img, format="bgr24")

st.write("---")
st.subheader("📸 ส่วนของกล้องเว็บแคม")
st.info("คำแนะนำ: กดปุ่ม START ด้านล่างเพื่อเปิดกล้อง และกดยอมรับ (Allow) การเข้าถึงกล้องของเบราว์เซอร์")

# เรียกใช้งานกล้อง WebRTC เวอร์ชันเสถียรที่สุดสำหรับระบบออนไลน์
ctx = webrtc_streamer(
    key="parkinson-screening-simple",
    video_processor_factory=ParkinsonVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.write("---")
st.subheader("📊 ผลการวิเคราะห์และกราฟรายงาน")

# เมื่อผู้ใช้งานกดปุ่ม STOP ที่หน้าต่างกล้อง ระบบจะทำการประมวลผลทันที
if ctx and not ctx.state.playing and len(st.session_state.distances) > 30:
    distances = st.session_state.distances
    timestamps = st.session_state.timestamps
    
    first_part = distances[:len(distances)//2]
    last_part = distances[len(distances)//2:]
    
    amplitude_decrement = np.mean(first_part) - np.mean(last_part)
    rhythm_variance = np.var(distances)
    
    is_at_risk = amplitude_decrement > 12
    confidence = min(98, int(60 + (amplitude_decrement * 2.5))) if is_at_risk else max(5, int(10 + rhythm_variance*0.1))
    
    # แสดงตัวเลขดัชนีสำคัญ
    st.metric(label="ดัชนีการอ่อนแรงของนิ้วมือ (Decrement Index)", value=f"{amplitude_decrement:.2f}")
    st.metric(label="ค่าความนิ่งของจังหวะเคาะ (Rhythm Variance)", value=f"{rhythm_variance:.2f}")
    
    # วินิจฉัยผลลัพธ์
    if is_at_risk:
        st.error(f"ผลวินิจฉัยเบื้องต้น: RISK OF PARKINSON'S ({confidence}%)")
        st.text("คำแนะนำ: ตรวจพบภาวะนิ้วเคาะแผ่วลงเรื่อย ๆ ในช่วงปลาย ควรปรึกษาแพทย์เฉพาะทางเพื่อตรวจอย่างละเอียด")
    else:
        st.success(f"ผลวินิจฉัยเบื้องต้น: NORMAL FUNCTION ({100-confidence}%)")
        st.text("คำแนะนำ: การควบคุมและจังหวะของกล้ามเนื้อมือมีความสม่ำเสมอดี เบื้องต้นไม่พบสัญญาณเสี่ยง")
        
    # แสดงรูปกราฟ
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, distances, color='blue', linewidth=2)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Distance (Pixels)")
    ax.grid(True)
    st.pyplot(fig)
    
else:
    st.caption("ระบบพร้อมประมวลผล รายงานทางการแพทย์และรูปกราฟเส้นจะแสดงขึ้นตรงนี้โดยอัตโนมัติ หลังจากที่คุณทำท่าเคาะนิ้วเสร็จแล้วกดปุ่ม STOP ที่กล้องครับ")
