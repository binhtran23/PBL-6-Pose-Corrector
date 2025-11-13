import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import os

# Khởi tạo các giải pháp MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Kích thước Video Cố định ---
OUTPUT_DIM = (1280, 720) # (Width, Height)

# --- Nạp Font Unicode (Tiếng Việt) ---
# Sử dụng một font TrueType hỗ trợ Tiếng Việt
# Tệp font này phải tồn tại
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 
# Nếu dùng Linux/Mac, bạn cần đổi đường dẫn này, ví dụ: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Kiểm tra xem tệp font có tồn tại không
if not os.path.exists(FONT_PATH):
    print(f"Cảnh báo: Không tìm thấy font tại {FONT_PATH}. Sẽ dùng font mặc định (có thể lỗi Tiếng Việt).")
    # Trên Linux/Mac, Pillow thường có thể tìm thấy font mặc định
    try:
        font_sm = ImageFont.load_default()
        font_md = ImageFont.load_default()
        font_lg = ImageFont.load_default()
        font_reason = ImageFont.load_default()
    except Exception as e:
        print(f"Không thể nạp font mặc định: {e}")
        # Thoát nếu không thể nạp bất kỳ font nào
        raise SystemExit("Lỗi nghiêm trọng: Không thể nạp font chữ.")
else:
    try:
        font_sm = ImageFont.truetype(FONT_PATH, 20)
        font_md = ImageFont.truetype(FONT_PATH, 25)
        font_lg = ImageFont.truetype(FONT_PATH, 30)
        font_reason = ImageFont.truetype(FONT_PATH, 40)
    except IOError:
        print(f"Lỗi khi nạp font từ {FONT_PATH}. Sử dụng font mặc định.")
        font_sm = ImageFont.load_default()
        font_md = ImageFont.load_default()
        font_lg = ImageFont.load_default()
        font_reason = ImageFont.load_default()

# --- Hàm trợ giúp Vẽ Unicode ---
def draw_text_unicode(image_pil, text, position, font, fill=(255, 255, 255)):
    """Vẽ văn bản Unicode lên ảnh PIL."""
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=fill)
    
# === HÀM MỚI ĐỂ SỬA LỖI MÉO HÌNH (ASPECT RATIO FIX) ===
def resize_and_pad(image, output_dim=(1280, 720)):
    """
    Resize ảnh và giữ nguyên tỷ lệ, thêm dải đen (padding)
    để vừa vặn với kích thước output_dim.
    """
    h, w = image.shape[:2]
    # Kiểm tra nếu ảnh không hợp lệ
    if h == 0 or w == 0:
        print("Cảnh báo: Nhận được frame ảnh không hợp lệ (height hoặc width = 0)")
        # Trả về một canvas đen
        return np.zeros((output_dim[1], output_dim[0], 3), dtype=np.uint8)

    out_w, out_h = output_dim
    scale = min(out_w / w, out_h / h)
    
    # Tính toán kích thước resize mới
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize ảnh
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tạo một canvas đen (ảnh nền)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Tính toán vị trí để dán ảnh đã resize vào giữa
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    
    # Dán ảnh vào
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
    
    # Trả về canvas, scale và padding (Lượt 3 không cần scale/pad)
    return canvas, scale, (pad_x, pad_y)
# ======================================================== 

# --- Hàm trợ giúp Tính toán ---
def calculate_distance(a, b):
    try:
        return math.sqrt((a['px'] - b['px'])**2 + (a['py'] - b['py'])**2)
    except:
        return 0

def calculate_angle(a, b, c):
    # (a, b, c là các dict)
    try:
        a_np = np.array([a['x'], a['y']])
        b_np = np.array([b['x'], b['y']])
        c_np = np.array([c['x'], c['y']])
        
        radians = np.arctan2(c_np[1]-b_np[1], c_np[0]-b_np[0]) - np.arctan2(a_np[1]-b_np[1], a_np[0]-b_np[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    except:
        return 0 # Trả về 0 nếu có lỗi (ví dụ: thiếu keypoint)

def get_view(lm_coords):
    """Xác định góc nhìn dựa trên visibility và vị trí."""
    try:
        vis_left_hip = lm_coords.get(23, {}).get('vis', 0)
        vis_right_hip = lm_coords.get(24, {}).get('vis', 0)
        vis_left_shoulder = lm_coords.get(11, {}).get('vis', 0)
        vis_right_shoulder = lm_coords.get(12, {}).get('vis', 0)
        
        nose_x = lm_coords.get(0, {}).get('x', 0.5)
        left_shoulder_x = lm_coords.get(11, {}).get('x', 0)
        right_shoulder_x = lm_coords.get(12, {}).get('x', 0)

        # Ưu tiên kiểm tra visibility
        if (vis_left_hip > 0.8 and vis_right_hip < 0.2) or (vis_left_shoulder > 0.8 and vis_right_shoulder < 0.2):
            return "Side (Right View)" # Thấy bên trái -> nhìn về bên phải
        if (vis_right_hip > 0.8 and vis_left_hip < 0.2) or (vis_right_shoulder > 0.8 and vis_left_shoulder < 0.2):
            return "Side (Left View)" # Thấy bên phải -> nhìn về bên trái

        # Nếu cả hai đều rõ
        if vis_left_hip > 0.8 and vis_right_hip > 0.8:
            return "Front"
            
        # Dự đoán dựa trên vị trí tương đối
        if right_shoulder_x > nose_x and left_shoulder_x > nose_x:
             return "Side (Left View)"
        if right_shoulder_x < nose_x and left_shoulder_x < nose_x:
             return "Side (Right View)"

        return "Front"
    except Exception:
        return "Unknown"

# === LƯỢT 1: HÀM PHÂN TÍCH ===
def analyze_video_to_dataframe(video_path):
    """
    Xử lý video, phát hiện lỗi rule-based và trả về DataFrame thô.
    """
    print(f"[Analyze] Bắt đầu phân tích: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Analyze] Lỗi: Không thể mở video '{video_path}'")
        return pd.DataFrame(), 0, {}

    # Lấy FPS gốc
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 1000: # Một số video lỗi có thể trả về fps rất cao
        print(f"[Analyze] Cảnh báo: FPS không hợp lệ ({fps}), đặt mặc định là 30.")
        fps = 30
        
    w_origin = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_origin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Analyze] Video gốc: {w_origin}x{h_origin} @ {fps:.2f} FPS")
    print(f"[Analyze] Resize video về: {OUTPUT_DIM}")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_frames_data = [] # Danh sách để lưu trữ dữ liệu từng frame
    landmarks_header = []
    for i in range(33):
        landmarks_header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_vis'])

    frame_id = 0
    rep_count = 0
    state = "down"
    view = "Initializing"
    
    current_rep_frames = 0
    rep_label = "N/A" # Dùng để quản lý trạng thái rep
    
    last_frame_time = time.time()
    last_wrist_pos = {'left': None, 'right': None}
    
    rep_time_boundaries = {} # {rep_id: {"start_frame": int, "end_frame": int}}

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        frame_id += 1
        
        # --- SỬA LỖI MÉO HÌNH (LƯỢT 1) ---
        # Thay vì ép (resize), chúng ta resize và pad (thêm viền đen)
        try:
            # image = cv2.resize(image, OUTPUT_DIM, interpolation=cv2.INTER_AREA) # <-- CODE CŨ
            image, _, _ = resize_and_pad(image, OUTPUT_DIM) # <-- CODE MỚI
        except Exception as e:
            print(f"Lỗi resize frame {frame_id}: {e}")
            continue
        # --- KẾT THÚC SỬA LỖI ---
            
        h, w = OUTPUT_DIM[1], OUTPUT_DIM[0] # (h, w) = (720, 1280)

        # --- Xử lý MediaPipe ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        current_frame_error = "" # Lỗi rule-based của frame này
        lm_coords = {}
        
        # --- CẬP NHẬT: Luôn điền dữ liệu ---
        lm_data_row = [0.0] * (33 * 4) # Khởi tạo 132 cột (33*4) với giá trị 0

        if results.pose_landmarks:
            landmarks_raw = results.pose_landmarks.landmark
            lm_data_row_temp = []
            
            for id, lm in enumerate(landmarks_raw):
                # lm.x, lm.y là tọa độ (0-1) tương đối với ảnh 1280x720 (đã pad)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_coords[id] = {
                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'vis': lm.visibility,
                    'px': cx, 'py': cy
                }
                lm_data_row_temp.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            # Nếu phát hiện đủ 33 landmarks (132 giá trị)
            if len(lm_data_row_temp) == 132:
                 lm_data_row = lm_data_row_temp
            # --- KẾT THÚC CẬP NHẬT ---

            # --- Task 1: Xác định View ---
            view = get_view(lm_coords)

            # --- Tính góc vai (cần cho cả Task 3 và Task 2) ---
            avg_shoulder_angle = 0
            try:
                left_shoulder_angle = calculate_angle(lm_coords[13], lm_coords[11], lm_coords[23])
                right_shoulder_angle = calculate_angle(lm_coords[14], lm_coords[12], lm_coords[24])
                avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
            except Exception:
                pass 

            # --- Task 3: Đếm Rep (State Machine) ---
            try:
                up_threshold = 75
                down_threshold = 30
                new_state = state

                if avg_shoulder_angle > up_threshold:
                    new_state = "up"
                elif avg_shoulder_angle < down_threshold:
                    new_state = "down"
                
                # Bắt đầu rep (down -> up)
                if new_state == "up" and state == "down":
                    rep_label = "In Progress"
                    rep_count += 1 # Đếm rep khi BẮT ĐẦU
                    print(f"[Analyze] Bat dau Rep {rep_count} tai frame {frame_id}")
                    # Ghi lại frame bắt đầu
                    rep_time_boundaries[rep_count] = {"start_frame": frame_id, "end_frame": frame_id}
                
                # Kết thúc rep (up -> down)
                elif new_state == "down" and state == "up":
                    rep_label = f"Rep {rep_count} Done"
                    print(f"[Analyze] Ket thuc Rep {rep_count} tai frame {frame_id}")
                    # Cập nhật frame kết thúc
                    if rep_count in rep_time_boundaries:
                        rep_time_boundaries[rep_count]["end_frame"] = frame_id
                state = new_state
            
            except Exception:
                pass

            # --- Task 2: Heuristics kiểm tra lỗi (theo từng frame) ---
            # Chỉ kiểm tra lỗi NẾU rep đang diễn ra
            if rep_label == "In Progress":
                
                # Lỗi 1: Lưng thẳng (chỉ check side view)
                if "Side" in view:
                    try:
                        side_idx = 23 if "Left View" in view else 24
                        ear = lm_coords[7] if "Left View" in view else 8
                        shoulder = lm_coords[11] if "Left View" in view else 12
                        hip = lm_coords[side_idx]
                        back_angle = calculate_angle(ear, shoulder, hip)
                        if back_angle > 165:
                            current_frame_error = "Lỗi: Lưng quá thẳng"
                    except: pass

                # Lỗi 2: Tay lên quá cao
                try:
                    if avg_shoulder_angle > 115: 
                        current_frame_error = "Lỗi: Tay lên quá cao"
                except: pass

                # Lỗi 3: Nhún vai
                try:
                    l_ear_y, l_shoulder_y = lm_coords[7]['py'], lm_coords[11]['py']
                    r_ear_y, r_shoulder_y = lm_coords[8]['py'], lm_coords[12]['py']
                    left_shrug_dist = abs(l_ear_y - l_shoulder_y)
                    right_shrug_dist = abs(r_ear_y - r_shoulder_y)
                    shoulder_width = calculate_distance(lm_coords[11], lm_coords[12])
                    if shoulder_width > 0:
                        shrug_ratio = (left_shrug_dist + right_shrug_dist) / (2 * shoulder_width)
                        if shrug_ratio < 0.15: # Nếu khoảng cách tai-vai quá nhỏ
                            current_frame_error = "Lỗi: Nhún vai"
                except: pass

                # Lỗi 4: Vung tay quá nhanh
                try:
                    current_time = time.time()
                    delta_time = current_time - last_frame_time
                    if delta_time > 0.01: 
                        left_wrist, right_wrist = lm_coords[15], lm_coords[14]
                        if last_wrist_pos['left'] is not None:
                            left_dist = calculate_distance(left_wrist, last_wrist_pos['left'])
                            right_dist = calculate_distance(right_wrist, last_wrist_pos['right'])
                            left_velocity = left_dist / delta_time
                            right_velocity = right_dist / delta_time
                            if left_velocity > 2000 or right_velocity > 2000:
                                current_frame_error = "Lỗi: Vung tay quá nhanh"
                        last_wrist_pos['left'], last_wrist_pos['right'] = left_wrist, right_wrist
                    last_frame_time = current_time
                except: pass
            
            # Cập nhật frame kết thúc (nếu rep vẫn đang diễn ra)
            if rep_label == "In Progress":
                if rep_count in rep_time_boundaries:
                    rep_time_boundaries[rep_count]["end_frame"] = frame_id

        # --- Ghi dữ liệu frame này ---
        frame_data = {
            'frame_id': frame_id,
            'view': view,
            'state': state,
            'rep_id': rep_count if "In Progress" in rep_label or state == "up" else 0, # Gán rep_id
            'rule_error': current_frame_error, # Lỗi rule-based của frame này
        }
        # Thêm 132 cột landmarks
        for i, col_name in enumerate(landmarks_header):
            frame_data[col_name] = lm_data_row[i]
            
        all_frames_data.append(frame_data)

    cap.release()
    print(f"[Analyze] Đã phân tích {frame_id} frames.")
    
    # Chuyển đổi sang DataFrame
    if not all_frames_data:
        print("[Analyze] Không có dữ liệu frame nào được thu thập.")
        return pd.DataFrame(), fps, {}
        
    df = pd.DataFrame(all_frames_data)
    
    # Chuyển đổi mốc thời gian (frame -> giây)
    rep_time_boundaries_sec = {}
    for rep_id, times in rep_time_boundaries.items():
        rep_time_boundaries_sec[rep_id] = {
            "start_time": times['start_frame'] / fps,
            "end_time": times['end_frame'] / fps
        }
    
    print(f"[Analyze] Hoàn thành. Tìm thấy {rep_count} reps.")
    return df, fps, rep_time_boundaries_sec

# === LƯỢT 3: HÀM TẠO VIDEO ===
def create_annotated_video(input_path, output_path, df_summary, rep_results_ml, fps, rep_time_boundaries_original):
    """
    Đọc lại video, vẽ chú thích (bao gồm lỗi động và slow-motion)
    và tính toán lại mốc thời gian.
    """
    print(f"[CreateVideo] Bắt đầu tạo video: {output_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[CreateVideo] Lỗi: Không thể mở video gốc '{input_path}'")
        return {}

    # Sử dụng codec 'avc1' (H.264) thay vì 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, OUTPUT_DIM)

    if not out.isOpened():
        print("[CreateVideo] CẢNH BÁO: Không thể mở VideoWriter với codec 'avc1'.")
        print("[CreateVideo] Thử lại với codec 'mp4v' (có thể không phát được trên web).")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, OUTPUT_DIM)
        if not out.isOpened():
            print("[CreateVideo] LỖI NGHIÊM TRỌNG: Không thể khởi tạo VideoWriter.")
            cap.release()
            return {}

    frame_id = 0
    
    # --- Biến theo dõi mốc thời gian ---
    current_output_frame_count = 0 # Số frame *thực tế* đã ghi ra
    new_rep_boundaries = {} # {rep_id: {"start_time": float}}
    
    # Chuyển đổi mốc thời gian gốc (giây -> frame) để dễ so sánh
    rep_start_frames_original = {}
    for rep_id, times in rep_time_boundaries_original.items():
        rep_start_frames_original[rep_id] = int(times['start_time'] * fps)
    
    print("[CreateVideo] Bắt đầu vòng lặp ghi video...")
    
    # Chuyển DataFrame sang dict để truy cập nhanh
    df_dict = df_summary.set_index('frame_id').to_dict('index')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        frame_id += 1

        # --- SỬA LỖI MÉO HÌNH (LƯỢT 3) ---
        try:
            # image = cv2.resize(image, OUTPUT_DIM, interpolation=cv2.INTER_AREA) # <-- CODE CŨ
            image, _, _ = resize_and_pad(image, OUTPUT_DIM) # <-- CODE MỚI
        except Exception as e:
            print(f"Lỗi resize frame {frame_id}: {e}")
            continue
        # --- KẾT THÚC SỬA LỖI ---

        # Lấy dữ liệu của frame này
        frame_data = df_dict.get(frame_id, {})
        
        rep_id = frame_data.get('rep_id', 0)
        state = frame_data.get('state', 'down')
        view = frame_data.get('view', 'Unknown')
        
        # --- CẬP NHẬT: Lỗi Động ---
        # Lấy lỗi rule-based của frame này
        reason_display = frame_data.get('rule_error', '')
        
        # Lấy nhãn ML tổng thể của rep này
        ml_label = ""
        if rep_id > 0:
            ml_label = rep_results_ml.get(rep_id, {}).get('label', '')
            if ml_label == "Correct":
                ml_label = "Đúng"
            elif ml_label == "Incorrect":
                ml_label = "Sai"

        # === CẬP NHẬT: VẼ LANDMARKS VÀ CONNECTIONS ===
        # 1. Tạo một dictionary tạm thời để lưu tọa độ pixel
        lm_pixels = {}
        for i in range(33):
            try:
                x = frame_data.get(f'lm_{i}_x', 0)
                y = frame_data.get(f'lm_{i}_y', 0)
                vis = frame_data.get(f'lm_{i}_vis', 0)
                
                if vis > 0.5:
                    # Tính toán tọa độ pixel dựa trên ảnh 1280x720 (đã pad)
                    lm_pixels[i] = {
                        'px': int(x * OUTPUT_DIM[0]),
                        'py': int(y * OUTPUT_DIM[1]),
                        'vis': vis
                    }
            except Exception:
                pass

        # 2. Vẽ các đường nối (Connections)
        if mp_pose.POSE_CONNECTIONS:
            for connection in mp_pose.POSE_CONNECTIONS:
                try:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    # Chỉ vẽ nếu cả hai điểm đều tồn tại và rõ
                    if start_idx in lm_pixels and end_idx in lm_pixels:
                        start_point = (lm_pixels[start_idx]['px'], lm_pixels[start_idx]['py'])
                        end_point = (lm_pixels[end_idx]['px'], lm_pixels[end_idx]['py'])
                        
                        # (B, G, R) - Màu trắng
                        cv2.line(image, start_point, end_point, (255, 255, 255), 2)
                except Exception:
                    pass
        
        # 3. Vẽ các điểm (Landmarks) - Vẽ sau để đè lên đường
        for i, data in lm_pixels.items():
            try:
                # (B, G, R)
                cv2.circle(image, (data['px'], data['py']), 5, (230, 66, 245), -1) # Màu tím
                cv2.circle(image, (data['px'], data['py']), 6, (255, 255, 255), 1) # Viền trắng
            except Exception:
                pass
        # ================================

        # --- Chuyển sang PIL để vẽ ---
        # (Giờ ảnh đã có landmark và connections)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # --- Vẽ giao diện (UI) ---
        
        # === CẬP NHẬT: Vẽ hình chữ nhật ở góc ===
        # Chuyển ảnh sang RGBA để hỗ trợ độ trong suốt
        image_pil = image_pil.convert('RGBA')
        
        # Tạo một lớp phủ mới, trong suốt
        overlay_pil = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay_pil)
        
        # Vẽ hình chữ nhật đen mờ ở góc
        rect_pos = [0, 0, 450, 230] # Góc trên bên trái
        draw_overlay.rectangle(rect_pos, fill=(0, 0, 0, 150)) # 150/255 độ mờ
        
        # Kết hợp ảnh gốc với lớp phủ
        image_pil = Image.alpha_composite(image_pil, overlay_pil)
        # ========================================
        
        # Vẽ thông tin (lên ảnh đã có nền mờ)
        draw_text_unicode(image_pil, f"FRAME: {frame_id}", (20, 20), font_md)
        draw_text_unicode(image_pil, f"VIEW: {view}", (20, 60), font_md)
        draw_text_unicode(image_pil, f"STATE: {state.upper()}", (20, 100), font_md)
        draw_text_unicode(image_pil, f"REP: {rep_id}", (20, 140), font_md)
        
        # Vẽ nhãn ML (Đúng/Sai)
        if ml_label:
            color = (0, 255, 0) if ml_label == "Đúng" else (255, 0, 0)
            draw_text_unicode(image_pil, f"ĐÁNH GIÁ: {ml_label}", (20, 180), font_lg, fill=color)

        # Vẽ Lỗi Động (nếu có)
        if reason_display:
            # Vẽ ở giữa màn hình
            try:
                # Bọc trong try-except để đề phòng lỗi font
                text_bbox = font_reason.getbbox(reason_display)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                pos_x = (OUTPUT_DIM[0] - text_w) // 2
                pos_y = (OUTPUT_DIM[1] - text_h) // 2
                
                # --- Thêm nền đen cho chữ đỏ ---
                draw = ImageDraw.Draw(image_pil, 'RGBA')
                rect_pos = [pos_x - 10, pos_y - 10, pos_x + text_w + 10, pos_y + text_h + 10]
                draw.rectangle(rect_pos, fill=(0, 0, 0, 150))
                # --- ---
                
                draw_text_unicode(image_pil, reason_display, (pos_x, pos_y), font_reason, fill=(255, 0, 0))
            except Exception as e:
                print(f"Lỗi khi vẽ text: {e}")

        # --- Chuyển về CV2 ---
        image_annotated = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # --- Ghi frame ---
        out.write(image_annotated)
        current_output_frame_count += 1
        
        # --- CẬP NHẬT: Tính toán mốc thời gian ---
        # Kiểm tra xem frame GỐC này có phải là điểm bắt đầu của rep nào không
        for rep_id_orig, start_frame_orig in rep_start_frames_original.items():
            if frame_id == start_frame_orig:
                # Ghi lại mốc thời gian MỚI (tính bằng giây)
                new_start_time_sec = (current_output_frame_count - 1) / fps
                new_rep_boundaries[rep_id_orig] = {"start_time": new_start_time_sec}
                print(f"[CreateVideo] Rep {rep_id_orig}: Mốc thời gian gốc {start_frame_orig} (frame) -> Mốc thời gian mới {new_start_time_sec:.2f} (giây)")

        # --- CẬP NHẬT: Slow-motion ---
        if reason_display: # Nếu frame này có lỗi
            slow_mo_frames = int(fps * 0.5) # Dừng 0.5 giây
            for _ in range(slow_mo_frames):
                out.write(image_annotated) # Ghi lại y hệt frame đó
                current_output_frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[CreateVideo] Đã tạo video thành công tại: {output_path}")
    
    # Trả về các mốc thời gian đã được tính toán lại
    return new_rep_boundaries