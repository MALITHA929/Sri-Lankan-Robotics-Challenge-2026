import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false" 

import cv2
import ctypes
import numpy as np
import serial
import time
import threading
import requests
import math
from queue import Queue
from picamera2 import Picamera2
from gpiozero import Button, LED 

# -----------------------------------------
# 1. HARDWARE & LED SETUP
# -----------------------------------------
reset_switch = Button(27, pull_up=True)
red_led = LED(17)
blue_led = LED(22)
green_led = LED(23)

WIDTH, HEIGHT = 640, 360

# --- VISION COLOR RANGES ---
# Red ranges for the boxes
color_ranges = [
    (np.array([0, 70, 50], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
    (np.array([160, 70, 50], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
]
# Blue ranges for the circles (Task 3)
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([150, 255, 255])

# -----------------------------------------
# 2. API & APRILTAG SETUP
# -----------------------------------------
API_BASE = "http://10.23.73.143:8000"
KEYS = { 0: 6180, 1: 3141, 2: 2718, 3: 8080, 4: 4040 }

lib = ctypes.CDLL("/usr/local/lib/libapriltag.so")

class apriltag_detection_t(ctypes.Structure):
    _fields_ = [("family", ctypes.c_void_p), ("id", ctypes.c_int), ("hamming", ctypes.c_int),
                ("decision_margin", ctypes.c_float), ("H", ctypes.c_void_p),
                ("c", ctypes.c_double * 2), ("p", ctypes.c_double * 8)]

class zarray_t(ctypes.Structure):
    _fields_ = [("el_sz", ctypes.c_size_t), ("size", ctypes.c_int), ("alloc", ctypes.c_int), ("data", ctypes.c_void_p)]

class apriltag_image_u8_t(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("stride", ctypes.c_int), ("buf", ctypes.c_void_p)]

lib.tagStandard52h13_create.restype = ctypes.c_void_p
lib.apriltag_detector_create.restype = ctypes.c_void_p
lib.apriltag_detector_add_family_bits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.apriltag_detector_detect.restype = ctypes.POINTER(zarray_t)
lib.apriltag_detections_destroy.argtypes = [ctypes.POINTER(zarray_t)]

family = lib.tagStandard52h13_create()
detector = lib.apriltag_detector_create()
lib.apriltag_detector_add_family_bits(detector, family, 2)

def decode_tag(tag_id):
    tag_str = str(tag_id).zfill(5)
    key_id = int(tag_str[0])
    payload = int(tag_str[1:])

    if key_id not in KEYS: return None, None, None
    K = KEYS[key_id]
    if key_id == 0: A = ((int(tag_str[1:][::-1]) * 7) + K) % 10000
    elif key_id == 1: A = ((int(tag_str[3:5] + tag_str[1:3]) * 3) + K) % 8750
    elif key_id == 2: A = (((9999 - payload) * 9) + K) % 8750
    elif key_id == 3: A = ((int(tag_str[4] + tag_str[2:4] + tag_str[1]) * 11) + K) % 8750
    elif key_id == 4: A = (payload ^ (payload // 2)) ^ K

    order = (A // 625) + 1
    remainder = A % 625
    grid_x = remainder // 25
    grid_y = remainder % 25
    return order, (grid_x - 12) * 0.4, (12 - grid_y) * 0.4

# -----------------------------------------
# 3. SERIAL COMMUNICATION & THREADING
# -----------------------------------------
arduino_connected = False
serial_queue = Queue()
tag_lock = threading.Lock()  # Protects our dictionary from Race Conditions

def find_arduino():
    for port in ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1']:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"[SERIAL] Connected to Arduino on {port}")
            time.sleep(2) 
            return ser
        except: continue
    return None

ser = find_arduino()
if ser: arduino_connected = True
else: print("[WARNING] Arduino not detected. Visual tracking only.")

def serial_sender():
    while True:
        if arduino_connected and not serial_queue.empty():
            data = serial_queue.get()
            try: ser.write(data.encode())
            except: break
        time.sleep(0.01)

if arduino_connected:
    threading.Thread(target=serial_sender, daemon=True).start()

# -----------------------------------------
# 4. SIMULATION API FUNCTIONS
# -----------------------------------------
def get_pose():
    try:
        r = requests.get(f"{API_BASE}/odometry", timeout=2).json()
        return r["pose"]["x"], r["pose"]["y"], r["pose"]["yaw"]
    except: return 0, 0, 0

def rotate(theta):
    try: requests.post(f"{API_BASE}/move_relative", json={"distance": 0.0, "rotation": float(theta)}, timeout=2)
    except: pass

def move(distance):
    try: requests.post(f"{API_BASE}/move_relative", json={"distance": float(distance), "rotation": 0.0}, timeout=2)
    except: pass

def get_camera_frame(cam_id):
    try:
        r = requests.get(f"{API_BASE}/camera/{cam_id}/frame", timeout=2)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: pass
    return None

def check_hostile():
    img_l = get_camera_frame("front_left")
    img_r = get_camera_frame("front_right")
    lower_green, upper_green = np.array([40, 50, 50]), np.array([80, 255, 255])
    
    for img in [img_l, img_r]:
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            if cv2.countNonZero(mask) > 1500: return True
    return False

def go_to_goal(x_goal, y_goal):
    while True:
        if check_hostile():
            print("[ARES] GREEN HOSTILE DETECTED! Yielding...")
            try: requests.post(f"{API_BASE}/stop", json={}, timeout=2)
            except: pass
            time.sleep(1)
            continue 

        x, y, yaw = get_pose()
        dx, dy = x_goal - x, y_goal - y
        distance = math.hypot(dx, dy)
        if distance < 0.2: break 

        target_angle = math.atan2(dy, dx)
        rotation_needed = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        if abs(rotation_needed) > 0.1:
            rotate(rotation_needed)
            time.sleep(0.5)

        step = min(distance, 0.4)
        move(step)
        time.sleep(0.5)

# -----------------------------------------
# 5. CORE LOGIC HELPERS
# -----------------------------------------
def drive_simulation(target_order, tag_id, tx, ty, current_mode, pause_physical=True):
    print(f"\n[SYSTEM] Driving Ares to Order {target_order} (Tag: {tag_id})...")
    
    if pause_physical:
        print("[SYSTEM] Pausing Physical Robot.")
        if arduino_connected: 
            serial_queue.put("STOP\n")
            serial_queue.put("STOP\n") 
        blue_led.on(); green_led.on(); red_led.off()
    
    go_to_goal(tx, ty)
    print(f"[ARES] Arrived at Order {target_order}!")
    
    if pause_physical:
        print("[SYSTEM] Resuming Physical Robot...")
        blue_led.off(); green_led.off(); red_led.off()
        if current_mode == 0: blue_led.on()
        elif current_mode == 1: blue_led.blink(on_time=0.5, off_time=0.5)
        elif current_mode in [2, 3, 4, 5, 6]: green_led.blink(on_time=0.5, off_time=0.5)

def flush_tags(start_order, end_order, current_mode, pause_physical=True):
    global scanned_tags
    if pause_physical:
        print(f"\n[SYSTEM] SYNC FLUSH! Pausing robot to execute tags {start_order} to {end_order}...")
    else:
        print(f"\n[SYSTEM] ASYNC FLUSH THREAD INITIATED! Executing tags {start_order} to {end_order} in the background...")
        
    for o in range(start_order, end_order + 1):
        has_tag = False
        
        # Thread-safe read & delete
        with tag_lock:
            if o in scanned_tags:
                tag_id, tx, ty = scanned_tags[o]
                del scanned_tags[o]
                has_tag = True
                
        # Drive OUTSIDE the lock to prevent freezing the camera thread!
        if has_tag:
            drive_simulation(o, tag_id, tx, ty, current_mode, pause_physical)
            
    print(f"[SYSTEM] Background Flush ({start_order}-{end_order}) Complete.")

def process_apriltag(gray_img, required_start, required_end):
    global scanned_tags
    img_c = apriltag_image_u8_t(gray_img.shape[1], gray_img.shape[0], gray_img.shape[1], gray_img.ctypes.data_as(ctypes.c_void_p))
    detections = lib.apriltag_detector_detect(detector, ctypes.byref(img_c))
    
    if detections.contents.size > 0:
        ptr_array = ctypes.cast(detections.contents.data, ctypes.POINTER(ctypes.c_void_p))
        for i in range(detections.contents.size):
            det = ctypes.cast(ptr_array[i], ctypes.POINTER(apriltag_detection_t)).contents
            tag_id = det.id
            order, tx, ty = decode_tag(tag_id)
            
            if order is not None and required_start <= order <= required_end:
                
                # Thread-safe write. If we see ANY valid tag, store it for the background flush!
                with tag_lock:
                    if order not in scanned_tags:
                        print(f"[VISION] Tag Acquired. ID: {tag_id} | Order: {order} | X:{tx} Y:{ty}")
                        scanned_tags[order] = (tag_id, tx, ty)
                        
                        if arduino_connected:
                            serial_queue.put("TAG_OK\n")
                        
                        threading.Thread(target=flash_green_success, daemon=True).start()
                    
    lib.apriltag_detections_destroy(detections)

def flash_green_success():
    green_led.on()
    time.sleep(1.5)
    green_led.off()

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        return cap
        
    print(f"\n[WARNING] Camera {index} locked. Waiting...")
    blue_led.off(); green_led.off()
    red_led.blink(on_time=0.1, off_time=0.1) 
    while not cap.isOpened():
        time.sleep(1.5) 
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    red_led.off() 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap

# -----------------------------------------
# 6. MASTER SEQUENTIAL LOOP
# -----------------------------------------
if __name__ == "__main__":
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (640,480)}))
    
    scanned_tags = {}            
    value = 0 
    prev_value = 0
    current_usb_index = -1
    cap = None 
    
    last_send_time = 0
    send_interval = 0.05

    print("[SYSTEM] Booted. Starting in Task 1 (Mode 0: PiCam Tag Scanning)...")
    picam2.start()
    blue_led.on() 

    try:
        while True:
            # ==========================================
            # A. HARDWARE RESET OVERRIDE
            # ==========================================
            if reset_switch.is_pressed:
                print("\n[RESET] Switch pressed! Resetting to Task 1 (Mode 0)...")
                value = 0
                prev_value = 0
                current_usb_index = -1
                with tag_lock:
                    scanned_tags.clear()
                
                try: picam2.stop()
                except: pass
                if cap: cap.release(); cap = None
                
                picam2.start()
                blue_led.on(); green_led.off(); red_led.off()
                cv2.destroyAllWindows() 
                reset_switch.wait_for_release()
                print("[SYSTEM] Resuming...")
                continue 

            # ==========================================
            # B. ARDUINO TASK/MODE SWITCHING
            # ==========================================
            if ser and ser.in_waiting > 0:
                mode_changed = False
                while ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if line:
                            new_value = int(line)
                            if new_value != value:
                                value = new_value
                                mode_changed = True
                    except ValueError: pass 
                
                if mode_changed:
                    print(f"[SYSTEM] Mode Switched to: {value}")
                    
                    # --- TASK TRANSITION ASYNC FLUSHING ---
                    # When Arduino explicitly tells us to flush tags 1-8
                    if value == 5:
                        threading.Thread(target=flush_tags, args=(1, 8, value, False), daemon=True).start()
                        
                    # When Arduino explicitly tells us to flush tags 9-14
                    elif value == 6:
                        threading.Thread(target=flush_tags, args=(9, 14, value, False), daemon=True).start()
                    
                    prev_value = value
                    
                    # --- CAMERA HARDWARE SWITCHING ---
                    blue_led.off(); green_led.off(); red_led.off()
                    cv2.destroyAllWindows() 
                    
                    # Pi Camera for Mode 0
                    if value == 0:
                        if cap: cap.release(); cap = None
                        current_usb_index = -1
                        try: picam2.start()
                        except: pass
                        blue_led.on() 
                    
                    # USB Cameras for Modes 1 through 6
                    elif value in [1, 2, 3, 4, 5, 6]:
                        try: picam2.stop() 
                        except: pass
                        
                        # Mode 1 and 5 use Camera 0. Modes 2, 3, 4, and 6 use Camera 2.
                        target_index = 0 if value in [1, 5] else 2 
                        
                        if cap is None or current_usb_index != target_index:
                            if cap: cap.release()
                            time.sleep(1)
                            cap = open_camera(target_index)
                            current_usb_index = target_index
                        
                        if value == 1: blue_led.blink(on_time=0.5, off_time=0.5)
                        elif value in [2, 3, 4, 5, 6]: green_led.blink(on_time=0.5, off_time=0.5)

            # ==========================================
            # C. MODE 0: TASK 1 APRILTAGS (PiCam 1-8)
            # ==========================================
            if value == 0:
                frame = picam2.capture_array()
                gray = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
                # Just scans and stores silently. Does not pause robot!
                process_apriltag(gray, required_start=1, required_end=8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Ares Vision - Mode 0 (Task 1 Tags)", frame_bgr)

            # ==========================================
            # D. MODE 1 & 2: TASK 2 BOX TRACK/FOLLOW
            # ==========================================
            elif value in [1, 2] and cap is not None:
                ret, frame = cap.read()
                if not ret: continue

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                for lower, upper in color_ranges:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                now = time.time()

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 800:
                        red_led.on() 
                        x, y, w, h_box = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h_box), (0, 255, 0), 2)
                        cv2.circle(frame, (x + w//2, y + h_box//2), 5, (255, 0, 0), -1)

                        if arduino_connected and (now - last_send_time > send_interval):
                            serial_queue.put(f"C1:{x},{y},{w},{h_box}\n")
                            last_send_time = now
                    else:
                        red_led.off()
                        if arduino_connected and (now - last_send_time > send_interval):
                            serial_queue.put("STOP\n")
                            last_send_time = now
                else:
                    red_led.off()
                    if arduino_connected and (now - last_send_time > send_interval):
                        serial_queue.put("STOP\n")
                        last_send_time = now
                        
                cv2.imshow(f"Ares Vision - Mode {value} (Box)", frame)

            # ==========================================
            # E. MODE 3: TASK 2 APRILTAGS (USBCam 9-14)
            # ==========================================
            elif value == 3 and cap is not None:
                ret, frame = cap.read()
                if not ret: continue
                
                gray = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                process_apriltag(gray, required_start=9, required_end=14)
                cv2.imshow("Ares Vision - Mode 3 (Task 2 Tags)", frame)

            # ==========================================
            # F. MODE 4: TASK 3 BLUE CIRCLES
            # ==========================================
            elif value == 4 and cap is not None:
                ret, frame = cap.read()
                if not ret: continue

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                top = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                now = time.time()

                if len(top) == 2:
                    m1, m2 = cv2.moments(top[0]), cv2.moments(top[1])
                    if m1["m00"] > 500 and m2["m00"] > 500:
                        red_led.on()
                        x1, y1 = int(m1["m10"]/m1["m00"]), int(m1["m01"]/m1["m00"])
                        x2, y2 = int(m2["m10"]/m2["m00"]), int(m2["m01"]/m2["m00"])
                        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                        dist = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))

                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.circle(frame, (mid_x, mid_y), 7, (0, 255, 0), -1)
                        cv2.putText(frame, f"Dist: {dist}", (20, 30), 1, 1, (0, 255, 0), 2)

                        if arduino_connected and (now - last_send_time > send_interval):
                            serial_queue.put(f"{mid_x},{mid_y},{dist}\n")
                            last_send_time = now
                    else:
                        red_led.off()
                        if arduino_connected and (now - last_send_time > send_interval):
                            serial_queue.put("STOP\n")
                            last_send_time = now
                else:
                    red_led.off()
                    if arduino_connected and (now - last_send_time > send_interval):
                        serial_queue.put("STOP\n")
                        last_send_time = now

                cv2.imshow("Ares Vision - Mode 4 (Blue Circles)", frame)
                
            # ==========================================
            # G. MODE 5 & 6: FLUSHING IDLE STATES
            # ==========================================
            elif value in [5, 6] and cap is not None:
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, f"ASYNC FLUSH MODE {value} ACTIVE...", (20, 30), 1, 1, (0, 0, 255), 2)
                    cv2.imshow(f"Ares Vision - Mode {value} (Idle)", frame)

            # --- Safety Quit Command ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q' pressed. Initiating shutdown...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Program stopped via Terminal (Ctrl+C).")

    finally:
        print("[SYSTEM] Cleaning up hardware and shutting down...")
        red_led.off(); blue_led.off(); green_led.off()
        try: picam2.stop()
        except: pass
        if cap: cap.release()
        cv2.destroyAllWindows() 
        if arduino_connected: ser.close()
        print("[SYSTEM] Shutdown complete.")