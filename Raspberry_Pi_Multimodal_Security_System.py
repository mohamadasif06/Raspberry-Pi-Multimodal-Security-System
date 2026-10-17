import os
import sys
import time
import threading
import subprocess
import pickle
from collections import deque
import signal
import RPi.GPIO as GPIO
import numpy as np
import cv2
import face_recognition
# ---- External deps mirrored from your originals ----
import serial
import requests
import tty
import termios
import speech_recognition as sr
# ---- Google Drive (OAuth) from your face script ----
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
# ============================================================
# EXIT HANDLER
# ============================================================
def handle_exit(signum, frame):
    global stop_threads
    print("\n[EXIT] Shutdown requested. Stopping all processes safely.")
    stop_threads = True
    GPIO.output(RELAY_PIN, GPIO.LOW)
    GPIO.cleanup()
    cv2.destroyAllWindows()
    os._exit(0)
# Handle both Ctrl+C and Ctrl+Z
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTSTP, handle_exit)
# ============================================================
# GPIO / RELAY SETUP
# ============================================================
RELAY_PIN = 10  # BCM GPIO10; verify your physical wiring before use.
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # Relay initially OFF
# ============================================================
# CONFIGURATION
# ============================================================
# ---- Face recognition configuration ----
DRIVE_FOLDER_ID = "1CTUqoTcpXJ2cEzbdnbNxADS7Q2tV3-3m"
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FRAME_SIZE = int(FRAME_WIDTH * FRAME_HEIGHT * 1.5)
TOLERANCE = 0.45
PROCESS_EVERY = 3
FONT = cv2.FONT_HERSHEY_DUPLEX
# ---- Fingerprint / GSM / GPS configuration ----
PRIMARY_PHONE = "+919380611160"
TERMINAL_PASSWORD = "1234"
EMERGENCY_KEY = "8"
GPS_PORT = "/dev/ttyAMA3"
GPS_BAUD = 9600
GSM_PORT = "/dev/serial0"
GSM_BAUD = 9600
FINGER_UART = "/dev/ttyAMA4"
FINGER_BAUD = 57600
FINGERPLACE_TIMEOUT = 10
VOICE_PROMPT_DELAY = 0.6
VOICE_TIMEOUT = 10
GPS_WARMUP = 1.2
GPS_READ_TIMEOUT = 12
REVERSE_GEOCODE_DELAY = 1.2
SMS_COOLDOWN = 2.5
USER_AGENT = "RPi-TheftSystem/1.0 (youremail@example.com)"
# ============================================================
# GLOBAL STATE
# ============================================================
known_encodings = []
known_names = []
enc_lock = threading.Lock()
encodings_loaded = threading.Event()
frame_queue = deque(maxlen=1)
stop_threads = False
# Emergency/keypad gating
keypad_active = False
_emergency_result = None  # None / "matched" / "failed"
# Last uploaded face ID (only when face is unmatched)
last_drive_file_id = None
# External handles initialized later
_drive_service = None
_fps_time = time.time()
_frame_counter = 0
# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
def build_drive_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)
def upload_to_drive_sync(service, file_path, folder_id=None):
    """Synchronously upload a file so its Drive link can be sent by SMS."""
    fname = os.path.basename(file_path)
    metadata = {"name": fname}
    if folder_id:
        metadata["parents"] = [folder_id]
    media = MediaFileUpload(file_path, resumable=True)
    uploaded = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id",
        )
        .execute()
    )
    return uploaded.get("id")
# ============================================================
# FACE ENCODINGS / CAMERA
# ============================================================
def build_encodings_from_dataset():
    encs, names = [], []
    if not os.path.isdir(DATASET_DIR):
        print(f"[WARN] Dataset directory '{DATASET_DIR}' not found.")
        return encs, names
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(image)
                if enc:
                    encs.append(enc[0])
                    names.append(person_name)
            except Exception as e:
                print(f"[WARN] Failed to load {img_path}: {e}")
    return encs, names
def encodings_loader():
    global known_encodings, known_names
    try:
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                with enc_lock:
                    known_encodings, known_names = data
                print(
                    f"[INFO] Loaded {len(known_names)} "
                    "cached encodings."
                )
                return
            print(
                "[WARN] Invalid pickle format; "
                "rebuilding from dataset."
            )
        encs, names = build_encodings_from_dataset()
        with enc_lock:
            known_encodings, known_names = encs, names
        try:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(
                    (known_encodings, known_names),
                    f,
                )
            print(
                f"[INFO] Encodings built and saved "
                f"({len(known_names)} faces)."
            )
        except Exception as e:
            print(f"[WARN] Could not save encodings: {e}")
    except Exception as e:
        print(f"[ERROR] Exception in encodings_loader: {e}")
    finally:
        encodings_loaded.set()
def start_rpicam_process():
    cmd = [
        "rpicam-vid",
        "-t",
        "0",
        "--width",
        str(FRAME_WIDTH),
        "--height",
        str(FRAME_HEIGHT),
        "--codec",
        "yuv420",
        "--inline",
        "--nopreview",
        "-o",
        "-",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=10**8,
    )
def rpicam_reader(proc):
    global stop_threads
    while not stop_threads:
        raw = proc.stdout.read(FRAME_SIZE)
        if len(raw) != FRAME_SIZE:
            continue
        yuv = np.frombuffer(
            raw,
            np.uint8,
        ).reshape(
            (int(FRAME_HEIGHT * 1.5), FRAME_WIDTH)
        )
        frame = cv2.cvtColor(
            yuv,
            cv2.COLOR_YUV2BGR_I420,
        )
        frame_queue.append(frame)
def crop_face_from_frame(
    frame,
    top,
    right,
    bottom,
    left,
    scale=1.6,
):
    h, w = frame.shape[:2]
    face_w = right - left
    face_h = bottom - top
    cx = left + face_w // 2
    cy = top + face_h // 2
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)
    left = max(0, cx - new_w // 2)
    right = min(w, cx + new_w // 2)
    top = max(0, cy - new_h // 2)
    bottom = min(h, cy + new_h // 2)
    return frame[top:bottom, left:right]
# ============================================================
# GSM / GPS / VOICE / KEYPAD
# ============================================================
def open_gsm():
    try:
        return serial.Serial(
            GSM_PORT,
            GSM_BAUD,
            timeout=2,
        )
    except Exception as e:
        print("[ERR] Cannot open GSM port:", e)
        return None
def open_gps():
    try:
        return serial.Serial(
            GPS_PORT,
            GPS_BAUD,
            timeout=1,
        )
    except Exception as e:
        print("[WARN] open_gps failed:", e)
        return None
def read_gps_once(
    gps_serial=None,
    timeout=GPS_READ_TIMEOUT,
):
    """Return (latitude, longitude) from one valid GGA sentence."""
    own_serial = False
    if gps_serial is None:
        gps_serial = open_gps()
        own_serial = True
    if gps_serial is None:
        return None, None
    start = time.time()
    while time.time() - start < timeout:
        try:
            line = (
                gps_serial.readline()
                .decode(errors="ignore")
                .strip()
            )
        except Exception:
            line = ""
        if not line:
            time.sleep(0.08)
            continue
        if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
            parts = line.split(",")
            try:
                if parts[2] and parts[4]:
                    lat_raw = float(parts[2])
                    lon_raw = float(parts[4])
                    lat_deg = int(lat_raw / 100)
                    lon_deg = int(lon_raw / 100)
                    lat_min = lat_raw - lat_deg * 100
                    lon_min = lon_raw - lon_deg * 100
                    lat = lat_deg + lat_min / 60
                    lon = lon_deg + lon_min / 60
                    if parts[3] == "S":
                        lat = -lat
                    if parts[5] == "W":
                        lon = -lon
                    if own_serial:
                        gps_serial.close()
                    return lat, lon
            except Exception:
                pass
        time.sleep(0.08)
    if own_serial:
        gps_serial.close()
    return None, None
def reverse_geocode(lat, lon):
    if lat is None or lon is None:
        return "No GPS fix"
    time.sleep(REVERSE_GEOCODE_DELAY)
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "format": "json",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=8,
        )
        if response.status_code != 200:
            return "Unknown Location"
        data = response.json()
        return data.get(
            "display_name",
            "Unknown Location",
        )
    except Exception as e:
        print("[WARN] reverse_geocode failed:", e)
        return "Unknown Location"
def send_sms(message, number=PRIMARY_PHONE):
    """
    Send an SMS using SIM900A/SIM800 text mode.
    Long messages are split into chunks of approximately
    150 characters, preferably at line breaks.
    """
    try:
        gsm = serial.Serial(
            GSM_PORT,
            GSM_BAUD,
            timeout=1,
        )
    except Exception as e:
        print("[ERR] Cannot open GSM port:", e)
        return False
    def send_at(cmd, delay=1):
        gsm.write((cmd + "\r").encode())
        time.sleep(delay)
        reply = gsm.read(
            gsm.in_waiting or 256
        ).decode(errors="ignore")
        print(
            f">> {cmd}\n"
            f"{reply.strip()}\n"
        )
        return reply
    try:
        # ---- GSM initialization ----
        send_at("AT")
        send_at("ATE0")
        send_at("AT+CMEE=2")
        send_at("AT+CFUN=1", 1.5)
        send_at("AT+CMGF=1")
        send_at('AT+CSCS="GSM"')
        send_at("AT+CSMP=17,167,0,0")
        time.sleep(1)
        # ---- Smart chunking ----
        message = message.strip()
        lines = message.split("\n")
        chunks = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) + 1 <= 150:
                current_chunk += line + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        # ---- Send each SMS chunk ----
        for idx, chunk in enumerate(chunks, 1):
            print(
                f"[GSM] Sending part "
                f"{idx}/{len(chunks)} "
                f"({len(chunk)} chars)"
            )
            send_at(
                f'AT+CMGS="{number}"',
                2,
            )
            gsm.write(
                (chunk + "\x1A").encode()
            )
            time.sleep(4)
        print("[OK] All message parts sent successfully.")
        return True
    except Exception as e:
        print("[ERR] GSM send_sms exception:", e)
        return False
    finally:
        try:
            gsm.close()
        except Exception:
            pass
def gps_place_map():
    gps = open_gps()
    time.sleep(GPS_WARMUP)
    lat, lon = read_gps_once(
        gps_serial=gps
    )
    if gps:
        gps.close()
    place = reverse_geocode(lat, lon)
    lat_s = (
        f"{lat:.6f}"
        if lat is not None
        else "None"
    )
    lon_s = (
        f"{lon:.6f}"
        if lon is not None
        else "None"
    )
    link = (
        f"https://www.google.com/maps"
        f"?q={lat_s},{lon_s}"
    )
    return lat_s, lon_s, place, link
def voice_command_with_prompt(
    countdown=3,
    timeout=VOICE_TIMEOUT,
):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("[VOICE] Prepare to speak in:")
    for i in range(countdown, 0, -1):
        print(
            i,
            end=" ",
            flush=True,
        )
        time.sleep(1)
    print("\n[VOICE] Listening now...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.5,
            )
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=4,
            )
        text = recognizer.recognize_google(audio)
        print("[VOICE] Heard:", text)
        return "start" in text.lower()
    except Exception as e:
        print("[VOICE] Error:", e)
    return False
def masked_input(prompt="Password: "):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        buf = ""
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                continue
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                return buf
            if ch in ("\x7f", "\b"):
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch == "\x03":
                raise KeyboardInterrupt
            buf += ch
            sys.stdout.write("*")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            old_settings,
        )
def terminal_thread():
    """
    Always-on keypad.
    Press '8' then Enter for emergency password mode.
    Type 'quit' or 'exit' to terminate the program.
    """
    global keypad_active, _emergency_result
    print(
        "\n[KEYPAD] Press '8' then Enter for "
        "emergency password mode. "
        "'quit' to exit.\n"
    )
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            print("[KEYPAD] Exit requested. Terminating.")
            os._exit(0)
        if line == EMERGENCY_KEY:
            # Engage emergency mode and block biometrics.
            keypad_active = True
            try:
                password = masked_input(
                    "Enter password (****): "
                )
            except KeyboardInterrupt:
                print(
                    "\n[KEYPAD] Password entry cancelled."
                )
                _emergency_result = None
                keypad_active = False
                continue
            if password == TERMINAL_PASSWORD:
                _emergency_result = "matched"
                print(
                    "[KEYPAD] Password matched "
                    "-> Relay/Motor ON."
                )
            else:
                _emergency_result = "failed"
                print(
                    "[KEYPAD] WRONG password "
                    "-> alert will be sent."
                )
            # Keep keypad_active True until main loop
            # processes the result.
            continue
        print(
            "[KEYPAD] Unrecognized input. "
            "Press '8' then Enter to enter password."
        )
# ============================================================
# FINGERPRINT SENSOR INITIALIZATION
# ============================================================
try:
    from pyfingerprint.pyfingerprint import PyFingerprint
    _finger = PyFingerprint(
        FINGER_UART,
        FINGER_BAUD,
        0xFFFFFFFF,
        0x00000000,
    )
    if not _finger.verifyPassword():
        print("[ERR] Fingerprint verify failed.")
        sys.exit(1)
    print("[OK] Fingerprint sensor ready.")
except Exception as e:
    print("[ERR] Fingerprint init error:", e)
    _finger = None
# ============================================================
# RELAY / MOTOR
# ============================================================
def activate_motor_or_relay():
    """
    Activate the vehicle relay or motor output.
    The relay is turned ON for 2 seconds and then
    returned to LOW.
    """
    print("[RELAY] Activating relay/motor output...")
    GPIO.output(
        RELAY_PIN,
        GPIO.HIGH,
    )
    time.sleep(2)
    GPIO.output(
        RELAY_PIN,
        GPIO.LOW,
    )
    print("[RELAY] Relay/motor deactivated.")
def send_alert_sms(
    prefix,
    include_drive=False,
    drive_file_id=None,
):
    """
    Send an alert SMS containing the detected location.
    For an unknown face, the second SMS also contains
    the Google Drive image link.
    """
    lat_s, lon_s, place, link = gps_place_map()
    # ---- First SMS: alert + place ----
    msg1_lines = [
        prefix,
        f"Place: {place}",
    ]
    sms_text_1 = "\n".join(msg1_lines)
    # ---- Second SMS: map + optional Drive link ----
    msg2_lines = [
        f"Google Map: {link}",
    ]
    if include_drive and drive_file_id:
        msg2_lines.append(
            "Drive: "
            f"https://drive.google.com/file/d/"
            f"{drive_file_id}/view"
        )
    sms_text_2 = "\n".join(msg2_lines)
    # ---- Log ----
    print("-" * 50)
    print("[SMS #1]")
    print(sms_text_1)
    print("[SMS #2]")
    print(sms_text_2)
    print("-" * 50)
    # ---- Send both messages ----
    send_sms(sms_text_1)
    time.sleep(2)
    send_sms(sms_text_2)
    time.sleep(SMS_COOLDOWN)
# ============================================================
# FACE VERIFICATION PHASE
# ============================================================
def perform_face_phase():
    """
    Return True if the face matches.
    If the face is unknown, capture the face, upload it
    to Google Drive, send an SMS alert, and return False.
    The phase aborts if keypad_active becomes True.
    """
    global last_drive_file_id
    # ---- Wait for face encodings ----
    if not encodings_loaded.is_set():
        print("[FACE] Waiting for encodings...")
        while not encodings_loaded.is_set():
            if keypad_active:
                return False
            time.sleep(0.2)
    # ---- Wait for a camera frame ----
    start_wait = time.time()
    frame = None
    while time.time() - start_wait < 3.0:
        if keypad_active:
            return False
        if frame_queue:
            frame = frame_queue[-1]
            if frame is not None and frame.size > 0:
                frame = frame.copy()
                cv2.imshow(
                    "Real-Time Recognition",
                    frame,
                )
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt
                break
        time.sleep(0.02)
    if frame is None or frame.size == 0:
        print("[FACE] No camera frame yet.")
        return False
    # ---- Downscale for face embedding ----
    small = cv2.resize(
        frame,
        (0, 0),
        fx=0.5,
        fy=0.5,
    )
    rgb_small = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB,
    )
    with enc_lock:
        encs_copy = list(known_encodings)
        names_copy = list(known_names)
    face_locs = face_recognition.face_locations(
        rgb_small,
        model="hog",
    )
    face_encs = face_recognition.face_encodings(
        rgb_small,
        face_locs,
    )
    if len(face_encs) == 0:
        cv2.putText(
            frame,
            "No face detected",
            (10, 30),
            FONT,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.imshow(
            "Real-Time Recognition",
            frame,
        )
        cv2.waitKey(1)
        return False
    top, right, bottom, left = face_locs[0]
    face_enc = face_encs[0]
    matched = False
    confidence_display = "Unknown"
    display_name = "Unknown"
    if encs_copy:
        distances = face_recognition.face_distance(
            encs_copy,
            face_enc,
        )
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]
        if best_distance < TOLERANCE:
            matched = True
            name = names_copy[best_idx]
            display_name = name
            # Pseudo-confidence for display only.
            conf = (1 - best_distance) * 100
            conf = max(
                90,
                min(conf, 96),
            )
            confidence_display = f"{conf:.1f}%"
            print(
                f"[FACE] Matched: "
                f"{name} ({confidence_display})"
            )
        else:
            pseudo_conf = max(
                0,
                (1 - best_distance) * 5,
            )
            confidence_display = f"{pseudo_conf:.1f}%"
    else:
        confidence_display = "0%"
    # ---- Display overlay ----
    label_text = (
        f"{display_name} | "
        f"{confidence_display}"
    )
    color = (
        (0, 255, 0)
        if matched
        else (0, 0, 255)
    )
    cv2.rectangle(
        frame,
        (5, 5),
        (355, 45),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label_text,
        (10, 35),
        FONT,
        0.8,
        color,
        2,
    )
    cv2.imshow(
        "Real-Time Recognition",
        frame,
    )
    cv2.waitKey(1)
    # ---- Rescale face coordinates ----
    top *= 2
    right *= 2
    bottom *= 2
    left *= 2
    if matched:
        last_drive_file_id = None
        return True
    # ---- Unknown face: save image ----
    os.makedirs(
        "unknown_faces",
        exist_ok=True,
    )
    crop = crop_face_from_frame(
        frame,
        top,
        right,
        bottom,
        left,
        scale=1.6,
    )
    ts = int(time.time())
    filename = f"unknown_{ts}.jpg"
    path = os.path.join(
        "unknown_faces",
        filename,
    )
    try:
        if crop is None or getattr(
            crop,
            "size",
            0,
        ) == 0:
            cv2.imwrite(path, frame)
        else:
            cv2.imwrite(path, crop)
        print(
            f"[ALERT] Unknown face saved: {path}"
        )
    except Exception as e:
        print(
            f"[WARN] Could not save image: {e}"
        )
    # ---- Upload to Google Drive ----
    drive_id = None
    if _drive_service and DRIVE_FOLDER_ID:
        try:
            drive_id = upload_to_drive_sync(
                _drive_service,
                path,
                DRIVE_FOLDER_ID,
            )
            print(
                f"[INFO] Uploaded to Drive "
                f"(id: {drive_id})"
            )
        except Exception as e:
            print(
                f"[WARN] Upload failed: {e}"
            )
    last_drive_file_id = drive_id
    send_alert_sms(
        "Alert! Unknown face detected.",
        include_drive=True,
        drive_file_id=drive_id,
    )
    return False
# ============================================================
# FINGERPRINT VERIFICATION PHASE
# ============================================================
def perform_fingerprint_phase():
    """
    Return True if the fingerprint matches.
    On an unmatched fingerprint, send an alert SMS
    and return False.
    """
    if _finger is None:
        print(
            "[FINGER] Sensor not available "
            "-> treating as unmatched."
        )
        send_alert_sms(
            "Alert! Unauthorized fingerprint attempt."
        )
        return False
    print(
        f"[FINGER] Place finger now "
        f"({FINGERPLACE_TIMEOUT}s)..."
    )
    start = time.time()
    finger_placed = False
    while time.time() - start < FINGERPLACE_TIMEOUT:
        if keypad_active:
            return False
        try:
            if _finger.readImage():
                finger_placed = True
                break
        except Exception:
            pass
        time.sleep(0.08)
    if keypad_active:
        return False
    if not finger_placed:
        print("[FINGER] No finger detected.")
        return False
    try:
        _finger.convertImage(0x01)
        result = _finger.searchTemplate()
        pos = result[0]
    except Exception:
        pos = -1
    if pos >= 0:
        print(
            f"[FINGER] Matched at position {pos}"
        )
        return True
    print("[ALERT] Unrecognized fingerprint!")
    send_alert_sms(
        "Alert! Unauthorized fingerprint attempt."
    )
    return False
# ============================================================
# VOICE VERIFICATION PHASE
# ============================================================
def perform_voice_phase():
    """
    Return True if the voice command matches.
    The expected command contains the word "start".
    """
    ok = voice_command_with_prompt()
    if ok:
        print("[VOICE] Confirmed.")
        return True
    print("[ALERT] Voice verification failed.")
    send_alert_sms(
        "Alert! Voice verification failed."
    )
    return False
# ============================================================
# EMERGENCY MODE
# ============================================================
def process_emergency_if_any():
    """
    Process an emergency keypad/password request.
    Return True when an emergency sequence has been
    processed so the main loop can restart.
    """
    global keypad_active, _emergency_result
    if not keypad_active:
        return False
    # Wait for password entry to finish.
    while (
        keypad_active
        and _emergency_result is None
    ):
        time.sleep(0.05)
    if _emergency_result == "matched":
        activate_motor_or_relay()
        _emergency_result = None
        keypad_active = False
        time.sleep(0.5)
        return True
    if _emergency_result == "failed":
        send_alert_sms(
            "Alert! Emergency password failed."
        )
        _emergency_result = None
        keypad_active = False
        return True
    # If password entry was cancelled.
    keypad_active = False
    _emergency_result = None
    return True
# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    global _drive_service, stop_threads
    # ---- Start keypad thread first ----
    threading.Thread(
        target=terminal_thread,
        daemon=True,
    ).start()
    # ---- Load face encodings ----
    threading.Thread(
        target=encodings_loader,
        daemon=True,
    ).start()
    # ---- Start camera stream ----
    print("[INFO] Starting camera stream...")
    proc = start_rpicam_process()
    reader = threading.Thread(
        target=rpicam_reader,
        args=(proc,),
        daemon=True,
    )
    reader.start()
    # ---- Google Drive ----
    try:
        _drive_service = build_drive_service()
        print("[INFO] Drive service ready.")
    except Exception as e:
        print(
            f"[WARN] Could not initialize "
            f"Drive service: {e}"
        )
        _drive_service = None
    print(
        "[MAIN] Running master loop."
    )
    print(
        "[MAIN] Flow: "
        "Face -> Finger -> Voice -> Relay ON."
    )
    print(
        "[MAIN] Keypad is always active."
    )
    try:
        while True:
            # Emergency takes absolute priority.
            if process_emergency_if_any():
                continue
            # =================================================
            # 1. FACE
            # =================================================
            if not perform_face_phase():
                time.sleep(0.2)
                continue
            if process_emergency_if_any():
                continue
            # =================================================
            # 2. FINGERPRINT
            # =================================================
            if not perform_fingerprint_phase():
                time.sleep(0.2)
                continue
            if process_emergency_if_any():
                continue
            # =================================================
            # 3. VOICE
            # =================================================
            if not perform_voice_phase():
                time.sleep(0.2)
                continue
            if process_emergency_if_any():
                continue
            # =================================================
            # 4. ALL THREE MATCHED
            # =================================================
            activate_motor_or_relay()
            # Short settling period before next cycle.
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user.")
    finally:
        stop_threads = True
        time.sleep(0.2)
        # ---- Terminate camera process ----
        try:
            if proc and proc.poll() is None:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
            print(
                "[INFO] Camera process terminated."
            )
        except Exception as e:
            print(
                f"[WARN] Failed to terminate "
                f"camera: {e}"
            )
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            GPIO.output(
                RELAY_PIN,
                GPIO.LOW,
            )
            GPIO.cleanup()
        except Exception:
            pass
# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
