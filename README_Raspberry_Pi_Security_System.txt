Raspberry Pi Multimodal Security System

Files:
- Raspberry_Pi_Multimodal_Security_System.py: Python source extracted from the supplied PDF and syntax-checked.
- requirements_raspberry_pi.txt: Python dependency list.

Important:
1. This program is designed for Raspberry Pi OS and Raspberry Pi GPIO/UART hardware; it will not run as-is on ordinary Windows because RPi.GPIO, serial ports, rpicam-vid, and hardware sensors are Raspberry Pi-specific.
2. Before running, verify RELAY_PIN, GPS_PORT, GSM_PORT, FINGER_UART, relay active level, and wiring.
3. Set a real User-Agent contact in USER_AGENT and check PRIMARY_PHONE and TERMINAL_PASSWORD before use.
4. Google Drive upload requires a valid credentials.json OAuth client file in the working directory. token.json is generated after authorization.
5. Create dataset/<person_name>/ and place that person's training images inside it.
6. Syntax validation does not verify hardware, credentials, or runtime behavior. Test with the relay/motor disconnected first.

On Raspberry Pi, from the directory containing the script:
    python3 -m pip install -r requirements_raspberry_pi.txt
    python3 Raspberry_Pi_Multimodal_Security_System.py

If pip refuses system-wide installation on Raspberry Pi OS, use a virtual environment where supported or install packages through apt as appropriate. face_recognition/dlib and PyAudio can need additional OS-level dependencies.
