"""Opens Computrabajo when you pick up your phone in front of the webcam.

Watches the webcam with a YOLO model. When a phone shows up in a few recent
frames, it opens https://co.computrabajo.com in Chrome (or Edge). The page
closes again when the phone disappears or nobody is in front of the camera.

Windows only. Setup:
    python -m venv .venv
    .venv\\Scripts\\pip install -r requirements.txt
    python phone_watch.py            (add --hidden to skip the camera window)
"""
import argparse
import collections
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform != "win32":
    sys.exit("This program only runs on Windows.")

script_dir = Path(__file__).resolve().parent
venv_python = script_dir / ".venv" / "Scripts" / "python.exe"

# OpenCV, YOLO and pygrabber are only installed in .venv. If the script was
# started with another Python (e.g. double-clicked), run it again with the
# .venv one and quit, so the imports below don't fail.
current_python = Path(sys.executable).resolve()

if venv_python.exists() and current_python != venv_python.resolve():
    exit_code = subprocess.call([str(venv_python), __file__, *sys.argv[1:]])
    sys.exit(exit_code)

import cv2
from pygrabber.dshow_graph import FilterGraph
from ultralytics import YOLO


URL = "https://co.computrabajo.com"
CAMERA_INDEX = 0
MIN_CONFIDENCE = 0.40
WINDOW = 6
NEEDED = 4
PHONE_GONE_SECONDS = 0.50
ABSENT_SECONDS = 3
SKIP_CAMERAS = ("OBS", "Virtual")


BROWSERS = (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe")
PAGE_SIZE = (1600, 1400)
PAGE_POSITION = (160, 0)
PROFILE = script_dir / ".browser-profile"  # separate browser profile, kept out of git


def find_browser():
    roots = [os.environ.get(v) for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    for browser in BROWSERS:
        for root in filter(None, roots):
            path = Path(root) / browser
            if path.exists():
                return path
    raise SystemExit("Could not find Chrome or Edge.")


def open_camera(preferred):
    names = FilterGraph().get_input_devices()
    real = [i for i, name in enumerate(names)
            if not any(word.lower() in name.lower() for word in SKIP_CAMERAS)]
    for index in sorted(real, key=lambda i: i != preferred):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        deadline = time.time() + 5
        while cam.isOpened() and time.time() < deadline:
            if cam.read()[0]:
                print(f"Using camera {index}: {names[index]}")
                return cam
            time.sleep(0.2)
        cam.release()
        print(f"Camera {index} ({names[index]}) did not work")
    return None


def apps_using_camera():
    # Windows keeps a list of apps that used the webcam in the registry (the same
    # one you see in Settings > Privacy > Camera). If an app's LastUsedTimeStop
    # is 0, it still has the camera open.
    import winreg

    webcam_key = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
    apps = []

    for folder in [webcam_key, webcam_key + r"\NonPackaged"]:
        try:
            parent = winreg.OpenKey(winreg.HKEY_CURRENT_USER, folder)
        except OSError:
            continue

        app_count = winreg.QueryInfoKey(parent)[0]
        for i in range(app_count):
            app_name = winreg.EnumKey(parent, i)
            try:
                with winreg.OpenKey(parent, app_name) as app_key:
                    stopped_at, _ = winreg.QueryValueEx(app_key, "LastUsedTimeStop")
            except OSError:
                continue

            if stopped_at == 0:
                # normal programs are saved as file paths with # instead of \
                apps.append(app_name.replace("#", "\\"))

    return apps


class Page:
    def __init__(self):
        self.browser = find_browser()
        self.proc = None

    def is_open(self):
        return self.proc is not None and self.proc.poll() is None

    def open(self):
        self.proc = subprocess.Popen([
            str(self.browser), f"--user-data-dir={PROFILE}", "--no-first-run",
            "--no-default-browser-check", "--new-window",
            "--window-size={},{}".format(*PAGE_SIZE),
            "--window-position={},{}".format(*PAGE_POSITION), URL,
        ])

    def close(self):
        if self.is_open():
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
        self.proc = None


def log(message):
    print(time.strftime("%H:%M:%S"), message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", action="store_true", help="don't show the camera window")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    model = YOLO(str(script_dir / "yolov8n.pt"))  # small, fast model; downloads itself the first time
    phone_class = next(i for i, name in model.names.items() if name == "cell phone")
    person_class = next(i for i, name in model.names.items() if name == "person")

    cam = open_camera(args.camera)
    if cam is None:
        busy = apps_using_camera()
        if busy:
            print("These apps are using the camera right now - close them and try again:")
            for app in busy:
                print("   ", app)
            if any(app.lower().endswith("python.exe") for app in busy):
                print("    (python.exe is probably an older copy of this program still running)")
        raise SystemExit("Could not open the webcam.")

    page = Page()
    recent = collections.deque(maxlen=WINDOW)
    armed = True
    last_phone = 0.0
    last_person = time.time()
    log("Watching... (press Q in the camera window or Ctrl+C to stop)")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.5)
                continue

            result = model(frame, imgsz=320, conf=MIN_CONFIDENCE,
                           classes=[phone_class, person_class], verbose=False)[0]
            found = [int(c) for c in result.boxes.cls]
            phone = phone_class in found
            person = person_class in found or phone
            recent.append(phone)
            now = time.time()

            if phone:
                last_phone = now
            if person:
                last_person = now
            phone_gone = now - last_phone >= PHONE_GONE_SECONDS
            nobody = now - last_person >= ABSENT_SECONDS

            if page.is_open() and (phone_gone or nobody):

                page.close()
            if phone_gone and not armed:
                armed = True


            if armed and sum(recent) >= NEEDED and not page.is_open():

                page.open()
                armed = False
                recent.clear()

            if not args.hidden:
                view = result.plot()
                """
                if page.is_open():
                    status, color = "Computrabajo OPEN", (0, 165, 255)
                elif armed:
                    status, color = "READY - pick up your phone", (0, 255, 0)
                else:
                    status, color = "put the phone down to re-arm", (0, 165, 255)
                cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if not person:
                    cv2.putText(view, f"nobody here ({now - last_person:.0f}s)", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif page.is_open() and not phone:
                    cv2.putText(view, f"no phone ({now - last_phone:.0f}s)", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                """
                cv2.imshow("Computrabajo CAM", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
