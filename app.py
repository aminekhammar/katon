import threading
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import streamlit as st

TOTAL_CAMS     = 12
MIN_CONF       = 0.15
FIRE_CLS       = 0
FPS_TARGET     = 8
CAM_INDEX_MAP  = {i: i - 1 for i in range(1, TOTAL_CAMS + 1)}

st.set_page_config(
    page_title="tiziri forest",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        m = YOLO("best.pt")
        m.to("cpu")
        print(f"[INFO] classes: {m.names}")
        return m, True
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, False

model, model_ok = load_model()

def new_cam_state(cid):
    return {
        "id":          cid,
        "cam_idx":     CAM_INDEX_MAP[cid],
        "available":   None,
        "running":     False,
        "stop":        False,
        "frame":       None,
        "fire":        False,
        "peak_conf":   0.0,
        "frame_count": 0,
        "fire_count":  0,
    }

def init_state():
    return {
        "cams":    {i: new_cam_state(i) for i in range(1, TOTAL_CAMS + 1)},
        "log":     deque(maxlen=30),
        "lock":    threading.Lock(),
        "booted":  False,
    }

if "shared" not in st.session_state:
    st.session_state["shared"] = init_state()

shared = st.session_state["shared"]


def run_detection(frame_bgr: np.ndarray):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if not model_ok:
        return rgb, False, 0.0

    res = model(frame_bgr, verbose=False, conf=MIN_CONF)[0]

    best_conf = 0.0
    detected  = False

    for box in res.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])

        print(f"[DEBUG] cls={cls} ({model.names.get(cls, '?')}) conf={conf:.2f}")

        if cls != FIRE_CLS:
            continue

        if conf > best_conf:
            best_conf = conf

        if conf >= MIN_CONF:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 60, 60), 3)
            cv2.putText(rgb, f"FIRE {int(conf*100)}%",
                        (x1, max(y1 - 8, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 60, 60), 2)

    if detected:
        cv2.rectangle(rgb, (0, 0), (w, 44), (200, 0, 0), -1)
        cv2.putText(rgb, f"  FIRE DETECTED  {int(best_conf*100)}%",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    else:
        cv2.rectangle(rgb, (0, 0), (w, 38), (0, 150, 0), -1)
        cv2.putText(rgb, f"  Clear  max={int(best_conf*100)}%",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(rgb, datetime.now().strftime("%H:%M:%S"),
                (w - 90, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return rgb, detected, best_conf


def cam_worker(cid: int, sh: dict):
    cam   = sh["cams"][cid]
    idx   = cam["cam_idx"]
    delay = 1.0 / FPS_TARGET
    last_alert = 0.0

    cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        with sh["lock"]:
            cam["available"] = False
            cam["running"]   = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    with sh["lock"]:
        cam["available"] = True
        cam["running"]   = True

    try:
        while not cam["stop"]:
            t0 = time.time()
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            rgb, fire, conf = run_detection(bgr)

            with sh["lock"]:
                cam["frame"]        = rgb
                cam["frame_count"] += 1
                if conf > cam["peak_conf"]:
                    cam["peak_conf"] = conf
                if fire:
                    cam["fire_count"] += 1
                    cam["fire"]        = True
                    now = time.time()
                    if now - last_alert > 30:
                        last_alert = now
                        sh["log"].appendleft({
                            "time":  datetime.now().strftime("%H:%M:%S"),
                            "cam":   f"Unit {cid}",
                            "conf":  conf,
                            "hits":  cam["fire_count"],
                            "total": cam["frame_count"],
                        })

            time.sleep(max(0.0, delay - (time.time() - t0)))

    finally:
        cap.release()
        with sh["lock"]:
            cam["running"] = False


if not shared["booted"]:
    shared["booted"] = True
    for cid in range(1, TOTAL_CAMS + 1):
        t = threading.Thread(target=cam_worker, args=(cid, shared), daemon=True)
        t.start()

if "selected_unit" not in st.session_state:
    st.session_state["selected_unit"] = None

if st.session_state.selected_unit is not None:
    cid = st.session_state.selected_unit

    if st.button("← Back to Map"):
        st.session_state.selected_unit = None
        st.rerun()

    with shared["lock"]:
        snap  = {**shared["cams"][cid]}
        frame = shared["cams"][cid]["frame"]

    avail   = snap["available"]
    on_fire = snap["fire"]

    st.title(f"{'🔥' if on_fire else '📹'} Unit {cid} — Camera {snap['cam_idx']}")

    if avail is None:
        st.warning("⏳ Opening camera...")
    elif avail is False:
        st.error(f"❌ Camera {snap['cam_idx']} unavailable")
        st.stop()
    else:
        st.success(f"🟢 Camera {snap['cam_idx']} running")

    if model_ok:
        with st.expander("🔍 Model Info"):
            st.write(f"**Classes:** {model.names}")
            st.write(f"**Fire Class ID:** {FIRE_CLS}")
            st.write(f"**Confidence Threshold:** {MIN_CONF}")

    c1, c2, c3 = st.columns(3)
    ph_total = c1.empty()
    ph_fire  = c2.empty()
    ph_conf  = c3.empty()
    ph_total.metric("Frames", snap["frame_count"])
    ph_fire.metric("🔥 Fire Frames", snap["fire_count"])
    ph_conf.metric("Peak Conf", f"{int(snap['peak_conf']*100)}%")

    st.divider()
    ph_alert = st.empty()
    ph_frame = st.empty()

    while True:
        with shared["lock"]:
            live  = shared["cams"][cid]
            frame = live["frame"]
            on_fire  = live["fire"]
            conf_val = live["peak_conf"]
            total    = live["frame_count"]
            fires    = live["fire_count"]

        if on_fire:
            ph_alert.error(f"🔥 Fire Detected! Peak: {int(conf_val*100)}%")
        else:
            ph_alert.empty()

        if frame is not None:
            ph_frame.image(frame,
                           caption=f"Unit {cid} (cam {CAM_INDEX_MAP[cid]}) — frame #{total}",
                           use_container_width=True)
        else:
            ph_frame.info("⏳ Waiting for frame...")

        ph_total.metric("Frames", total)
        ph_fire.metric("🔥 Fire Frames", fires)
        ph_conf.metric("Peak Conf", f"{int(conf_val*100)}%")

        time.sleep(0.1)

with shared["lock"]:
    all_cams = {cid: {**shared["cams"][cid]} for cid in range(1, TOTAL_CAMS + 1)}
    log      = list(shared["log"])

fire_total    = sum(c["fire"]              for c in all_cams.values())
active_total  = sum(c["running"]           for c in all_cams.values())
down_total    = sum(c["available"] is False for c in all_cams.values())

st.title("🌲 Tiziri Forest — Smart Surveillance System")
st.caption(
    f"YOLO: {'✅ Loaded' if model_ok else '❌ Unavailable'} | "
    f"📹 Active: {active_total} | "
    f"❌ Down: {down_total} | "
    f"🔥 Alerts: {fire_total}"
)

if model_ok:
    with st.expander("🔍 Model Info"):
        st.write(f"**Classes:** {model.names}")
        st.write(f"**Fire Class ID:** `{FIRE_CLS}` → `{model.names.get(FIRE_CLS, 'NOT FOUND ❌')}`")
        st.write(f"**Confidence Threshold:** `{MIN_CONF}`")
        if model.names.get(FIRE_CLS) is None:
            st.error(f"⚠️ FIRE_CLS={FIRE_CLS} not in model classes — check FIRE_CLS value.")

with st.sidebar:
    st.header("📡 Alerts Log")
    if log:
        for entry in log:
            st.error(
                f"🔥 {entry['cam']}  \n"
                f"🕐 {entry['time']} — Conf: {int(entry['conf']*100)}%  \n"
                f"🎞️ {entry['hits']}/{entry['total']} frames"
            )
    else:
        st.success("✅ No alerts — all clear")

    if not model_ok:
        st.warning("To enable YOLO:\n```\npip install ultralytics opencv-python\n```\nPlace best.pt here")

    st.divider()
    st.subheader("🗺️ Camera Status")
    for cid, cam in all_cams.items():
        avail = cam["available"]
        if avail is True and cam["running"]:
            st.success(f"Unit {cid} — Cam {cam['cam_idx']} 🟢")
        elif avail is False:
            st.error(f"Unit {cid} — Cam {cam['cam_idx']} ❌")
        else:
            st.warning(f"Unit {cid} — ⏳ Checking...")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📹 Total Units",   TOTAL_CAMS)
col2.metric("🟢 Active",        active_total)
col3.metric("❌ Down",           down_total)
col4.metric("🔥 Alerts",        fire_total)

st.divider()
st.subheader("🗺️ Unit Map")

rows = [list(range(1, TOTAL_CAMS + 1))[i:i+4] for i in range(0, TOTAL_CAMS, 4)]
for row in rows:
    cols = st.columns(4)
    for col, cid in zip(cols, row):
        with col:
            cam   = all_cams[cid]
            avail = cam["available"]

            if avail is False:
                st.warning(f"📷 Unit {cid}\nCam {cam['cam_idx']} ❌")
                st.button("Unavailable", key=f"c{cid}",
                          use_container_width=True, disabled=True)
            elif cam["fire"]:
                st.error(f"🔥 Unit {cid}\nConf: {int(cam['peak_conf']*100)}%")
                if st.button("🔥 View", key=f"c{cid}", use_container_width=True):
                    st.session_state.selected_unit = cid
                    st.rerun()
            elif cam["running"]:
                st.success(f"✅ Unit {cid}\n{cam['frame_count']} frames")
                if st.button("📹 View", key=f"c{cid}", use_container_width=True):
                    st.session_state.selected_unit = cid
                    st.rerun()
            else:
                st.info(f"⏳ Unit {cid}\nCam {cam['cam_idx']}")
                if st.button("📹 View", key=f"c{cid}", use_container_width=True):
                    st.session_state.selected_unit = cid
                    st.rerun()

time.sleep(3)
st.rerun()