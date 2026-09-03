"""
Webcam gesture -> futbolista meme detector (desktop version).

Abre dos ventanas lado a lado:
  - "Camera": video en vivo de tu webcam con landmarks dibujados
  - "Meme": la foto del futbolista correspondiente al gesto que estés haciendo

Futbolistas y Gestos reconocidos:
  - default (sin manos / neutral)        -> memes/Bielsa_poker.jpg (Marcelo Bielsa con cara de póker)
  - puño cerrado (fist ✊)                 -> memes/Mbappe.jpg (Kylian Mbappé festejando con el puño)
  - mirada de reojo (side-eye 👀)         -> memes/Ancelotti_sideeye.jpg (Carlo Ancelotti con ceja levantada)
  - dedo del medio (🖕)                  -> memes/Messi.jpeg (Lionel Messi con el dedo del medio)
  - cuernos de rock (🤘)                 -> memes/Charles.jpeg (Charles Aránguiz con señal de rock)
  - dos manos en las orejas (🤪)          -> memes/Neymar.jpeg (Neymar con festejo en las orejas)
  - 3 dedos arriba (3️⃣)                  -> memes/DiMaria.jpeg (Ángel Di María mostrando 3 dedos)
  - máscara sobre la cara (🎭)           -> memes/Dybala.jpeg (Paulo Dybala mask sobre nariz/boca)
  - palma abierta 5 dedos (🖐️)           -> memes/Cristiano.jpeg (Cristiano Ronaldo con 5 dedos abiertos)
  - 3-0-4 con 2 manos                    -> memes/Lamine.jpeg (Lamine Yamal cruzando manos 3 y 4)

Presiona 'q' o 'ESC' para salir.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "imagenes" if (ROOT / "imagenes").exists() else ROOT / "memes"

# Mapeo de gestos a imágenes de futbolistas
GESTURE_MEMES = {
    "default": ["Bielsa_poker.jpg"],
    "fist": ["Mbappe.jpg"],
    "sideEye": ["Ancelotti_sideeye.jpg"],
    "messi": ["Messi.jpeg"],
    "charles": ["Charles.jpeg"],
    "neymar": ["Neymar.jpeg"],
    "diMaria": ["DiMaria.jpeg"],
    "dybala": ["Dybala.jpeg"],
    "cristiano": ["Cristiano.jpeg"],
    "lamine304": ["Lamine.jpeg"],
}

STABLE_FRAMES_REQUIRED = 4
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200
SIDE_EYE_YAW_DEG = 15.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- Funciones geométricas y de landmarks --------------------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip, wrist=0):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    angle_ok = angle_deg(v1, v2) < 45
    dist_ok = dist(pts[tip], pts[wrist]) > dist(pts[pip], pts[wrist])
    return angle_ok and dist_ok


def yaw_from_transform_matrix(matrix):
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    # Pulgar extendido
    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_mcp_dist = dist(pts[4], pts[2])
    thumb_pip_dist = dist(pts[3], pts[2])
    thumb_out = (thumb_pinky_spread > 0.95) or (thumb_mcp_dist > thumb_pip_dist and dist(pts[4], pts[5]) > 0.55 * hand_scale)

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "middleTip": pts[12],
        "thumbTip": pts[4],
        "wrist": pts[0],
        "palmCenter": pts[9],
    }


class GestureState:
    def __init__(self):
        self.last_face = None  # (mouth_center, face_width, mouth_open, yaw_deg, t)
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            yaw_deg = 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg = yaw_from_transform_matrix(face_result.facial_transformation_matrixes[0])

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, now)
            self.last_yaw_debug = yaw_deg
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and (now - self.last_face[4] < FACE_STALE_MS)

        # Sin manos detectadas: evaluar mirada de reojo (Ancelotti) o default (Bielsa)
        if not hand_result.hand_landmarks:
            if face_is_fresh and abs(self.last_yaw_debug) > SIDE_EYE_YAW_DEG:
                return "sideEye"
            return "default"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        # =========================================================================
        # 1. GESTOS CON DOS MANOS
        # =========================================================================
        if len(hands) == 2:
            h0, h1 = hands[0], hands[1]
            ext0 = sum([h0["indexUp"], h0["middleUp"], h0["ringUp"], h0["pinkyUp"]])
            ext1 = sum([h1["indexUp"], h1["middleUp"], h1["ringUp"], h1["pinkyUp"]])

            # Lamine Yamal (3-0-4): Una mano mostrando 3 dedos y la otra 4 (o mano abierta)
            if (ext0 == 3 and ext1 >= 3) or (ext1 == 3 and ext0 >= 3):
                if not (ext0 == 4 and ext1 == 4):
                    return "lamine304"

            # Neymar: Ambas manos abiertas al costado de las orejas (🤪)
            if ext0 >= 4 and ext1 >= 4:
                if face_is_fresh:
                    mouth_center, face_width, _, _, _ = self.last_face
                    at_ears_y = (h0["palmCenter"][1] < mouth_center[1] + face_width * 0.6 and
                                 h1["palmCenter"][1] < mouth_center[1] + face_width * 0.6)
                    separated_x = abs(h0["palmCenter"][0] - h1["palmCenter"][0]) > face_width * 0.8
                    if at_ears_y and separated_x:
                        return "neymar"
                else:
                    return "neymar"

        # =========================================================================
        # 2. GESTOS CON UNA MANO
        # =========================================================================
        h = hands[0]

        # A. Messi: Dedo del medio levantado exclusivamente (🖕)
        if h["middleUp"] and not h["indexUp"] and not h["ringUp"] and not h["pinkyUp"]:
            return "messi"

        # B. Charles Aránguiz: Cuernos / Rock (🤘) (índice y meñique arriba, medio y anular abajo)
        if h["indexUp"] and h["pinkyUp"] and not h["middleUp"] and not h["ringUp"]:
            return "charles"

        # C. Di María: 3 dedos (3️⃣)
        # Modo A (foto Di María): pulgar + índice + medio arriba, anular y meñique abajo
        di_maria_a = h["thumbOut"] and h["indexUp"] and h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]
        # Modo B (3 clásico): índice + medio + anular arriba, meñique abajo
        di_maria_b = h["indexUp"] and h["middleUp"] and h["ringUp"] and not h["pinkyUp"]
        if di_maria_a or di_maria_b:
            return "diMaria"

        # D. Kylian Mbappé: Puño cerrado (✊)
        if h["curledCount"] == 4:
            return "fist"

        # E. Dybala Mask: Mano cubriendo la parte inferior del rostro (nariz/boca con índice/pulgar)
        if face_is_fresh:
            mouth_center, face_width, _, _, _ = self.last_face
            d_mouth = dist(h["palmCenter"], mouth_center) / face_width
            d_index = dist(h["indexTip"], mouth_center) / face_width
            if (d_mouth < 0.9 or d_index < 0.6) and h["curledCount"] >= 2:
                return "dybala"

        # F. Cristiano Ronaldo: Palma abierta mostrando los 5 dedos (🖐️)
        if h["curledCount"] == 0:
            return "cristiano"

        # G. Mirada de reojo (Ancelotti) si la cabeza está girada y no hay gesto específico
        if face_is_fresh and abs(self.last_yaw_debug) > SIDE_EYE_YAW_DEG:
            return "sideEye"

        return "default"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        imgs = []
        for name in files:
            p = MEMES / name
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen del meme: {p}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    lines = [
        f"Gesto detectado: {gesture}",
        f"Giro de cabeza (yaw): {state.last_yaw_debug:+.1f} deg  (umbral: +/-{SIDE_EYE_YAW_DEG:.1f})",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 26
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara web (index 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # Modo espejo

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                current_meme = random.choice(memes[gesture])

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
