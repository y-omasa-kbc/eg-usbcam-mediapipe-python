import cv2
import mediapipe as mp

# MediaPipeのセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 手のランドマーク検出を初期化
hands = mp_hands.Hands(
    static_image_mode=False,  # 動画ストリームに対応
    max_num_hands=2,         # 同時に検出する手の最大数
    min_detection_confidence=0.5,  # 検出信頼度
    min_tracking_confidence=0.5    # トラッキング信頼度
)

# カメラを開く
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラから画像を取得できませんでした")
        break

    # カメラ画像をBGRからRGBに変換（MediaPipeはRGBを使用）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手のランドマークを検出
    result = hands.process(rgb_frame)

    # 検出結果を描画
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 各ランドマークの座標を取得して表示
            for idx, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                print(f"ランドマーク {idx}: ({x}, {y})")
                
            # フレームにランドマークを描画
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    # 結果を表示
    cv2.imshow('Hand Detection', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
hands.close()
