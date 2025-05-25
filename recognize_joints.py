import cv2
import mediapipe as mp

#ウェブカメラの初期化
cap = cv2.VideoCapture(0)

#手検出モデルを初期化する
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    #ウェブカメラからフレームを読み込む
    ret, frame = cap.read()

    if not ret:
        break

    # 画像をRGBに変換する
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #手検出モデルでフレームを処理する
    results = mp_hands.process(rgb_frame)

    #フレームに手のランドマークを描く
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #landmarkの数だけ繰り返し 各Landmarkのindexを変数iに代入
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                # iの値を円の隣に表示
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                


    # フレームをウィンドウに表示する
    cv2.imshow('Webcam Feed with Joint Positions', frame)

    # q'が押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ウェブカメラを解除し、すべてのウィンドウを閉じます。
cap.release()
cv2.destroyAllWindows()
