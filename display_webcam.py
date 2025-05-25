# Webカメラの映像をPythonで取得するサンプル
# 本サンプルは取得した映像を変更せず、そのまま表示している
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # ウェブカメラからフレームを読み込む
    ret, frame = cap.read()

    if not ret:
        break

   # フレームをウィンドウに表示する
    cv2.imshow('Webcam Feed', frame)

   # q'が押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ウェブカメラを解除し、すべてのウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
