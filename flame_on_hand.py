# 
import cv2
import mediapipe as mp

# flame1 flame2, flame3の各イメージを読み込んでおき，描画フレームごとに使用するイメージを切り替える。
flame_images = [cv2.imread('flame1.png', cv2.IMREAD_UNCHANGED),
               cv2.imread('flame2.png', cv2.IMREAD_UNCHANGED),
               cv2.imread('flame3.png', cv2.IMREAD_UNCHANGED)]



hand_landmarks_data = ...

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the webcam
cap = cv2.VideoCapture(0)

frame_count = 0

while cap.isOpened():
    frame_count += 1

    flame_image=flame_images[int( (frame_count / 4) % 3 )]

    success, frame = cap.read()
    if not success:
        break

    # 画像をRGBに変換する
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手のランドマークを処理する
    results = mp_hands.process(rgb_frame)

    x0,y0,x9,y9,x12,y12 = 0,0,0,0,0,0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                # Landmarkの座標を幅、高さの%からピクセル数に変換
                if i == 0:
                    x0, y0 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if i == 9:
                    x9, y9 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if i == 12:
                    x12, y12 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

    # x9, y9, x12, y12のいずれも0ではない場合
    if x0 * y0 * x9 * y9 * x12 * y12 != 0:
        x, y = int( (x0 + x9) / 2 ), int( (y0 + y9) / 2 )
        # ランドマーク9と12の距離を計算する
        distance = ((x12 - x9) ** 2 + (y12 - y9) ** 2) ** 0.5
        flamesize = distance * 3

        #オーバーレイするイメージの縦がflamesizeになるようリサイズファクターを算出
        resize_factor = flamesize / flame_image.shape[0]

        scaled_flame_image = cv2.resize(flame_image, None, fx=resize_factor, fy=resize_factor)

        # 拡大縮小された炎像の中心を取得
        scaled_flame_center_x = int(scaled_flame_image.shape[1] / 2)
        scaled_flame_center_y = int(scaled_flame_image.shape[0])

        # フレーム上に拡大縮小された炎の画像を配置する位置を計算する。
        flame_position_x = x - scaled_flame_center_x
        flame_position_y = y - scaled_flame_center_y

        # 炎がフレーム内に完全に収まっているか確認する
        if flame_position_x >= 0 and flame_position_y >= 0 and \
            flame_position_x + scaled_flame_image.shape[1] <= frame.shape[1] and \
            flame_position_y + scaled_flame_image.shape[0] <= frame.shape[0]:
            # 炎の画像からアルファチャンネルを抽出する
            b, g, r, a = cv2.split(scaled_flame_image)

            # アルファチャンネルからマスクを作成する
            mask = a
            # マスクを反転させる
            inv_mask = cv2.bitwise_not(mask)

            #rgb_frameから関心領域（ROI）を抽出する。
            x, y = flame_position_x, flame_position_y  # Example position
            roi = rgb_frame[y:y+scaled_flame_image.shape[0], x:x+scaled_flame_image.shape[1]]

            # マスクを使って炎領域と背景領域を抽出する。
            flame_region = cv2.merge((r, g, b))
            flame_region = cv2.bitwise_and(flame_region, flame_region, mask=mask)
            background_region = cv2.bitwise_and(roi, roi, mask=inv_mask)

            # 炎領域と背景領域を追加する
            combined = cv2.add(flame_region, background_region)

            # 合成した画像を元のフレームに戻す
            rgb_frame[y:y+scaled_flame_image.shape[0], x:x+scaled_flame_image.shape[1]] = combined

    # Convert the image back to BGR for display
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('Flame on Hand', output_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
