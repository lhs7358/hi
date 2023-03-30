import cv2
import math

# 얼굴을 감지하기 위한 미리 학습된 Haar Cascade 분류기(대상 분류 알고리즘) 로드
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# 기본 카메라로부터 비디오 캡처를 시작
cap = cv2.VideoCapture(0)

# 비디오 스트림의 각 프레임을 반복 처리
while True:
    # 비디오 스트림으로부터 프레임을 읽어오기
    ret, frame = cap.read()

    # 화면 중앙 픽셀값 960 540
    height, width = frame.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)

    # 프레임을 회색조 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 회색조 이미지에서 얼굴을 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

    # 감지된 얼굴 주변에 초록색 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 얼굴 사각형 내의 영역을 ROI(관심 영역 지정)로 정의
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 오른쪽 끝은 1700, 왼쪽 끝은 300(맥북 카메라기준)
        cv2_rectangle_center = [int(x + w / 2), int(y + h / 2)]

        # 픽셀값과 실제 가로, 세로 길이 비율
        proportion_x = 40/700
        proportion_y = 50/150000

        # 비율 적용한 가로 길이(화면상의 정가운데 픽셀값 - 이동한거리에 따른 픽셀값 * 비율)
        real_x = (center_x - cv2_rectangle_center[0])*proportion_x

        # 비율 적용한 세로 길이(화면에 인식되는 얼굴의 사각형의 넓이 * 비율)
        real_y = w*h*proportion_y

        # 빗변 길이(피타고라스 사용)
        real_r = (real_x**2 + real_y**2)**0.5

        # 높이/빗변으로 sin값 받아오기
        sin_x = real_x/real_r

        # sin값이 -1에서 1사이가 아닐 경우 예외 처리
        if not -1 <= sin_x <= 1:
            continue

        # 아크사인값으로 각도값 받아오기 및 디그리 변환
        rad = math.asin(sin_x)
        degree = round(math.degrees(rad), -1)

        print(degree)

    # 감지된 얼굴과 ROIs가 포함된 프레임을 표시
    cv2.imshow('Video', frame)

    # 'q' 키가 눌리면 반복문을 종료합니다
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처를 해제 및 윈도우 종료
cap.release()
cv2.destroyAllWindows()
