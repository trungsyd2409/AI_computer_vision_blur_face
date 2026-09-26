# Pinch to Pixelate

Project computer vision: **khoảng cách giữa ngón cái và ngón trỏ tay phải điều khiển mức pixel hoá (mosaic) của tất cả khuôn mặt** trên camera.

- Chụm 2 ngón lại → mặt rõ nét (0%)
- Mở 2 ngón ra xa → mặt vỡ pixel (100%)
- Bỏ tay ra khỏi khung hình → mức pixel **được giữ nguyên** (trạng thái `LOCKED`)
- FPS hiển thị ở **góc trên bên phải**, có phím tắt để đổi FPS mục tiêu của camera
- Camera được tối ưu tự động (MJPG + DirectShow + đọc frame ở luồng riêng) để lên được 30 FPS

Công nghệ: Python, OpenCV, MediaPipe Tasks API (Hand Landmarker + Face Detector).

---

## 1. Cài đặt (Windows)

Yêu cầu: **Python 3.10 – 3.12**, webcam.

Mở terminal (PowerShell) trong thư mục project:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Lần chạy đầu tiên, app sẽ **tự tải 2 file model** (~8 MB) vào thư mục `models/`.
> Nếu không tải được (mạng chặn), hãy tải tay và bỏ vào `models/`:
> - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
> - https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

## 2. Chạy

```powershell
python main.py                 # webcam mặc định
python main.py --camera 1      # dùng webcam khác (0, 1, 2...)
python main.py --source video.mp4   # test bằng file video
```

## 3. Cách dùng

| Hành động | Kết quả |
|---|---|
| Giơ **tay phải**, chụm ngón cái + ngón trỏ | Mức pixel về 0% |
| Mở rộng 2 ngón | Mức pixel tăng dần tới 100% |
| Bỏ tay ra khỏi khung hình | Giữ nguyên mức cuối (`LOCKED` màu cam) |
| Giơ tay trái | Bị bỏ qua (skeleton màu xám) |

Khoảng cách được **chia cho kích thước lòng bàn tay** (cổ tay → khớp ngón giữa), nên đứng gần hay xa camera thì cử chỉ vẫn cho cùng kết quả. Mức pixel được làm mượt (smoothing) để không bị giật.

### Phím tắt

| Phím | Chức năng |
|---|---|
| `1` `2` `3` `4` | FPS mục tiêu 15 / 24 / 30 / 60 |
| `+` / `-` | FPS mục tiêu +5 / −5 |
| `P` | Mở bảng setting gốc của driver webcam (Windows) |
| `M` | Bật/tắt chế độ gương (mirror) |
| `O` | Bật/tắt skeleton bàn tay + đường nối 2 ngón |
| `R` | Reset mức pixel về 0% |
| `H` | Hiện/ẩn bảng phím tắt |
| `Q` / `Esc` | Thoát (tự lưu setting) |

### Đọc thông tin FPS (góc trên phải)

```
FPS 29.7                    ← FPS thực tế của app (sau khi xử lý AI)
CAM 30.0 | target 30        ← FPS camera thật sự gửi về | FPS bạn yêu cầu
DSHOW MJPG 1280x720         ← backend | định dạng ảnh | độ phân giải
```

Màu số FPS: **xanh** ≥ 90% mục tiêu, **vàng** ≥ 50%, **đỏ** < 50%.

- `CAM` thấp (≈10) → vấn đề nằm ở **camera** (xem mục 4).
- `CAM` cao nhưng `FPS` thấp → **máy xử lý không kịp**, giảm `detect_width` hoặc độ phân giải trong `settings.json`.

## 4. Vì sao camera chỉ chạy 10 FPS? Cách khắc phục

App đã tự làm 3 việc sau (bạn không cần làm gì):

1. **Dùng codec MJPG**: mặc định webcam trên Windows gửi ảnh dạng thô YUY2, ở 720p/1080p băng thông USB chỉ đủ ~5–10 FPS. MJPG (ảnh nén) cho phép 30+ FPS. Nếu dòng thông tin hiện `YUY2` thay vì `MJPG`, camera của bạn không hỗ trợ MJPG ở độ phân giải đó → thử giảm độ phân giải.
2. **Dùng backend DirectShow** (`DSHOW`) – thường nhận lệnh FPS/MJPG tốt hơn MSMF mặc định.
3. **Đọc camera ở luồng riêng** – xử lý AI không làm chậm việc lấy ảnh.

Nếu vẫn thấp, thử theo thứ tự:

1. **Bật thêm đèn**. Phòng tối → camera tự tăng thời gian phơi sáng (ví dụ 1/10 giây) → tối đa chỉ 10 FPS. Đây là nguyên nhân rất phổ biến.
2. Nhấn **`P`** để mở bảng setting driver, tắt **Low Light Compensation** / **Auto Exposure** (tên tuỳ hãng), rồi chỉnh Exposure tay.
3. Giảm độ phân giải: sửa `settings.json` → `"width": 640, "height": 480`.
4. Đóng các app khác đang dùng camera (Zoom, Teams, OBS, Camera).
5. Cắm webcam vào cổng USB 3.0, không qua hub.

## 5. File `settings.json`

Được tạo tự động khi thoát app. Bạn có thể sửa bằng tay (khi app đang tắt):

| Key | Mặc định | Ý nghĩa |
|---|---|---|
| `camera_index` | 0 | Webcam số mấy |
| `backend` | `"auto"` | `auto` / `dshow` / `msmf` / `any` |
| `width`, `height` | 1280, 720 | Độ phân giải yêu cầu |
| `target_fps` | 30 | FPS mục tiêu |
| `use_mjpg` | true | Dùng codec MJPG |
| `mirror` | true | Lật ảnh như gương |
| `control_hand` | `"Right"` | Tay điều khiển: `Right` / `Left` / `Any` |
| `pinch_min_ratio` | 0.25 | Tỉ lệ ngón/lòng bàn tay ứng với 0% |
| `pinch_max_ratio` | 1.5 | Tỉ lệ ứng với 100% (giảm xuống nếu phải mở tay quá rộng) |
| `smoothing_tau` | 0.08 | Độ mượt (giây). Lớn hơn = mượt hơn nhưng chậm hơn |
| `max_blocks` / `min_blocks` | 48 / 4 | Số ô pixel ngang mặt ở mức ~0% / 100% |
| `face_padding` | 0.25 | Nới rộng vùng mặt thêm 25% |
| `face_hold_sec` | 0.35 | Giữ vùng mặt cũ nếu mất nhận diện tạm thời (tránh lộ mặt 1 frame) |
| `detect_width` | 640 | Ảnh được thu nhỏ về chiều rộng này trước khi đưa vào AI (nhỏ = nhanh) |
| `last_pixel_level` | 0.0 | Mức pixel lần trước (tự lưu) |

Xoá `settings.json` để về mặc định.

## 6. Cấu trúc code

```
main.py            Vòng lặp chính: đọc camera, gọi pipeline, vẽ HUD, xử lý phím
app/
  camera.py        ThreadedCamera: MJPG + DirectShow + luồng đọc riêng, đo FPS camera
  detectors.py     MediaPipe HandLandmarker + FaceDetector, tự tải model
  gesture.py       Tính tỉ lệ pinch, chọn tay phải, làm mượt, khoá mức khi mất tay
  effects.py       Hiệu ứng pixel hoá + giữ vùng mặt khi detection chập chờn
  hud.py           Vẽ FPS, thanh %, skeleton, bảng phím tắt
  pipeline.py      Ghép tất cả: 1 frame vào → 1 frame đã xử lý ra
  settings.py      Đọc / ghi settings.json
```

Luồng xử lý mỗi frame:

```
Camera (thread) → lật gương → thu nhỏ → MediaPipe (tay + mặt)
   → chọn tay phải → tỉ lệ pinch → mức pixel (làm mượt / LOCKED)
   → pixel hoá từng mặt → vẽ overlay + FPS → hiển thị
```

## 7. Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `Cannot open camera/video source: 0` | Camera đang bị app khác dùng, hoặc thử `--camera 1` |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Không liên quan project này (project dùng Tasks API mới). Nếu gặp ở code khác: bản mediapipe mới đã bỏ `mp.solutions` |
| `cv2` lỗi lạ sau khi cài | Có thể bị cài cả `opencv-python` và `opencv-contrib-python`. Chạy `pip uninstall opencv-python opencv-contrib-python -y` rồi `pip install -r requirements.txt` lại |
| Nhận nhầm tay trái/phải | Đảm bảo `mirror = true`; hoặc đặt `control_hand` = `"Any"` |
| Phải mở tay rất rộng mới được 100% | Giảm `pinch_max_ratio` (ví dụ 1.2) |
