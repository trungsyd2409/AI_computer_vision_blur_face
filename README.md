# Pinch to Pixelate

Project computer vision: **khoảng cách giữa ngón cái và ngón trỏ điều khiển hiệu ứng trên tất cả khuôn mặt** trong camera.

- **Tay phải → PIXEL:** chụm 2 ngón = mặt rõ nét (0%), mở xa = mặt vỡ pixel (100%)
- **Tay trái → SWIRL:** chụm 2 ngón = không xoắn (0%), mở xa = mặt xoắn như xoáy nước (100%)
- Hiệu ứng PIXEL nằm **đè lên** SWIRL, các ô pixel có **noise** nhấp nháy màu
- Bỏ tay ra khỏi khung hình → mức của tay đó **được giữ nguyên** (trạng thái `LOCKED`)
- Camera mặc định 640x480, cửa sổ hiển thị rộng 640 px và luôn giữ đúng tỉ lệ ảnh
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
| Giơ **tay phải** (skeleton xanh ngọc), chụm ngón cái + ngón trỏ | PIXEL về 0% |
| Tay phải mở rộng 2 ngón | PIXEL tăng dần tới 100% |
| Giơ **tay trái** (skeleton tím), chụm 2 ngón | SWIRL về 0% |
| Tay trái mở rộng 2 ngón | Mặt xoắn dần, tối đa `Swirl max deg` độ ở tâm mặt |
| Bỏ tay ra khỏi khung hình | Mức của tay đó giữ nguyên (`LOCKED` màu cam) |

Góc dưới bên trái có 2 thanh: `PIXEL` và `SWIRL`. Hai tay dùng được cùng lúc. Mặt được **xoắn trước, rồi pixel hoá đè lên trên**.

Khoảng cách được **chia cho kích thước lòng bàn tay** (cổ tay → khớp ngón giữa), nên đứng gần hay xa camera thì cử chỉ vẫn cho cùng kết quả. Mức pixel được làm mượt (smoothing) để không bị giật.

### Phím tắt

| Phím | Chức năng |
|---|---|
| `X` | Mở/đóng **cửa sổ Settings** (camera + hiệu ứng) |
| `1` `2` `3` `4` | FPS mục tiêu 15 / 24 / 30 / 60 |
| `+` / `-` | FPS mục tiêu +5 / −5 |
| `P` | Mở bảng setting gốc của driver webcam (Windows) |
| `M` | Bật/tắt chế độ gương (mirror) |
| `O` | Bật/tắt skeleton bàn tay + đường nối 2 ngón |
| `R` | Reset PIXEL và SWIRL về 0% |
| `H` | Hiện/ẩn bảng phím tắt |
| `Q` / `Esc` | Thoát (tự lưu setting) |

### Cửa sổ Settings (phím `X`)

Nhấn `X` để mở một cửa sổ riêng có các thanh trượt. Nhấn `X` lần nữa hoặc bấm nút đóng cửa sổ để tắt.

| Thanh trượt | Ý nghĩa |
|---|---|
| `FPS` | FPS mục tiêu (5–60) |
| `Resolution` | 0 = 640x480, 1 = 800x600, 2 = 960x540, 3 = 1280x720, 4 = 1600x900, 5 = 1920x1080 |
| `Brightness` / `Contrast` / `Saturation` | Độ sáng / tương phản / bão hoà màu |
| `Gain` | Khuếch đại tín hiệu. Tăng Gain để ảnh sáng hơn mà không cần tăng exposure |
| `Auto exposure` | 1 = camera tự chỉnh phơi sáng, 0 = chỉnh tay |
| `Exposure -` | Thời gian phơi sáng dạng `-N` = 1/2^N giây. **Số càng lớn → ảnh tối hơn nhưng FPS cao hơn.** 5 = 1/32 s (đủ cho 30 FPS), 4 = 1/16 s (tối đa 16 FPS) |
| `Pinch 0% at` / `Pinch 100% at` | Tỉ lệ ngón/lòng bàn tay ứng với 0% và 100% (x100). Giảm `Pinch 100% at` nếu phải mở tay quá rộng |
| `Smooth ms` | Độ mượt (mili giây) |
| `Blocks at 0%` / `Blocks at 100%` | Số ô pixel ngang mặt ở mức thấp / cao nhất (ít ô = vỡ hơn) |
| `Pixel noise` | Độ nhiễu màu của các ô pixel (0 = tắt, 100 = rất nhiễu). Noise cũng tăng theo mức PIXEL |
| `Swirl max deg` | Góc xoắn ở tâm mặt khi SWIRL = 100% (0–1080 độ) |
| `Window width` | Chiều rộng cửa sổ camera (px). Chiều cao tự tính theo tỉ lệ ảnh |

- Các thông số ảnh và hiệu ứng áp dụng **ngay lập tức**.
- `FPS` và `Resolution` áp dụng sau khi bạn **thả thanh trượt khoảng 0.6 giây**, vì camera phải mở lại (hình sẽ khựng khoảng 1 giây). Nếu camera không nhận, app tự quay về giá trị cũ.
- Kéo `Exposure -` sẽ tự chuyển `Auto exposure` về 0 (chỉnh tay).
- Phần chữ phía trên các thanh trượt cho biết camera **thực sự** đang chạy ở độ phân giải và FPS nào, giá trị camera báo lại (`Camera reports`), và cảnh báo màu đỏ nếu exposure quá dài so với FPS mục tiêu.
- Mỗi webcam có khoảng giá trị khác nhau. Nếu kéo mà ảnh không đổi, xem dòng `Camera reports`: nếu số không đổi thì camera không hỗ trợ thông số đó hoặc đã chạm giới hạn.
- Mọi thay đổi được lưu vào `settings.json` khi thoát. Lần sau mở app, các thông số ảnh bạn đã chỉnh sẽ được áp dụng lại.

**Công thức cho phòng tối muốn giữ 30 FPS:** `Auto exposure` = 0 → `Exposure -` = 5 → tăng `Gain` và `Brightness` tới khi ảnh đủ sáng.

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
2. Nhấn **`X`** → đặt `Auto exposure` = 0, `Exposure -` = 5, tăng `Gain`. Hoặc nhấn **`P`** để mở bảng setting driver, tắt **Low Light Compensation** / **Auto Exposure** (tên tuỳ hãng), rồi chỉnh Exposure tay.
3. Giảm độ phân giải: sửa `settings.json` → `"width": 640, "height": 480`.
4. Đóng các app khác đang dùng camera (Zoom, Teams, OBS, Camera).
5. Cắm webcam vào cổng USB 3.0, không qua hub.

## 5. File `settings.json`

Được tạo tự động khi thoát app. Bạn có thể sửa bằng tay (khi app đang tắt):

| Key | Mặc định | Ý nghĩa |
|---|---|---|
| `camera_index` | 0 | Webcam số mấy |
| `backend` | `"auto"` | `auto` / `dshow` / `msmf` / `any` |
| `width`, `height` | 640, 480 | Độ phân giải camera yêu cầu |
| `target_fps` | 30 | FPS mục tiêu |
| `use_mjpg` | true | Dùng codec MJPG |
| `camera_props` | `{}` | Các thông số ảnh bạn đã chỉnh trong cửa sổ Settings. Xoá dòng trong này để trả về mặc định của camera |
| `mirror` | true | Lật ảnh như gương |
| `display_width` | 640 | Chiều rộng cửa sổ hiển thị (giữ tỉ lệ ảnh) |
| `control_hand` | `"Right"` | Tay điều khiển PIXEL: `Right` / `Left` / `Any` |
| `swirl_hand` | `"Left"` | Tay điều khiển SWIRL: `Right` / `Left` / `Any` |
| `pinch_min_ratio` | 0.25 | Tỉ lệ ngón/lòng bàn tay ứng với 0% |
| `pinch_max_ratio` | 1.5 | Tỉ lệ ứng với 100% (giảm xuống nếu phải mở tay quá rộng) |
| `smoothing_tau` | 0.08 | Độ mượt (giây). Lớn hơn = mượt hơn nhưng chậm hơn |
| `max_blocks` / `min_blocks` | 48 / 4 | Số ô pixel ngang mặt ở mức ~0% / 100% |
| `face_padding` | 0.25 | Nới rộng vùng mặt thêm 25% |
| `pixel_noise` | 35 | Độ nhiễu màu của ô pixel (0–100) |
| `max_swirl_deg` | 540 | Góc xoắn tối đa ở tâm mặt |
| `swirl_radius` | 1.0 | Kích thước vòng xoắn so với vùng mặt (lớn hơn = xoắn lan rộng hơn) |
| `face_hold_sec` | 0.35 | Giữ vùng mặt cũ nếu mất nhận diện tạm thời (tránh lộ mặt 1 frame) |
| `detect_width` | 640 | Ảnh được thu nhỏ về chiều rộng này trước khi đưa vào AI (nhỏ = nhanh) |
| `last_pixel_level` / `last_swirl_level` | 0.0 | Mức PIXEL / SWIRL lần trước (tự lưu) |

Xoá `settings.json` để về mặc định.

## 6. Cấu trúc code

```
main.py            Vòng lặp chính: đọc camera, gọi pipeline, vẽ HUD, xử lý phím
app/
  camera.py        ThreadedCamera: MJPG + DirectShow + luồng đọc riêng, đo FPS camera
  detectors.py     MediaPipe HandLandmarker + FaceDetector, tự tải model
  gesture.py       Tính tỉ lệ pinch, chọn tay phải, làm mượt, khoá mức khi mất tay
  effects.py       Hiệu ứng pixel hoá + xoắn (swirl) + giữ vùng mặt khi detection chập chờn
  hud.py           Vẽ FPS, thanh %, skeleton, bảng phím tắt
  pipeline.py      Ghép tất cả: 1 frame vào → 1 frame đã xử lý ra
  settings.py      Đọc / ghi settings.json
  settings_window.py  Cửa sổ Settings (phím X): thanh trượt camera + hiệu ứng
```

Luồng xử lý mỗi frame:

```
Camera (thread) → lật gương → thu nhỏ → MediaPipe (tay + mặt)
   → tay phải → mức PIXEL, tay trái → mức SWIRL (làm mượt / LOCKED)
   → xoắn rồi pixel hoá + noise đè lên từng mặt → thu nhỏ về display_width → vẽ HUD → hiển thị
```

## 7. Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `Cannot open camera/video source: 0` | Camera đang bị app khác dùng, hoặc thử `--camera 1` |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Không liên quan project này (project dùng Tasks API mới). Nếu gặp ở code khác: bản mediapipe mới đã bỏ `mp.solutions` |
| `cv2` lỗi lạ sau khi cài | Có thể bị cài cả `opencv-python` và `opencv-contrib-python`. Chạy `pip uninstall opencv-python opencv-contrib-python -y` rồi `pip install -r requirements.txt` lại |
| Nhận nhầm tay trái/phải | Đảm bảo `mirror = true`; hoặc đặt `control_hand` = `"Any"` |
| Phải mở tay rất rộng mới được 100% | Giảm `pinch_max_ratio` (ví dụ 1.2) |
