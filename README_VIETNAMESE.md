# Hướng Dẫn Sử Dụng DocLayout-YOLO với Module Tinh Chỉnh (Refinement)

## 🎯 Tổng Quan

DocLayout-YOLO với module tinh chỉnh là một hệ thống phân tích bố cục tài liệu được cải tiến, sử dụng kết hợp thông tin thị giác và văn bản để đạt độ chính xác cao hơn. Hệ thống được tối ưu để chạy trên GPU RTX 2050 4GB với thời gian huấn luyện tối đa 3 giờ.

### Kiến Trúc Hệ Thống

```
📄 Ảnh tài liệu → YOLOv10 → Phát hiện vùng
                      ↓
📝 OCR → Trích xuất văn bản → Đặc trưng ngữ nghĩa
                      ↓
🧠 Module tinh chỉnh (MLP) → Dự đoán cải tiến
```

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt Môi Trường

```bash
# Tạo và kích hoạt môi trường conda
conda create -n dla python=3.10 -y
conda activate dla

# Cài đặt các thư viện cần thiết
pip install -e .
pip install easyocr huggingface_hub

# Kiểm tra cài đặt
python -c "from doclayout_yolo import YOLOv10Refined; print('✅ Cài đặt thành công!')"
```

### 2. Chuẩn Bị Dữ Liệu

```bash
# Tải và chuẩn bị dataset DocLayNet mini (cho training nhanh)
python prepare_dataset.py --data-dir ./layout_data

# Tạo subset nhỏ để test nhanh
python create_mini_subset.py
```

### 3. Huấn Luyện Nhanh (< 1 giờ)

#### Chỉ Baseline Model
```bash
# Huấn luyện model cơ bản (30 phút)
python train_fast.py --model n
```

#### Huấn Luyện Hai Giai Đoạn (Baseline + Refinement)
```bash
# Huấn luyện đầy đủ (60 phút)
python train_fast.py --model n --refinement

# Model lớn hơn (90 phút)
python train_fast.py --model m --refinement
```

### 4. Đánh Giá Kết Quả

```bash
# So sánh baseline vs refined model
python evaluate_fast.py \
    --base experiments/fast_train_n_*/base_model/weights/best.pt \
    --refined experiments/fast_train_n_*/refinement_model/weights/best.pt

# Đánh giá một model đơn lẻ
python evaluate_fast.py --model path/to/model.pt
```

## 📋 Các Kịch Bản Sử Dụng Chi Tiết

### Kịch Bản 1: Phát Triển và Thử Nghiệm Nhanh

**Mục tiêu**: Test ý tưởng mới trong ≤ 30 phút

```bash
# 1. Huấn luyện baseline nhanh (model nano, 25 epochs)
python train_fast.py --model n --hardware local_development

# 2. Đánh giá ngay
python evaluate_fast.py --model experiments/latest/base_model/weights/best.pt

# 3. Xem kết quả
cat experiments/latest/training_summary.yaml
```

**Thời gian**: ~30 phút  
**GPU**: ~2GB VRAM  
**Kết quả**: mAP50 dự kiến ~0.20-0.25

### Kịch Bản 2: Training Refinement Module Tối Ưu

**Mục tiêu**: Đạt kết quả tốt nhất trong 90 phút

```bash
# 1. Train baseline với model medium (50 epochs)
python train_fast.py --model m --hardware local_development

# 2. Train refinement với partial freezing (15 epochs)
python train_fast.py --model m --refinement --hardware local_development

# 3. Đánh giá so sánh chi tiết
python evaluate_fast.py \
    --base experiments/latest/base_model/weights/best.pt \
    --refined experiments/latest/refinement_model/weights/best.pt \
    --output results/comparison_$(date +%Y%m%d_%H%M)
```

**Thời gian**: ~90 phút  
**GPU**: ~3.5GB VRAM  
**Kết quả**: mAP50 dự kiến ~0.25-0.30 (baseline), cải thiện 1-3% với refinement

### Kịch Bản 3: Triển Khai Lên Cloud (Kaggle/Colab)

**Chuẩn bị file config**:
```bash
# Tạo config cho Kaggle
cp configs/hardware_configs.yaml kaggle_config.yaml
# Chỉnh sửa kaggle_config.yaml theo nhu cầu
```

**Kaggle Notebook**:
```python
# Cell 1: Setup
!git clone https://github.com/kythuatdulieu/DocLayout-YOLO.git
%cd DocLayout-YOLO
!pip install -e . easyocr huggingface_hub

# Cell 2: Quick training
!python train_fast.py --model m --refinement --hardware kaggle

# Cell 3: Evaluation
!python evaluate_fast.py \
    --base experiments/latest/base_model/weights/best.pt \
    --refined experiments/latest/refinement_model/weights/best.pt
```

## ⚙️ Cấu Hình Chi Tiết

### Cấu Hình Hardware

```yaml
# configs/hardware_configs.yaml
local_development:  # RTX 2050 4GB
  batch_size: 8
  image_size: 512
  base_epochs: 50
  refinement_epochs: 15
  mixed_precision: true
  
kaggle:  # GPU mạnh hơn
  batch_size: 16
  image_size: 1120
  base_epochs: 100
  refinement_epochs: 30
```

### Tùy Chỉnh Tham Số Training

```python
# train_fast.py với tham số tùy chỉnh
python train_fast.py \
    --model m \
    --refinement \
    --hardware local_development \
    --experiment-name "test_new_strategy" \
    --data doclaynet  # Sử dụng dataset đầy đủ
```

## 📊 Hiểu Kết Quả Đánh Giá

### Đọc Báo Cáo Evaluation

```yaml
# evaluation_results/evaluation_summary.yaml
models_evaluated: [baseline, refined]
summary_metrics:
  baseline:
    mAP50: 0.249      # Độ chính xác trên IoU=0.5
    fps: 45.2         # Tốc độ xử lý (frames/second)
    model_size_mb: 6.2 # Kích thước model
  refined:
    mAP50: 0.267      # Cải thiện +1.8%
    fps: 42.1         # Giảm tốc độ 7%
    model_size_mb: 6.8 # Tăng kích thước 0.6MB

improvement_analysis:
  mAP50_delta: +0.018  # Cải thiện 1.8%
  latency_increase_percent: +7.4%  # Tăng thời gian xử lý 7.4%
```

### Phân Tích Per-Class

```json
{
  "per_class": {
    "Table": {"ap50": 0.414},      // Tốt nhất
    "Text": {"ap50": 0.470},       // Rất tốt
    "Picture": {"ap50": 0.156},    // Cần cải thiện
    "Section-header": {"ap50": 0.137}
  }
}
```

### Tiêu Chí Đánh Giá Thành Công

- **✅ Tốt**: mAP50 improvement > 1% với speed impact < 15%
- **⚠️ Khá**: mAP50 improvement > 0.5% với speed impact < 20%  
- **❌ Cần cải thiện**: mAP50 improvement ≤ 0% hoặc speed impact > 25%

## 🔧 Xử Lý Sự Cố

### Lỗi Thường Gặp

1. **CUDA Out of Memory**
   ```bash
   # Giảm batch size
   python train_fast.py --model n --batch-size 4
   ```

2. **OCR không hoạt động**
   ```bash
   pip install easyocr
   # Hoặc disable OCR tạm thời
   export DISABLE_OCR=1
   ```

3. **Dataset không tìm thấy**
   ```bash
   python prepare_dataset.py --data-dir ./layout_data
   ```

### Tối Ưu Hiệu Suất

1. **Tăng tốc độ training**: Giảm epochs, tăng learning rate
2. **Tiết kiệm VRAM**: Giảm batch size, image size
3. **Cải thiện accuracy**: Tăng epochs, sử dụng model lớn hơn

## 📚 Mở Rộng và Tùy Chỉnh

### Thêm Dataset Mới

```python
# 1. Tạo file YAML cho dataset mới
# mydataset.yaml
path: /path/to/mydataset
train: images/train
val: images/val
nc: 11  # số lượng classes
names: ['Caption', 'Footnote', ...]

# 2. Training với dataset mới
python train_fast.py --data mydataset.yaml --model m
```

### Tùy Chỉnh Text Features

```python
# doclayout_yolo/nn/modules/ocr_utils.py
class CustomTextFeatureExtractor(TextFeatureExtractor):
    def extract_features(self, text, bbox, image_shape):
        # Thêm features tùy chỉnh
        custom_features = self.extract_custom_features(text)
        base_features = super().extract_features(text, bbox, image_shape)
        return np.concatenate([base_features, custom_features])
```

### Tích Hợp với Pipeline Khác

```python
from doclayout_yolo import YOLOv10Refined

# Load model đã train
model = YOLOv10Refined("path/to/refined/model.pt")

# Sử dụng trong pipeline
def process_document(image_path):
    results = model.predict_with_refinement(image_path)
    return results
```

## 🎯 Kết Luận

Hệ thống DocLayout-YOLO với module tinh chỉnh cung cấp:

- **Accuracy**: Cải thiện 1-3% mAP50 so với baseline
- **Speed**: < 15% tăng thời gian xử lý
- **Memory**: Tương thích với GPU 4GB
- **Scalability**: Dễ dàng triển khai lên cloud

**Quy trình khuyến nghị**:
1. Bắt đầu với `train_fast.py --model n` để test
2. Nếu kết quả tốt, chuyển sang `--model m --refinement`  
3. Đánh giá với `evaluate_fast.py`
4. Triển khai production

**Lưu ý quan trọng**: Luôn sử dụng `conda activate dla` trước khi chạy bất kỳ script nào!