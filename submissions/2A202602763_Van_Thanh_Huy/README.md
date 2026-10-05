# DeepWeeds — Lab Day 2

Bài làm của Văn Thanh Huy, MSSV 2A202602763. Kết quả được nhập từ ZIP output Kaggle đã tải về ngày 05/10/2026; hash nguồn nằm trong `import_manifest.json`.

## Kết quả test fold 0

Đánh giá toàn bộ 3.507 ảnh, các seed 0, 1, 2. Các số dưới được tính lại bằng `eval.py` gốc từ file dự đoán, không chạy model trên test thêm lần nữa.

| Cấu hình | Accuracy mean ± std | Macro-F1 mean ± std | ECE mean ± std |
|---|---|---|---|
| T00 baseline | 0.974242 ± 0.004430 | 0.966933 ± 0.005723 | 0.015578 ± 0.003459 |
| F01 cuối | 0.975192 ± 0.002613 | 0.969585 ± 0.003142 | 0.006865 ± 0.002187 |

Baseline: ConvNeXt-Tiny, fine-tune, CE, augmentation cơ bản, 1-view. Chung kết: ConvNeXt-Tiny + CutMix + five-crop, temperature fit riêng trên validation mỗi seed. Huấn luyện 12 epoch, batch size 32. Cải thiện macro-F1 0.002652 nhỏ hơn std lớn nhất 0.005723; chưa kết luận cải thiện vượt nhiễu.

## Nội dung

- `results.xlsx`: bảng backbone, training, inference, chung kết, theo lớp và độ trễ.
- `report.md`: báo cáo tổng hợp từ kết quả chạy thật; cần bổ sung diễn giải cá nhân trước khi nộp chính thức.
- `curves/`, `eda/`: biểu đồ, phân bố lớp và ảnh mẫu/lỗi.
- `predictions/`: dự đoán val và đủ 6 file dự đoán test.
- `logs/runs/`: config, môi trường thư viện, history và kết quả từng seed; không chứa checkpoint.
- `eval_out/`, `grade.txt`: kết quả evaluator và điểm phần I tính được. `grade.txt` chưa chấm I4a/I4b/I5 do lần gọi CLI thiếu các đối số tương ứng; đây không phải điểm toàn bài.
- `code/`: evaluator gốc, script dùng trên Kaggle và notebook portable để import.

## Chạy lại

Import `code/deepweeds_kaggle.ipynb` vào Kaggle, bật GPU/Internet và gắn DeepWeeds có ảnh cùng các CSV fold 0 gốc. Chạy từ trên xuống. CSV train/val/test chỉ cần `Filename,Label`; `Species` là tùy chọn. Bản code nhập từ ZIP đã được sửa yêu cầu cột này, giữ nguyên toàn bộ dự đoán và dữ liệu chia tập.

Notebook đi kèm được tạo từ source local để tái lập, không phải bản export notebook đã thực thi trên Kaggle. Chưa có URL công khai của notebook hoặc link checkpoint trong các file tải về. Phiên bản thư viện, GPU và trọng số ghi trong `logs/runs/**/environment.json`.

Để tính lại metric mà không training lại, từ root repo chạy:

```bash
python eval.py score --pred "submissions/2A202602763_Van_Thanh_Huy/predictions/F01_seed*_test.csv" --tag F01 --out eval_out
python eval.py score --pred "submissions/2A202602763_Van_Thanh_Huy/predictions/T00_seed*_test.csv" --tag T00 --out eval_out
```

Các lệnh trên kiểm tra định dạng/xác suất và tính metric. Muốn đối chiếu tên ảnh/nhãn với CSV gốc, thêm `--test-csv <test_subset0.csv> --labels <labels.csv>` khi đã có dataset.
