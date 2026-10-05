# DeepWeeds trên Kaggle

Import `deepweeds_kaggle.ipynb` qua chức năng Import Notebook của Kaggle. Notebook chứa sẵn tất cả script và bản `eval.py` gốc; không cần clone repo.

Bật GPU và Internet trong Settings. Add Input dataset DeepWeeds có ảnh và các file CSV gốc: `labels.csv`, `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`. Notebook tự tìm đường dẫn; đặt `IMAGES_DIR`/`LABELS_DIR` nếu có nhiều dataset phù hợp. Đừng dùng một dataset chỉ có ảnh/nhãn mà thiếu split gốc.

Chạy từ trên xuống với `EPOCHS=12`, `BATCH_SIZE=32`. Notebook thực hiện EDA, kiểm tra pipeline, 5 backbone, 3 trục training + kết hợp, 5 phương pháp inference ngoài baseline, baseline/final 3 seed và xuất kết quả. GPU/T4 kép hiện dùng một GPU; chưa dùng distributed training.

Để kiểm tra nhanh, đặt epoch=1 và RUN_FINAL=False, dùng output riêng. Đây không phải kết quả đủ chất lượng để nộp. Đo thời gian epoch đầu để ước lượng ngân sách; toàn bài cần nhiều giờ, có thể chia phiên.

Checkpoint và config lưu mỗi epoch. Trong cùng phiên, chạy lại ô để resume. Với phiên mới, Save Version trước khi đóng; Add Input output phiên trước, đặt `RESTORE_FROM` đến thư mục `deepweeds_lab`, giữ nguyên cấu hình và đường dẫn output. Script từ chối reuse khi hyperparameter khác. Nếu lỗi OOM, chọn batch nhỏ hơn và thư mục output mới cho toàn bộ so sánh.

`RUN_SCREEN=False` cho phiên chỉ chạy chung kết hoặc xuất kết quả đã có selection. `RUN_FINAL=False` dừng trước test. Sau khi `final_lock.json` tồn tại, không sàng lại. Không tự xóa dấu `test_started.json` để chạy lại test bị ngắt: giữ bằng chứng, giải thích trong báo cáo.

Kết quả ở `/kaggle/working/deepweeds_lab`: `results.xlsx`, `report.md`, `curves/`, `predictions/`, `eda/`, `runs/`, `eval_out/`, `grade.txt`, `deepweeds_submission.zip`. ZIP chứa code, log nhỏ, biểu đồ, dự đoán và báo cáo; không chứa dataset/checkpoint. Điền MSSV/họ tên, link notebook, nhận xét về ảnh lỗi và giả thuyết trước khi nộp. Report là bản tổng hợp tự động, không thay thế phần phân tích của bạn.

Các module riêng ở `code/`. Chạy local từ root repo:

```bash
python -m pip install -r kaggle/requirements.txt
python -m kaggle.code.suite --stage prepare --images-dir data/images --labels-dir data/labels --output lab_output
python -m kaggle.code.suite --stage screen --images-dir data/images --labels-dir data/labels --output lab_output
python -m kaggle.code.suite --stage final --images-dir data/images --labels-dir data/labels --output lab_output
python -m kaggle.code.suite --stage export --images-dir data/images --labels-dir data/labels --output lab_output
```

Rebuild notebook sau khi sửa source: `python kaggle/build_notebook.py`.

Giới hạn: đo GPU ở máy Kaggle thực tế, không có số liệu dựng sẵn. GMAC dùng fvcore (1 multiply-add = 1), in operator chưa hỗ trợ. Tiền xử lý decode/resize/normalize không nằm trong độ trễ; view TTA/aggregation có. Sàng lọc một seed, final ít nhất ba seed; độ phân giải test cố định theo cấu hình train để hỗ trợ DeiT. Chỉ hỗ trợ gộp BN an toàn cho Sequential và timm ResNet; API ensemble yêu cầu tự đảm bảo cùng thứ tự ảnh. Không sửa code đề trong starter/ hoặc eval.py.
