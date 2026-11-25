import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Đọc dữ liệu tự động theo thư mục file script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "Du_lieu_SV_Gop_Thong_tin_Tot_nghiep.xlsx")
df_raw = pd.read_excel(data_file, sheet_name="SV_Tong_hop_Kem_TN", header=3)

# Tính điểm Năm 1, Năm 2, Năm 3
df_raw["Điểm Năm 1"] = (
    pd.to_numeric(df_raw["HK1_22.23_ĐTBHT"], errors="coerce") + 
    pd.to_numeric(df_raw["HK2_22.23_ĐTBHT"], errors="coerce")
) / 2

df_raw["Điểm Năm 2"] = (
    pd.to_numeric(df_raw["HK1_23.24_ĐTBHT"], errors="coerce") + 
    pd.to_numeric(df_raw["HK2_23.24_ĐTBHT"], errors="coerce")
) / 2

df_raw["Điểm Năm 3"] = (
    pd.to_numeric(df_raw["HK1_24.25_ĐTBHT"], errors="coerce") + 
    pd.to_numeric(df_raw["HK2_24.25_ĐTBHT"], errors="coerce")
) / 2

# Lọc bỏ dòng thiếu điểm
df = df_raw.dropna(subset=["Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3"]).copy().reset_index(drop=True)

# Gán nhãn: Sớm vs Không sớm (Đúng hạn + Muộn)
df["Tốt nghiệp sớm"] = df["Tốt nghiệp sớm"].astype(str).str.strip()
df["Thực tế"] = df["Tốt nghiệp sớm"].apply(lambda x: "Sớm" if x == "Sớm" else "Không sớm")
y = (df["Thực tế"] == "Sớm").astype(int)

# 2. Chia cố định: 70% Train - 30% Test (random_state=42, stratify theo nhãn y)
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.30,
    random_state=42,
    stratify=y
)

df["Vai trò"] = "Train (Huấn luyện)"
df.loc[test_idx, "Vai trò"] = "Test (Kiểm thử)"
df["STT"] = range(1, len(df) + 1)

# 3. Huấn luyện mô hình ANN
features = ["Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3"]
X_train = df.loc[train_idx, features].values
y_train = y.loc[train_idx].values

X_test = df.loc[test_idx, features].values
y_test = y.loc[test_idx].values
X_all = df[features].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X_all)

ann_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.01,
    max_iter=1000,
    random_state=42
)
ann_model.fit(X_train_scaled, y_train)

# Đánh giá & tính xác suất
y_test_pred = ann_model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_test_pred) * 100
f1_test = f1_score(y_test, y_test_pred, average="macro") * 100

all_probs = ann_model.predict_proba(X_all_scaled)
df["XS Không sớm (%)"] = np.round(all_probs[:, 0] * 100, 2)
df["XS Sớm (%)"] = np.round(all_probs[:, 1] * 100, 2)

all_preds = ann_model.predict(X_all_scaled)
df["Dự đoán"] = np.where(all_preds == 1, "Sớm", "Không sớm")
df["Kết quả"] = np.where(df["Thực tế"] == df["Dự đoán"], "Đúng", "Lệch")

# 4. Xuất file Excel định dạng chuẩn
wb = openpyxl.Workbook()
wb.remove(wb.active)

header_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
header_font = Font(name="Calibri", size=11, bold=True, color="000000")
data_font = Font(name="Calibri", size=11, bold=False, color="000000")
bold_data_font = Font(name="Calibri", size=11, bold=True, color="000000")
title_font = Font(name="Calibri", size=15, bold=True, color="000000")
subtitle_font = Font(name="Calibri", size=11, bold=False, color="000000")

thin_side = Side(style="thin", color="000000")
thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
align_center = Alignment(horizontal="center", vertical="center")

# Sheet 1: Tong_Quan_Ket_Qua
ws1 = wb.create_sheet(title="Tong_Quan_Ket_Qua")
ws1.column_dimensions["A"].width = 4
for c in ["B", "C", "D", "E", "F", "G"]:
    ws1.column_dimensions[c].width = 22

ws1.cell(row=2, column=2, value="BÁO CÁO DỰ ĐOÁN TIẾN ĐỘ TỐT NGHIỆP SỚM (SỚM / KHÔNG SỚM)").font = title_font
ws1.cell(row=3, column=2, value="Mô hình: ANN (Multi-Layer Perceptron: 32-16) | Chia cố định: 70% Train - 30% Test").font = subtitle_font

h1 = ["Tổng số SV", "Tỷ lệ Train/Test", "Số SV Train (70%)", "Số SV Test (30%)", "Accuracy Test (%)", "Macro F1 (%)"]
for j, h in enumerate(h1, start=2):
    c = ws1.cell(row=5, column=j, value=h)
    c.fill, c.font, c.border, c.alignment = header_fill, header_font, thin_border, align_center

r1_vals = [len(df), "70% - 30%", len(train_idx), len(test_idx), f"{acc_test:.2f}%", f"{f1_test:.2f}%"]
for j, v in enumerate(r1_vals, start=2):
    c = ws1.cell(row=6, column=j, value=v)
    c.font, c.border, c.alignment = bold_data_font, thin_border, align_center

h2 = ["Tập Đánh Giá", "Số Lượng SV", "Dự Đoán Đúng", "Dự Đoán Lệch", "Độ Chính Xác (%)"]
for j, h in enumerate(h2, start=2):
    c = ws1.cell(row=9, column=j, value=h)
    c.fill, c.font, c.border, c.alignment = header_fill, header_font, thin_border, align_center

test_correct = (df.loc[test_idx, "Kết quả"] == "Đúng").sum()
test_wrong = (df.loc[test_idx, "Kết quả"] == "Lệch").sum()
r2_vals = ["Tập Kiểm thử (Test 30%)", len(test_idx), int(test_correct), int(test_wrong), f"{acc_test:.2f}%"]
for j, v in enumerate(r2_vals, start=2):
    c = ws1.cell(row=10, column=j, value=v)
    c.font, c.border, c.alignment = data_font, thin_border, align_center

# Sheet 2: Tap_Kiem_Thu_Test
ws2 = wb.create_sheet(title="Tap_Kiem_Thu_Test")
df_test = df.loc[test_idx].copy().reset_index(drop=True)
df_test["STT"] = range(1, len(df_test) + 1)

cols_test = ["STT", "Mã SV", "Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3", "Thực tế", "Dự đoán", "XS Không sớm (%)", "XS Sớm (%)", "Kết quả"]
headers_test = ["STT", "Mã SV", "Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3", "Thực tế", "Dự đoán", "XS Không sớm (%)", "XS Sớm (%)", "Kết quả"]

ws2.column_dimensions["A"].width = 4
for c_idx, h in enumerate(headers_test, start=2):
    ws2.column_dimensions[get_column_letter(c_idx)].width = 22 if "XS" in h else 16
    c = ws2.cell(row=5, column=c_idx, value=h)
    c.fill, c.font, c.border, c.alignment = header_fill, header_font, thin_border, align_center

for row_i, row_data in df_test[cols_test].iterrows():
    for col_j, val in enumerate(row_data, start=2):
        c = ws2.cell(row=6 + row_i, column=col_j, value=val)
        c.font, c.border, c.alignment = data_font, thin_border, align_center

# Sheet 3: Toan_Bo_Du_Lieu_Train_Test
ws3 = wb.create_sheet(title="Toan_Bo_Du_Lieu_Train_Test")
cols_all = ["STT", "Mã SV", "Vai trò", "Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3", "Thực tế", "Dự đoán", "XS Không sớm (%)", "XS Sớm (%)", "Kết quả"]
headers_all = ["STT", "Mã SV", "Vai trò", "Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3", "Thực tế", "Dự đoán", "XS Không sớm (%)", "XS Sớm (%)", "Kết quả"]

ws3.column_dimensions["A"].width = 4
for c_idx, h in enumerate(headers_all, start=2):
    ws3.column_dimensions[get_column_letter(c_idx)].width = 22 if "XS" in h else 18
    c = ws3.cell(row=5, column=c_idx, value=h)
    c.fill, c.font, c.border, c.alignment = header_fill, header_font, thin_border, align_center

for row_i, row_data in df[cols_all].iterrows():
    for col_j, val in enumerate(row_data, start=2):
        c = ws3.cell(row=6 + row_i, column=col_j, value=val)
        c.font, c.border, c.alignment = data_font, thin_border, align_center

output_path = os.path.join(current_dir, "Ket_Qua_Du_Doan_Tot_Nghiep_ANN_70Train_30Test.xlsx")
wb.save(output_path)
print(f"Đã lưu thành công tại: {output_path}")