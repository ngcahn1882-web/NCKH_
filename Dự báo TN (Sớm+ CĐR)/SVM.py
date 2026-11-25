import os
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Đọc dữ liệu tự động theo thư mục chứa file script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "Du_lieu_SV_Gop_Thong_tin_Tot_nghiep.xlsx")
df_raw = pd.read_excel(data_file, sheet_name="SV_Tong_hop_Kem_TN", header=3)

# Tính điểm Năm 1, Năm 2, Năm 3
df_raw["Điểm Năm 1"] = (
    pd.to_numeric(df_raw["HK1_22.23_ĐTBHT"], errors="coerce")
    + pd.to_numeric(df_raw["HK2_22.23_ĐTBHT"], errors="coerce")
) / 2

df_raw["Điểm Năm 2"] = (
    pd.to_numeric(df_raw["HK1_23.24_ĐTBHT"], errors="coerce")
    + pd.to_numeric(df_raw["HK2_23.24_ĐTBHT"], errors="coerce")
) / 2

df_raw["Điểm Năm 3"] = (
    pd.to_numeric(df_raw["HK1_24.25_ĐTBHT"], errors="coerce")
    + pd.to_numeric(df_raw["HK2_24.25_ĐTBHT"], errors="coerce")
) / 2

df = (
    df_raw.dropna(subset=["Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3"])
    .copy()
    .reset_index(drop=True)
)

# Gán nhãn: Đúng hạn + Muộn = Không sớm | Sớm = Sớm
df["Tốt nghiệp sớm"] = df["Tốt nghiệp sớm"].astype(str).str.strip()
df["Thực tế"] = df["Tốt nghiệp sớm"].apply(
    lambda x: "Sớm" if x == "Sớm" else "Không sớm"
)
y = (df["Thực tế"] == "Sớm").astype(int)

# 2. Chia 10 nhóm Stratified cố định giống hệt thuật toán ANN (random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
group_assignments = np.zeros(len(df), dtype=int)
for fold_no, (_, test_indices) in enumerate(skf.split(df, y), start=1):
  group_assignments[test_indices] = fold_no

df["Nhóm"] = [f"Nhóm {g}" for g in group_assignments]
test_groups = ["Nhóm 1", "Nhóm 2", "Nhóm 3"]
df["Vai trò"] = df["Nhóm"].apply(
    lambda g: "Test (Kiểm thử)" if g in test_groups else "Train (Huấn luyện)"
)
df["STT"] = range(1, len(df) + 1)

train_mask = df["Vai trò"] == "Train (Huấn luyện)"
test_mask = df["Vai trò"] == "Test (Kiểm thử)"

features = ["Điểm Năm 1", "Điểm Năm 2", "Điểm Năm 3"]
X_train = df.loc[train_mask, features].values
y_train = y.loc[train_mask].values

X_test = df.loc[test_mask, features].values
y_test = y.loc[test_mask].values
X_all = df[features].values

# 3. Chuẩn hóa Z-Score và Huấn luyện mô hình SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X_all)

svm_model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Dự đoán xác suất & nhãn
all_probs = svm_model.predict_proba(X_all_scaled)
df["XS Không sớm (%)"] = np.round(all_probs[:, 0] * 100, 2)
df["XS Sớm (%)"] = np.round(all_probs[:, 1] * 100, 2)

all_preds = svm_model.predict(X_all_scaled)
df["Dự đoán"] = np.where(all_preds == 1, "Sớm", "Không sớm")
df["Kết quả"] = np.where(df["Thực tế"] == df["Dự đoán"], "Đúng", "Lệch")

y_test_pred = svm_model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_test_pred) * 100
macro_f1 = f1_score(y_test, y_test_pred, average="macro") * 100

group_stats = []
for grp in test_groups:
  sub = df[df["Nhóm"] == grp]
  total_g = len(sub)
  correct_g = (sub["Thực tế"] == sub["Dự đoán"]).sum()
  wrong_g = total_g - correct_g
  acc_g = (correct_g / total_g) * 100 if total_g > 0 else 0
  group_stats.append((f"{grp} (Test)", total_g, correct_g, wrong_g, f"{acc_g:.2f}%"))

# 4. Xuất file Excel chuẩn mẫu 3 sheet
wb = openpyxl.Workbook()
wb.remove(wb.active)

header_fill = PatternFill(
    start_color="DCE6F1", end_color="DCE6F1", fill_type="solid"
)
header_font = Font(name="Calibri", size=11, bold=True, color="000000")
data_font = Font(name="Calibri", size=11, bold=False, color="000000")
bold_data_font = Font(name="Calibri", size=11, bold=True, color="000000")
title_font = Font(name="Calibri", size=15, bold=True, color="000000")
subtitle_font = Font(name="Calibri", size=11, bold=False, color="000000")

thin_side = Side(style="thin", color="000000")
thin_border = Border(
    left=thin_side, right=thin_side, top=thin_side, bottom=thin_side
)
align_center = Alignment(horizontal="center", vertical="center")

# Sheet 1: Tong_Quan_Ket_Qua
ws1 = wb.create_sheet(title="Tong_Quan_Ket_Qua")
ws1.column_dimensions["A"].width = 4
for c in ["B", "C", "D", "E", "F", "G"]:
  ws1.column_dimensions[c].width = 22

ws1.cell(
    row=2,
    column=2,
    value="BÁO CÁO DỰ ĐOÁN TIẾN ĐỘ TỐT NGHIỆP SỚM (SỚM / KHÔNG SỚM)",
).font = title_font
ws1.cell(
    row=3,
    column=2,
    value=(
        "Mô hình: SVM (Support Vector Machine: kernel=rbf, C=1.0) | 10 Folds: 7"
        " Train - 3 Test"
    ),
).font = subtitle_font

h1 = [
    "Tổng số SV",
    "Số nhóm Stratified",
    "Số SV Train",
    "Số SV Test",
    "Accuracy Test (%)",
    "Macro F1 (%)",
]
for j, h in enumerate(h1, start=2):
  c = ws1.cell(row=5, column=j, value=h)
  c.fill, c.font, c.border, c.alignment = (
      header_fill,
      header_font,
      thin_border,
      align_center,
  )

r1_vals = [
    len(df),
    10,
    int(train_mask.sum()),
    int(test_mask.sum()),
    f"{acc_test:.2f}%",
    f"{macro_f1:.2f}%",
]
for j, v in enumerate(r1_vals, start=2):
  c = ws1.cell(row=6, column=j, value=v)
  c.font, c.border, c.alignment = bold_data_font, thin_border, align_center

h2 = [
    "Nhóm Đánh Giá",
    "Số Lượng SV",
    "Dự Đoán Đúng",
    "Dự Đoán Lệch",
    "Độ Chính Xác (%)",
]
for j, h in enumerate(h2, start=2):
  c = ws1.cell(row=9, column=j, value=h)
  c.fill, c.font, c.border, c.alignment = (
      header_fill,
      header_font,
      thin_border,
      align_center,
  )

for i, g in enumerate(group_stats, start=10):
  for j, val in enumerate(g, start=2):
    c = ws1.cell(row=i, column=j, value=val)
    c.font, c.border, c.alignment = data_font, thin_border, align_center

# Sheet 2: Du_Lieu_3_Nhom_Test
ws2 = wb.create_sheet(title="Du_Lieu_3_Nhom_Test")
df_test = df[test_mask].copy().reset_index(drop=True)
df_test["STT"] = range(1, len(df_test) + 1)

cols_test = [
    "STT",
    "Mã SV",
    "Nhóm",
    "Điểm Năm 1",
    "Điểm Năm 2",
    "Điểm Năm 3",
    "Thực tế",
    "Dự đoán",
    "XS Không sớm (%)",
    "XS Sớm (%)",
    "Kết quả",
]
headers_test = [
    "STT",
    "Mã SV",
    "Nhóm",
    "Điểm Năm 1",
    "Điểm Năm 2",
    "Điểm Năm 3",
    "Thực tế",
    "Dự đoán",
    "XS Không sớm (%)",
    "XS Sớm (%)",
    "Kết quả",
]

ws2.column_dimensions["A"].width = 4
for c_idx, h in enumerate(headers_test, start=2):
  ws2.column_dimensions[get_column_letter(c_idx)].width = (
      20 if "XS" in h else 15
  )
  c = ws2.cell(row=5, column=c_idx, value=h)
  c.fill, c.font, c.border, c.alignment = (
      header_fill,
      header_font,
      thin_border,
      align_center,
  )

for row_i, row_data in df_test[cols_test].iterrows():
  for col_j, val in enumerate(row_data, start=2):
    c = ws2.cell(row=6 + row_i, column=col_j, value=val)
    c.font, c.border, c.alignment = data_font, thin_border, align_center

# Sheet 3: Toan_Bo_10_Nhom
ws3 = wb.create_sheet(title="Toan_Bo_10_Nhom")
cols_all = [
    "STT",
    "Mã SV",
    "Nhóm",
    "Vai trò",
    "Điểm Năm 1",
    "Điểm Năm 2",
    "Điểm Năm 3",
    "Thực tế",
    "Dự đoán",
    "XS Không sớm (%)",
    "XS Sớm (%)",
    "Kết quả",
]
headers_all = [
    "STT",
    "Mã SV",
    "Nhóm",
    "Vai trò",
    "Điểm Năm 1",
    "Điểm Năm 2",
    "Điểm Năm 3",
    "Thực tế",
    "Dự đoán",
    "XS Không sớm (%)",
    "XS Sớm (%)",
    "Kết quả",
]

ws3.column_dimensions["A"].width = 4
for c_idx, h in enumerate(headers_all, start=2):
  ws3.column_dimensions[get_column_letter(c_idx)].width = (
      20 if "XS" in h else 16
  )
  c = ws3.cell(row=5, column=c_idx, value=h)
  c.fill, c.font, c.border, c.alignment = (
      header_fill,
      header_font,
      thin_border,
      align_center,
  )

for row_i, row_data in df[cols_all].iterrows():
  for col_j, val in enumerate(row_data, start=2):
    c = ws3.cell(row=6 + row_i, column=col_j, value=val)
    c.font, c.border, c.alignment = data_font, thin_border, align_center

output_file = os.path.join(
    current_dir, "Ket_Qua_Du_Doan_Tien_Do_Tot_Nghiep_SVM.xlsx"
)
wb.save(output_file)
print(f"Đã lưu kết quả tại: {output_file}")