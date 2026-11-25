import os
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, confusion_matrix
)
from sklearn.neural_network import MLPClassifier

# 1. Tự động lấy đường dẫn thư mục chứa file .py hiện tại và đọc dữ liệu gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
file_in = os.path.join(BASE_DIR, 'Du_lieu_SV_Gop_Thong_tin_Tot_nghiep_1.xlsx')
df_raw = pd.read_excel(file_in, sheet_name='SV_Tong_hop_Kem_TN', header=None)

# Dòng 4 trong Excel (index 3) là dòng tiêu đề cột chuẩn
df = df_raw.iloc[4:].copy()
df.columns = df_raw.iloc[3].values
df = df.reset_index(drop=True)

# 2. Tiền xử lý nhãn mục tiêu
df['Thực tế'] = df['Tốt nghiệp sớm'].apply(
    lambda x: 'Sớm' if str(x).strip() == 'Sớm' else 'Không sớm'
)
POS_LABEL = 'Sớm'
NEG_LABEL = 'Không sớm'

# 3. Các cột điểm trung bình học tập (ĐTBHT) & điểm rèn luyện (Tổng điểm_Khoa) 6 kỳ (Năm 1 - Năm 3)
dtb_6 = [
    'HK1_22.23_ĐTBHT', 'HK2_22.23_ĐTBHT',
    'HK1_23.24_ĐTBHT', 'HK2_23.24_ĐTBHT',
    'HK1_24.25_ĐTBHT', 'HK2_24.25_ĐTBHT'
]
drl_6 = [
    'HK1_22.23_Tổng điểm_Khoa', 'HK2_22.23_Tổng điểm_Khoa',
    'HK1_23.24_Tổng điểm_Khoa', 'HK2_23.24_Tổng điểm_Khoa',
    'HK1_24.25_Tổng điểm_Khoa', 'HK2_24.25_Tổng điểm_Khoa'
]

for col in dtb_6 + drl_6:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Tính trung bình theo năm & đặc trưng xu hướng tăng/giảm (KHÔNG dùng CĐR và TCTL)
df['Điểm Năm 1'] = df[['HK1_22.23_ĐTBHT', 'HK2_22.23_ĐTBHT']].mean(axis=1)
df['Điểm Năm 2'] = df[['HK1_23.24_ĐTBHT', 'HK2_23.24_ĐTBHT']].mean(axis=1)
df['Điểm Năm 3'] = df[['HK1_24.25_ĐTBHT', 'HK2_24.25_ĐTBHT']].mean(axis=1)

df['ĐRL Năm 1'] = df[['HK1_22.23_Tổng điểm_Khoa', 'HK2_22.23_Tổng điểm_Khoa']].mean(axis=1)
df['ĐRL Năm 2'] = df[['HK1_23.24_Tổng điểm_Khoa', 'HK2_23.24_Tổng điểm_Khoa']].mean(axis=1)
df['ĐRL Năm 3'] = df[['HK1_24.25_Tổng điểm_Khoa', 'HK2_24.25_Tổng điểm_Khoa']].mean(axis=1)

df['ĐTB_TB_3Nam'] = df[dtb_6].mean(axis=1)
df['ĐRL_TB_3Nam'] = df[drl_6].mean(axis=1)
df['Xu_Huong_ĐTB'] = df['Điểm Năm 3'] - df['Điểm Năm 1']
df['Xu_Huong_ĐRL'] = df['ĐRL Năm 3'] - df['ĐRL Năm 1']

# Lọc bỏ các dòng thiếu điểm 6 kỳ (còn lại 478 SV)
df_clean = df.dropna(subset=dtb_6 + drl_6).copy().reset_index(drop=True)
df_clean['Mã SV'] = df_clean['Mã SV'].astype(str).str.strip()

# Tập 16 đặc trưng thuần túy từ quá trình học tập & rèn luyện (không CĐR, không TCTL)
feature_cols = dtb_6 + drl_6 + ['ĐTB_TB_3Nam', 'ĐRL_TB_3Nam', 'Xu_Huong_ĐTB', 'Xu_Huong_ĐRL']

X = df_clean[feature_cols]
y = (df_clean['Thực tế'] == POS_LABEL).astype(int)

# 4. Chia dữ liệu cố định 70% Train (334 SV) - 30% Test (144 SV)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df_clean.index, test_size=0.3, random_state=42, stratify=y
)

# 5. Chuẩn hóa & Huấn luyện ANN chống overfitting (16-8, alpha=1.5, early_stopping=True)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_all_sc = scaler.transform(X)

model = MLPClassifier(
    hidden_layer_sizes=(16, 8), alpha=1.5, max_iter=1000,
    early_stopping=True, validation_fraction=0.15, random_state=42
)
model.fit(X_train_sc, y_train)

# 6. Dự đoán và tính xác suất
y_pred_train = model.predict(X_train_sc)
y_pred_test = model.predict(X_test_sc)
proba_test = model.predict_proba(X_test_sc)
y_pred_all = model.predict(X_all_sc)
proba_all = model.predict_proba(X_all_sc)

MODEL_DESC = 'Mô hình: ANN (MLP 16-8, L2 alpha=1.5, Early Stopping) | Chỉ dùng ĐTBHT & ĐRL (Không CĐR) | Chia: 70% Train - 30% Test'
file_out = 'ANN_Output_Som_KhongCDR.xlsx'

# 7. Đóng gói kết quả toàn bộ dữ liệu
df_clean['Vai trò'] = 'Train (Huấn luyện)'
df_clean.loc[indices_test, 'Vai trò'] = 'Test (Kiểm thử)'
df_clean['Dự đoán'] = np.where(y_pred_all == 1, POS_LABEL, NEG_LABEL)
df_clean['XS Không sớm (%)'] = np.round(proba_all[:, 0] * 100, 2)
df_clean['XS Sớm (%)'] = np.round(proba_all[:, 1] * 100, 2)
df_clean['Kết quả'] = np.where(df_clean['Thực tế'] == df_clean['Dự đoán'], 'Đúng', 'Lệch')

df_test = df_clean.loc[indices_test].copy().reset_index(drop=True)
df_test['STT'] = range(1, len(df_test) + 1)

df_all = df_clean.copy().reset_index(drop=True)
df_all['STT'] = range(1, len(df_all) + 1)

cols_export_test = [
    'STT', 'Mã SV', 'Điểm Năm 1', 'Điểm Năm 2', 'Điểm Năm 3',
    'ĐRL Năm 1', 'ĐRL Năm 2', 'ĐRL Năm 3',
    'Thực tế', 'Dự đoán', 'XS Không sớm (%)', 'XS Sớm (%)', 'Kết quả'
]

cols_export_all = [
    'STT', 'Mã SV', 'Vai trò', 'Điểm Năm 1', 'Điểm Năm 2', 'Điểm Năm 3',
    'ĐRL Năm 1', 'ĐRL Năm 2', 'ĐRL Năm 3',
    'Thực tế', 'Dự đoán', 'XS Không sớm (%)', 'XS Sớm (%)', 'Kết quả'
]

# 8. Xuất file Excel chuẩn Format 3 sheet
file_out_path = os.path.join(BASE_DIR, file_out)
wb = openpyxl.Workbook()
wb.remove(wb.active)

font_title = Font(name='Calibri', size=14, bold=True)
font_sub = Font(name='Calibri', size=11, italic=True)
font_hdr = Font(name='Calibri', size=11, bold=True)
font_reg = Font(name='Calibri', size=11)
fill_hdr = PatternFill(fill_type='solid', start_color='DCE6F1', end_color='DCE6F1')
thin_side = Side(style='thin', color='BFBFBF')
border_thin = Border(top=thin_side, bottom=thin_side, left=thin_side, right=thin_side)
align_center = Alignment(horizontal='center', vertical='center', wrap_text=True)

# --- SHEET 1: Tong_Quan_Ket_Qua ---
ws1 = wb.create_sheet(title='Tong_Quan_Ket_Qua')
ws1.append([])
ws1.append(['', 'BÁO CÁO DỰ ĐOÁN TIẾN ĐỘ TỐT NGHIỆP (SỚM / KHÔNG SỚM) - KHÔNG DÙNG CĐR'])
ws1.append(['', MODEL_DESC])
ws1.append([])
ws1['B2'].font = font_title
ws1['B3'].font = font_sub

acc_train = accuracy_score(y_train, y_pred_train)
acc = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test, average='macro')
auc = roc_auc_score(y_test, proba_test[:, 1])

ws1.append(['', 'Tổng số SV', 'Tỷ lệ Train/Test', 'Accuracy Train (%)', 'Accuracy Test (%)', 'Chênh lệch (Gap)', 'Macro F1 (%)', 'ROC-AUC (%)'])
ws1.append(['', len(df_clean), '70% - 30%', f"{acc_train*100:.2f}%", f"{acc*100:.2f}%", f"{(acc_train-acc)*100:+.2f}%", f"{f1*100:.2f}%", f"{auc*100:.2f}%"])
for col_i in range(2, 10):
    ws1.cell(5, col_i).font = font_hdr
    ws1.cell(5, col_i).fill = fill_hdr
    ws1.cell(5, col_i).alignment = align_center
    ws1.cell(5, col_i).border = border_thin
    ws1.cell(6, col_i).font = font_reg
    ws1.cell(6, col_i).alignment = align_center
    ws1.cell(6, col_i).border = border_thin

ws1.append([])
ws1.append([])

correct_test = int((df_test['Kết quả'] == 'Đúng').sum())
wrong_test = int((df_test['Kết quả'] == 'Lệch').sum())
ws1.append(['', 'Tập Đánh Giá', 'Số Lượng SV', 'Dự Đoán Đúng', 'Dự Đoán Lệch', 'Độ Chính Xác (%)'])
ws1.append(['', 'Tập Kiểm thử (Test 30%)', len(df_test), correct_test, wrong_test, f"{acc*100:.2f}%"])
for col_i in range(2, 8):
    ws1.cell(9, col_i).font = font_hdr
    ws1.cell(9, col_i).fill = fill_hdr
    ws1.cell(9, col_i).alignment = align_center
    ws1.cell(9, col_i).border = border_thin
    ws1.cell(10, col_i).font = font_reg
    ws1.cell(10, col_i).alignment = align_center
    ws1.cell(10, col_i).border = border_thin

for col_i in range(2, 10):
    ws1.column_dimensions[get_column_letter(col_i)].width = 20

# --- SHEET 2: Tap_Kiem_Thu_Test ---
ws2 = wb.create_sheet(title='Tap_Kiem_Thu_Test')
ws2.append([])
ws2.append([])
ws2.append([])
ws2.append([''] + cols_export_test)
for col_i in range(2, len(cols_export_test) + 2):
    ws2.cell(4, col_i).font = font_hdr
    ws2.cell(4, col_i).fill = fill_hdr
    ws2.cell(4, col_i).alignment = align_center
    ws2.cell(4, col_i).border = border_thin
    ws2.column_dimensions[get_column_letter(col_i)].width = 16

for r_idx, row in enumerate(df_test[cols_export_test].values, start=5):
    ws2.append([''] + list(row))
    for col_i in range(2, len(cols_export_test) + 2):
        ws2.cell(r_idx, col_i).font = font_reg
        ws2.cell(r_idx, col_i).alignment = align_center
        ws2.cell(r_idx, col_i).border = border_thin

# --- SHEET 3: Toan_Bo_Du_Lieu_Train_Test ---
ws3 = wb.create_sheet(title='Toan_Bo_Du_Lieu_Train_Test')
ws3.append([])
ws3.append([])
ws3.append([])
ws3.append([''] + cols_export_all)
for col_i in range(2, len(cols_export_all) + 2):
    ws3.cell(4, col_i).font = font_hdr
    ws3.cell(4, col_i).fill = fill_hdr
    ws3.cell(4, col_i).alignment = align_center
    ws3.cell(4, col_i).border = border_thin
    ws3.column_dimensions[get_column_letter(col_i)].width = 16

for r_idx, row in enumerate(df_all[cols_export_all].values, start=5):
    ws3.append([''] + list(row))
    for col_i in range(2, len(cols_export_all) + 2):
        ws3.cell(r_idx, col_i).font = font_reg
        ws3.cell(r_idx, col_i).alignment = align_center
        ws3.cell(r_idx, col_i).border = border_thin

wb.save(file_out_path)
print(f"=== XUẤT FILE {file_out} THÀNH CÔNG | Train={acc_train*100:.2f}%, Test={acc*100:.2f}%, Macro F1={f1*100:.2f}%, AUC={auc*100:.2f}% ===")
