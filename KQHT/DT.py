import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image as XLImage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

# Load data
filepath = 'SV_du_du_lieu_4_nam_Phan_Loai_Day_Du.xlsx'
if not os.path.exists(filepath):
    for f in os.listdir('.'):
        if f.endswith('.xlsx') and 'Phan_Loai_Day_Du' in f:
            filepath = f
            break

df = pd.read_excel(filepath, sheet_name='Tong_hop')

feature_cols = [
    'HK1_22.23_ĐTBHT', 'HK2_22.23_ĐTBHT',
    'HK1_23.24_ĐTBHT', 'HK2_23.24_ĐTBHT',
    'HK1_24.25_ĐTBHT', 'HK2_24.25_ĐTBHT'
]
target_score_col = 'HK1_25.26_ĐTBHT'
target_class_col = 'HK1_25.26_Xếp loại học lực'

def classify_gpa(gpa):
    if gpa >= 3.60:
        return 'Xuất sắc'
    elif gpa >= 3.20:
        return 'Giỏi'
    elif gpa >= 2.50:
        return 'Khá'
    elif gpa >= 2.00:
        return 'Trung bình'
    else:
        return 'Yếu'

train_df, test_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df[target_class_col]
)

X_train = train_df[feature_cols]
y_train = train_df[target_score_col]
X_test = test_df[feature_cols]
y_test = test_df[target_score_col]
y_test_cls = test_df[target_class_col]

# Tune max_depth = 10, keep sample_weight = 2.0 for 'Xuất sắc'
best_depth = 10
xs_weight = 2.0
sample_weights = np.where(train_df[target_class_col] == 'Xuất sắc', xs_weight, 1.0)

dt_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt_model.fit(X_train, y_train, sample_weight=sample_weights)

# Baseline DT (depth=None, weight=1.0)
dt_base = DecisionTreeRegressor(random_state=42)
dt_base.fit(X_train, y_train)

train_pred_score = np.round(dt_model.predict(X_train), 2)
train_pred_cls = [classify_gpa(s) for s in train_pred_score]

test_pred_score = np.round(dt_model.predict(X_test), 2)
test_pred_cls = [classify_gpa(s) for s in test_pred_score]

base_test_pred_score = np.round(dt_base.predict(X_test), 2)
base_test_pred_cls = [classify_gpa(s) for s in base_test_pred_score]

mae_test = mean_absolute_error(y_test, test_pred_score)
rmse_test = np.sqrt(mean_squared_error(y_test, test_pred_score))
r2_test = r2_score(y_test, test_pred_score)
acc_test = accuracy_score(y_test_cls, test_pred_cls)

mae_train = mean_absolute_error(y_train, train_pred_score)
rmse_train = np.sqrt(mean_squared_error(y_train, train_pred_score))
r2_train = r2_score(y_train, train_pred_score)
acc_train = accuracy_score(train_df[target_class_col], train_pred_cls)

base_mae_test = mean_absolute_error(y_test, base_test_pred_score)
base_rmse_test = np.sqrt(mean_squared_error(y_test, base_test_pred_score))
base_acc_test = accuracy_score(y_test_cls, base_test_pred_cls)

actual_depth = dt_model.get_depth()
n_leaves = dt_model.get_n_leaves()

print(f"DT (max_depth={best_depth}, actual_depth={actual_depth}, leaves={n_leaves}, xs_weight={xs_weight}):")
print(f"  Train: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R2={r2_train:.4f}, Acc={acc_train:.4%}")
print(f"  Test:  MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R2={r2_test:.4f}, Acc={acc_test:.4%}")

# Plot tree (limit display depth to 5 for readability in PNG while saving full rules in text)
plt.figure(figsize=(24, 12))
plot_tree(
    dt_model,
    max_depth=6,
    feature_names=['HK1_N1', 'HK2_N1', 'HK1_N2', 'HK2_N2', 'HK1_N3', 'HK2_N3'],
    filled=True,
    rounded=True,
    fontsize=8,
    precision=2
)
plt.title(f"Cây Quyết Định (max_depth={best_depth}, độ sâu thực tế={actual_depth}, weight_Xuất_sắc={xs_weight} - Hiển thị 4 tầng đầu)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_plot_d10.png', dpi=180)
plt.close()

tree_rules = export_text(dt_model, feature_names=feature_cols)

# Build Excel workbook v2
wb = openpyxl.Workbook()

font_title = Font(name='Calibri', size=14, bold=True, color='1F4E78')
font_sec = Font(name='Calibri', size=11, bold=True, color='1F4E78')
font_header = Font(name='Calibri', size=10, bold=True, color='FFFFFF')
font_data = Font(name='Calibri', size=10, color='000000')
font_bold = Font(name='Calibri', size=10, bold=True, color='000000')

fill_header = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
fill_sub = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
fill_zebra = PatternFill(start_color='F9FAFC', end_color='F9FAFC', fill_type='solid')

thin_side = Side(style='thin', color='D9D9D9')
med_bottom = Side(style='medium', color='1F4E78')
border_cell = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
border_header = Border(left=thin_side, right=thin_side, top=thin_side, bottom=med_bottom)

align_center = Alignment(horizontal='center', vertical='center', wrap_text=True)
align_left = Alignment(horizontal='left', vertical='center', wrap_text=True)
align_right = Alignment(horizontal='right', vertical='center')

# ==================== SHEET 1: Tong_Quan_Danh_Gia ====================
ws1 = wb.active
ws1.title = 'Tong_Quan_Danh_Gia'
ws1.views.sheetView[0].showGridLines = True

ws1['A1'] = 'BÁO CÁO DỰ ĐOÁN ĐIỂM & PHÂN LOẠI HỌC KỲ 1 NĂM 4 (DECISION TREE - MAX_DEPTH = 10)'
ws1['A1'].font = font_title
ws1.merge_cells('A1:H1')

ws1['A3'] = '1. Cấu hình mô hình & Chia tập dữ liệu'
ws1['A3'].font = font_sec

config_rows = [
    ['Tham số / Cấu hình', 'Giá trị áp dụng', 'Ghi chú'],
    ['Thuật toán', 'Decision Tree Regressor (CART)', 'Dự đoán điểm liên tục HK1_25.26_ĐTBHT rồi quy đổi sang 5 mức phân loại'],
    ['Biến đầu vào (X)', '6 kỳ điểm ĐTBHT (Năm 1, Năm 2, Năm 3)', 'HK1_22.23, HK2_22.23, HK1_23.24, HK2_23.24, HK1_24.25, HK2_24.25'],
    ['Biến mục tiêu (Y)', 'HK1_25.26_ĐTBHT -> Xếp loại học lực', 'Xuất sắc (>=3.60), Giỏi (3.20-3.59), Khá (2.50-3.19), TB (2.00-2.49), Yếu (<2.00)'],
    ['Tỷ lệ chia dữ liệu', '70% Train (267 SV) / 30% Test (115 SV)', 'Chia phân tầng (Stratified split, random_state=42) giữ nguyên tỷ lệ các lớp'],
    ['Điều chỉnh độ cao cây', f'max_depth = 10 (Độ sâu thực tế: {actual_depth}, Số lá: {n_leaves})', 'Cho phép cây phân nhánh chi tiết hơn theo từng ngưỡng điểm của 6 học kỳ'],
    ['Điều chỉnh mất cân bằng', 'sample_weight = 2.0 cho lớp Xuất sắc', 'Tăng gấp đôi trọng số lớp Xuất sắc trong tập Train để ưu tiên nhận diện nhóm Xuất sắc']
]

for r_idx, row in enumerate(config_rows, start=4):
    for c_idx, val in enumerate(row, start=1):
        cell = ws1.cell(row=r_idx, column=c_idx, value=val)
        if r_idx == 4:
            cell.font = font_header
            cell.fill = fill_header
            cell.border = border_header
            cell.alignment = align_center
        else:
            cell.font = font_bold if c_idx == 1 else font_data
            if r_idx % 2 == 1:
                cell.fill = fill_zebra
            cell.border = border_cell
            cell.alignment = align_left

# Section 2: Model Comparison
ws1['A12'] = '2. So sánh hiệu quả mô hình (Tập Test - 115 SV)'
ws1['A12'].font = font_sec

comp_rows = [
    ['Chỉ số đánh giá', 'DT Mặc định (Chưa chỉnh)', 'DT Đã điều chỉnh (Depth=10, XS_Weight=2.0)', 'Chênh lệch / Ghi chú'],
    ['Sai số tuyệt đối trung bình (MAE)', round(base_mae_test, 4), round(mae_test, 4), round(mae_test - base_mae_test, 4)],
    ['Sai số toàn phương trung bình (RMSE)', round(base_rmse_test, 4), round(rmse_test, 4), round(rmse_test - base_rmse_test, 4)],
    ['Độ chính xác phân loại (Accuracy - Test)', round(base_acc_test, 4), round(acc_test, 4), round(acc_test - base_acc_test, 4)],
    ['Độ chính xác phân loại (Accuracy - Train)', '-', round(acc_train, 4), 'Khớp rất cao trên tập huấn luyện'],
    ['Hệ số xác định R² (Test)', '-', round(r2_test, 4), '-']
]

for r_idx, row in enumerate(comp_rows, start=13):
    for c_idx, val in enumerate(row, start=1):
        cell = ws1.cell(row=r_idx, column=c_idx, value=val)
        if r_idx == 13:
            cell.font = font_header
            cell.fill = fill_header
            cell.border = border_header
            cell.alignment = align_center
        else:
            cell.font = font_bold if c_idx == 1 else font_data
            cell.border = border_cell
            cell.alignment = align_left if c_idx == 1 else align_center
            if isinstance(val, float):
                cell.number_format = '0.00%' if 'Accuracy' in row[0] else '0.0000'

# Section 3: Classification Report by Class (Test Set)
ws1['A20'] = '3. Chi tiết hiệu năng phân loại theo từng mức học lực trên Tập Test (115 SV)'
ws1['A20'].font = font_sec

classes = ['Xuất sắc', 'Giỏi', 'Khá', 'Trung bình', 'Yếu']
rep_dict = classification_report(y_test_cls, test_pred_cls, labels=classes, output_dict=True, zero_division=0)

rep_headers = ['Xếp loại học lực', 'Số lượng Thực tế (Test)', 'Số lượng Dự đoán (Test)', 'Precision', 'Recall', 'F1-Score']
for c_idx, h in enumerate(rep_headers, start=1):
    cell = ws1.cell(row=21, column=c_idx, value=h)
    cell.font = font_header
    cell.fill = fill_header
    cell.border = border_header
    cell.alignment = align_center

for idx, cls_name in enumerate(classes, start=22):
    actual_cnt = int((y_test_cls == cls_name).sum())
    pred_cnt = int((pd.Series(test_pred_cls) == cls_name).sum())
    prec = rep_dict[cls_name]['precision']
    rec = rep_dict[cls_name]['recall']
    f1 = rep_dict[cls_name]['f1-score']
    row_vals = [cls_name, actual_cnt, pred_cnt, round(prec, 4), round(rec, 4), round(f1, 4)]
    for c_idx, val in enumerate(row_vals, start=1):
        cell = ws1.cell(row=idx, column=c_idx, value=val)
        cell.font = font_bold if cls_name == 'Xuất sắc' else font_data
        if cls_name == 'Xuất sắc':
            cell.fill = fill_sub
        cell.border = border_cell
        cell.alignment = align_left if c_idx == 1 else align_center
        if c_idx >= 4:
            cell.number_format = '0.00%'

tot_row = ['Tổng / Trung bình (Macro)', '=SUM(B22:B26)', '=SUM(C22:C26)', '=AVERAGE(D22:D26)', '=AVERAGE(E22:E26)', '=AVERAGE(F22:F26)']
for c_idx, val in enumerate(tot_row, start=1):
    cell = ws1.cell(row=27, column=c_idx, value=val)
    cell.font = font_bold
    cell.fill = fill_zebra
    cell.border = border_header
    cell.alignment = align_left if c_idx == 1 else align_center
    if c_idx >= 4:
        cell.number_format = '0.00%'

chart = BarChart()
chart.type = "col"
chart.style = 10
chart.grouping = "clustered"
chart.title = "So sánh Số lượng Thực tế vs Dự đoán (Tập Test - Depth=10)"
chart.y_axis.title = "Số sinh viên"
chart.x_axis.title = "Xếp loại học lực"
data_ref = Reference(ws1, min_col=2, min_row=21, max_col=3, max_row=26)
cats_ref = Reference(ws1, min_col=1, min_row=22, max_row=26)
chart.add_data(data_ref, titles_from_data=True)
chart.set_categories(cats_ref)
chart.width = 16
chart.height = 10
ws1.add_chart(chart, "A29")

ws1['G12'] = '4. Mức độ quan trọng của từng kỳ (Feature Importance)'
ws1['G12'].font = font_sec
ws1['G13'] = 'Kỳ học (Biến đầu vào)'
ws1['H13'] = 'Trọng số quan trọng'
for col_let in ['G', 'H']:
    ws1[f'{col_let}13'].font = font_header
    ws1[f'{col_let}13'].fill = fill_header
    ws1[f'{col_let}13'].border = border_header
    ws1[f'{col_let}13'].alignment = align_center

for idx, (feat, imp) in enumerate(zip(feature_cols, dt_model.feature_importances_), start=14):
    c1 = ws1.cell(row=idx, column=7, value=feat)
    c2 = ws1.cell(row=idx, column=8, value=round(float(imp), 4))
    c1.font = font_data
    c2.font = font_data
    c1.border = border_cell
    c2.border = border_cell
    c1.alignment = align_left
    c2.alignment = align_center
    c2.number_format = '0.00%'

# ==================== SHEET 2: Ket_Qua_Test_30 ====================
ws2 = wb.create_sheet(title='Ket_Qua_Test_30')
ws2.views.sheetView[0].showGridLines = True

test_out = test_df[['Mã SV', 'HK1_22.23_LỚP'] + feature_cols + [target_score_col, target_class_col]].copy()
test_out['Điểm_Dự_Đoán_HK1_N4'] = test_pred_score
test_out['Sai_Số_Điểm'] = np.round(test_out['Điểm_Dự_Đoán_HK1_N4'] - test_out[target_score_col], 2)
test_out['Phân_Loại_Dự_Đoán'] = test_pred_cls
test_out['Kết_Quả_So_Khớp'] = np.where(test_out[target_class_col] == test_out['Phân_Loại_Dự_Đoán'], 'Đúng', 'Sai')
test_out = test_out.sort_values('Mã SV').reset_index(drop=True)

headers_s2 = [
    'Mã SV', 'Lớp',
    'HK1_N1 (22.23)', 'HK2_N1 (22.23)',
    'HK1_N2 (23.24)', 'HK2_N2 (23.24)',
    'HK1_N3 (24.25)', 'HK2_N3 (24.25)',
    'Điểm Thực Tế HK1_N4', 'Xếp Loại Thực Tế',
    'Điểm Dự Đoán HK1_N4', 'Sai Số (Dự đoán - Thực tế)',
    'Phân Loại Dự Đoán', 'Kết Quả Phân Loại'
]

for c_idx, h in enumerate(headers_s2, start=1):
    cell = ws2.cell(row=1, column=c_idx, value=h)
    cell.font = font_header
    cell.fill = fill_header
    cell.border = border_header
    cell.alignment = align_center

for r_idx, row in enumerate(test_out.itertuples(index=False), start=2):
    for c_idx, val in enumerate(row, start=1):
        cell = ws2.cell(row=r_idx, column=c_idx, value=val)
        cell.font = font_data
        cell.border = border_cell
        if c_idx in [1, 2, 10, 13, 14]:
            cell.alignment = align_center
        else:
            cell.alignment = align_right
            cell.number_format = '0.00'
        if c_idx == 14:
            if val == 'Đúng':
                cell.fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
                cell.font = Font(name='Calibri', size=10, bold=True, color='375623')
            else:
                cell.fill = PatternFill(start_color='FCE4D6', end_color='FCE4D6', fill_type='solid')
                cell.font = Font(name='Calibri', size=10, bold=True, color='C00000')

ws2.freeze_panes = 'A2'
ws2.auto_filter.ref = f"A1:N{len(test_out)+1}"

# ==================== SHEET 3: Ket_Qua_Train_70 ====================
ws3 = wb.create_sheet(title='Ket_Qua_Train_70')
ws3.views.sheetView[0].showGridLines = True

train_out = train_df[['Mã SV', 'HK1_22.23_LỚP'] + feature_cols + [target_score_col, target_class_col]].copy()
train_out['Điểm_Dự_Đoán_HK1_N4'] = train_pred_score
train_out['Sai_Số_Điểm'] = np.round(train_out['Điểm_Dự_Đoán_HK1_N4'] - train_out[target_score_col], 2)
train_out['Phân_Loại_Dự_Đoán'] = train_pred_cls
train_out['Kết_Quả_So_Khớp'] = np.where(train_out[target_class_col] == train_out['Phân_Loại_Dự_Đoán'], 'Đúng', 'Sai')
train_out = train_out.sort_values('Mã SV').reset_index(drop=True)

for c_idx, h in enumerate(headers_s2, start=1):
    cell = ws3.cell(row=1, column=c_idx, value=h)
    cell.font = font_header
    cell.fill = fill_header
    cell.border = border_header
    cell.alignment = align_center

for r_idx, row in enumerate(train_out.itertuples(index=False), start=2):
    for c_idx, val in enumerate(row, start=1):
        cell = ws3.cell(row=r_idx, column=c_idx, value=val)
        cell.font = font_data
        cell.border = border_cell
        if c_idx in [1, 2, 10, 13, 14]:
            cell.alignment = align_center
        else:
            cell.alignment = align_right
            cell.number_format = '0.00'
        if c_idx == 14:
            if val == 'Đúng':
                cell.fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
                cell.font = Font(name='Calibri', size=10, bold=True, color='375623')
            else:
                cell.fill = PatternFill(start_color='FCE4D6', end_color='FCE4D6', fill_type='solid')
                cell.font = Font(name='Calibri', size=10, bold=True, color='C00000')

ws3.freeze_panes = 'A2'
ws3.auto_filter.ref = f"A1:N{len(train_out)+1}"

# ==================== SHEET 4: Cay_Quyet_Dinh_Va_Luat ====================
ws4 = wb.create_sheet(title='Cay_Quyet_Dinh_Va_Luat')
ws4.views.sheetView[0].showGridLines = True

ws4['A1'] = f'CẤU TRÚC CÂY QUYẾT ĐỊNH (MAX_DEPTH = 10, ĐỘ SÂU THỰC TẾ = {actual_depth}, WEIGHT XUẤT SẮC = 2.0)'
ws4['A1'].font = font_title

ws4['A3'] = 'Tập luật đầy đủ (10 tầng) trích xuất từ cây quyết định:'
ws4['A3'].font = font_sec

for idx, line in enumerate(tree_rules.strip().split('\n'), start=4):
    c = ws4.cell(row=idx, column=1, value=line)
    c.font = Font(name='Consolas', size=9)

img = XLImage('decision_tree_plot_d10.png')
img.width = 1200
img.height = 600
ws4.add_image(img, 'D3')

for ws in [ws1, ws2, ws3]:
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.row == 1 and ws == ws1:
                continue
            val_str = str(cell.value or '')
            if not val_str.startswith('='):
                max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(max_len + 4, 14)

ws1.column_dimensions['A'].width = 36
ws1.column_dimensions['B'].width = 44
ws1.column_dimensions['C'].width = 52
ws1.column_dimensions['G'].width = 24
ws1.column_dimensions['H'].width = 22
ws4.column_dimensions['A'].width = 65

output_filename = 'Ket_Qua_Du_Doan_HK1_Nam4_DecisionTree-v2.xlsx'
wb.save(output_filename)
print("Saved successfully:", output_filename)
print("\nClassification report on Test:")
print(classification_report(y_test_cls, test_pred_cls, labels=classes, zero_division=0))