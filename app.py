import streamlit as st
import pandas as pd
import plotly.express as px
import os
import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
from PIL import Image

st.set_page_config(
    page_title="SROIE OCR Dashboard",
    page_icon="🧾",
    layout="wide"
)

# Cache resource agar model tidak diload ulang setiap kali klik tombol (Biar Cepat)
@st.cache_resource
def load_model():
    # Menggunakan parameter yang sudah kita update sebelumnya
    return PaddleOCR(use_textline_orientation=True, lang='en')

ocr = load_model()

def clean_company_name(name):
    if pd.isna(name): return "UNKNOWN"
    name = str(name).upper()
    name = re.sub(r'^[0-9\W]+', '', name)
    remove_words = ["SDN BHD", "SDN. BHD.", "ENTERPRISE", "TRADING", "STORE", "RESTAURANT", "SHOP"]
    for word in remove_words:
        name = name.replace(word, "")
    return name.strip()

def normalize_text(s):
    s = str(s).replace('\ufffd', '').strip()
    return re.sub(r'\s+', ' ', s)

def poly_centroid(poly):
    cx = int(np.mean(poly[:,0]))
    cy = int(np.mean(poly[:,1]))
    return cx, cy

def group_rows(items, y_tol=15):
    items_sorted = sorted(items, key=lambda x: x['cy'])
    rows = []
    for it in items_sorted:
        placed = False
        for row in rows:
            if abs(it['cy'] - row['cy_mean']) <= y_tol:
                row['items'].append(it)
                row['cy_mean'] = sum(i['cy'] for i in row['items']) / len(row['items'])
                placed = True
                break
        if not placed:
            rows.append({'items':[it], 'cy_mean': it['cy']})
    for row in rows:
        row['items'] = sorted(row['items'], key=lambda x: x['xmin'])
    return rows

def row_to_string(row):
    full_text = " ".join([it['text'] for it in row['items']])
    return {'text': full_text, 'cy': row['cy_mean']}

def extract_info_heuristics(rows_merged):
    # Default values
    company = "UNKNOWN"
    date = None
    total = 0.0

    # 1. Extract Company (Baris teratas valid)
    for r in rows_merged[:4]:
        txt = r['text']
        if len(txt) > 3 and not re.match(r'^[\d\W]+$', txt):
            company = clean_company_name(txt)
            break
            
    # 2. Extract Date
    full_text = " ".join([r['text'] for r in rows_merged])
    date_pattern = r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
    match_date = re.search(date_pattern, full_text)
    if match_date:
        date = match_date.group()

    # 3. Extract Total
    # Cari dari bawah ke atas
    for r in reversed(rows_merged):
        txt_low = r['text'].lower()
        if "total" in txt_low and "sub" not in txt_low:
            # Cari angka desimal
            m = re.search(r"\d+\.\d{2}", r['text'])
            if m:
                total = float(m.group())
                break
                
    return company, date, total

def process_uploaded_image(image_path):
    # 1. Run OCR
    result = ocr.ocr(image_path)
    
    if result is None or len(result) == 0:
        return None

    res_obj = result[0]
    raw_items = []
    
    # --- LOGIKA ROBUST (SAMA SEPERTI YANG SUKSES DI NOTEBOOK) ---
    try:
        # KASUS A: Handle PaddleX / Dictionary Output
        # Kita cast ke dict biar aman, karena tipe aslinya mungkin 'OCRResult'
        if hasattr(res_obj, 'keys') or isinstance(res_obj, dict):
            res_dict = dict(res_obj)
            
            # Cari nama key yang benar (rec_texts vs rec_text)
            keys_text = ['rec_texts', 'rec_text', 'texts', 'text']
            keys_poly = ['dt_polys', 'rec_polys', 'polys', 'boxes']
            
            key_text = next((k for k in keys_text if k in res_dict), None)
            key_poly = next((k for k in keys_poly if k in res_dict), None)
            
            if key_text and key_poly:
                texts = res_dict[key_text]
                polys = res_dict[key_poly]
                
                # Zip text dan poly jadi satu
                for poly_raw, text in zip(polys, texts):
                    # Pastikan format poly benar (N, 2)
                    poly = np.array(poly_raw, dtype=np.float32).reshape(-1, 2)
                    cx, cy = poly_centroid(poly)
                    
                    raw_items.append({
                        'text': normalize_text(text),
                        'poly': poly,
                        'cx': cx, 'cy': cy,
                        'xmin': int(np.min(poly[:, 0])),
                        'xmax': int(np.max(poly[:, 0]))
                    })
            elif key_text:
                 # Fallback jika tidak ada poly (Hanya teks)
                 for text in res_dict[key_text]:
                     # Dummy coordinates biar program gak error
                     raw_items.append({
                         'text': normalize_text(text), 
                         'cx': 0, 'cy': 0, 'xmin': 0
                     })

        # KASUS B: Handle Standard List Output
        elif isinstance(res_obj, list):
            for line in res_obj:
                poly = np.array(line[0], dtype=np.float32).reshape(-1, 2)
                content = line[1]
                text = content[0] if isinstance(content, (list, tuple)) else str(content)
                cx, cy = poly_centroid(poly)
                raw_items.append({
                    'text': normalize_text(text),
                    'poly': poly,
                    'cx': cx, 'cy': cy,
                    'xmin': int(np.min(poly[:, 0])),
                    'xmax': int(np.max(poly[:, 0]))
                })
                
    except Exception as e:
        print(f"Error parsing: {e}")
        return None

    if not raw_items: return None

    # Grouping & Extraction
    # Cek apakah kita punya koordinat valid untuk grouping
    if 'cx' in raw_items[0] and raw_items[0]['cx'] != 0:
        rows = group_rows(raw_items, y_tol=12)
        rows_merged = [row_to_string(r) for r in rows]
    else:
        # Jika tidak ada koordinat, anggap 1 baris per item
        rows_merged = [{'text': it['text']} for it in raw_items]

    company, date, total = extract_info_heuristics(rows_merged)
    
    return {
        "Company": company,
        "Date": date,
        "Total": total,
        "Raw_Text": "\n".join([r.get('text', '') for r in rows_merged])
    }

st.title("🧾 Dashboard Analitik Struk Belanja")
st.markdown("Sistem OCR End-to-End untuk ekstraksi informasi struk otomatis.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Gambar")
    uploaded_file = st.file_uploader("Upload struk (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    st.header("2. Data Historis")
    # Cek apakah file CSV hasil batch processing ada
    csv_path = "hasil_ekstraksi_sroie.csv"
    if os.path.exists(csv_path):
        st.success(f"Database terhubung: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        st.warning("File CSV belum ditemukan. Jalankan notebook batch processing dulu.")
        df = pd.DataFrame()

# --- TABS UTAMA ---
tab1, tab2 = st.tabs(["🔍 Scanner OCR", "📊 Dashboard Analitik"])

# === TAB 1: SCANNER ===
with tab1:
    col1, col2 = st.columns(2)
    
    if uploaded_file:
        # Tampilkan Gambar
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Tombol Proses
            process_btn = st.button("Jalankan Ekstraksi Info", type="primary")

        # Hasil Ekstraksi
        with col2:
            st.subheader("Hasil Ekstraksi")
            if process_btn:
                with st.spinner("Sedang membaca teks..."):
                    # Simpan file sementara agar bisa dibaca PaddleOCR
                    temp_path = "temp_upload.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Jalankan Fungsi Pipeline
                    data = process_uploaded_image(temp_path)
                    
                    if data:
                        # Tampilkan Metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Total Belanja", f"RM {data['Total']:.2f}")
                        m2.metric("Tanggal", str(data['Date']))
                        
                        st.write(f"**Nama Toko:** {data['Company']}")
                        
                        with st.expander("Lihat Teks Mentah (Raw Output)"):
                            st.text(data['Raw_Text'])
                            
                        st.success("Ekstraksi Selesai!")
                    else:
                        st.error("Gagal mendeteksi teks. Coba gambar lain.")
    else:
        st.info("Silakan upload gambar struk di sidebar sebelah kiri untuk memulai.")

# === TAB 2: ANALITIK ===
with tab2:
    if not df.empty:
        # Cleaning Data on-the-fly untuk visualisasi
        df['Company_Clean'] = df['Company'].apply(clean_company_name)
        df['Date_Clean'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df_clean = df.dropna(subset=['Date_Clean', 'Total'])
        
        # KPI ROW
        total_spend = df_clean['Total'].sum()
        avg_spend = df_clean['Total'].mean()
        top_merch = df_clean['Company_Clean'].mode()[0]
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Pengeluaran Historis", f"RM {total_spend:,.2f}")
        k2.metric("Rata-rata Transaksi", f"RM {avg_spend:,.2f}")
        k3.metric("Toko Terfavorit", top_merch)
        
        st.divider()
        
        # CHARTS ROW
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Top 10 Pengeluaran per Merchant")
            top_spenders = df_clean.groupby('Company_Clean')['Total'].sum().sort_values(ascending=False).head(10).reset_index()
            fig_bar = px.bar(top_spenders, x='Total', y='Company_Clean', orientation='h', color='Total', color_continuous_scale='Viridis')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("Tren Pengeluaran Harian")
            daily = df_clean.groupby('Date_Clean')['Total'].sum().reset_index().sort_values('Date_Clean')
            fig_line = px.line(daily, x='Date_Clean', y='Total', markers=True, title="Timeline Belanja")
            st.plotly_chart(fig_line, use_container_width=True)
            
        st.subheader("Data Detail")
        st.dataframe(df_clean[['Company_Clean', 'Date', 'Total', 'Filename']], use_container_width=True)
        
    else:
        st.warning("Data historis tidak tersedia. Silakan proses dataset train di notebook dan simpan ke 'hasil_ekstraksi_sroie.csv'.")