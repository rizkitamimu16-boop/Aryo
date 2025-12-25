from flask import Flask, render_template, request, session, send_file, flash, redirect, url_for
import pandas as pd
import os
import math
import random
import locale
from werkzeug.utils import secure_filename
from kneed import KneeLocator
import io

# --- Konfigurasi Global ---
last_result_df = None

# --- Konfigurasi Locale ---
try:
    locale.setlocale(locale.LC_NUMERIC, "id_ID.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_NUMERIC, "Indonesian_Indonesia.1252")
    except locale.Error:
        print("Warning: Locale for Rupiah formatting could not be set.")

app = Flask(__name__)
app.secret_key = 'random_secret_key_1234567890'
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.jinja_env.globals.update(enumerate=enumerate)

# --- Filter Jinja Custom ---
def format_rupiah(value):
    try:
        return locale.format_string("%d", int(round(float(value))), grouping=True)
    except Exception:
        return str(value)

app.jinja_env.filters["format_rupiah"] = format_rupiah

# --- Fungsi Utilitas ---
def euclidean(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def min_max_scaler(df, cols):
    df_scaled = df.copy()
    min_vals = df_scaled[cols].min()
    max_vals = df_scaled[cols].max()
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1 
    df_scaled[cols] = (df_scaled[cols] - min_vals) / range_vals
    return df_scaled, min_vals, max_vals

def read_csv_flexible(filepath: str) -> pd.DataFrame:
    read_attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1"},
    ]
    for opts in read_attempts:
        try:
            return pd.read_csv(filepath, **opts)
        except Exception:
            continue
    raise ValueError("Format CSV tidak dikenali atau file rusak.")

def clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^\d,\.]", "", regex=True)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

# --- Logika Penentuan K Optimal ---
def find_optimal_k(wcss_list, k_values):
    kn = KneeLocator(k_values, wcss_list, curve="convex", direction="decreasing")
    if kn.knee is not None:
        return int(kn.knee)
    diffs = []
    for i in range(1, len(wcss_list) - 1):
        diff = (wcss_list[i - 1] - wcss_list[i]) - (wcss_list[i] - wcss_list[i + 1])
        diffs.append(diff)
    if diffs:
        return diffs.index(max(diffs)) + 2
    return max(2, int(len(k_values) ** 0.5))

# --- Fungsi K-Means Internal ---
def run_kmeans_internal(data, k, max_iter=100):
    if len(data) < k:
        return None, None, []
    random.seed(42)
    centroids = random.sample(data, k)
    centroid_history = [centroids.copy()]
    for _ in range(max_iter):
        clusters = {i: [] for i in range(k)}
        for point in data:
            distances = [euclidean(point, c) for c in centroids]
            clusters[distances.index(min(distances))].append(point)
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_c = [sum(p[j] for p in clusters[i]) / len(clusters[i]) for j in range(len(data[0]))]
                new_centroids.append(new_c)
            else:
                new_centroids.append(centroids[i])
        centroid_history.append(new_centroids.copy())
        if new_centroids == centroids:
            break
        centroids = new_centroids
    wcss = 0
    for i in range(k):
        for point in clusters[i]:
            wcss += sum((a - b) ** 2 for a, b in zip(point, centroids[i]))
    return wcss, centroids, centroid_history

# --- Route Utama ---
@app.route("/", methods=["GET", "POST"])
def index():
    global last_result_df
    error = None
    elbow_data = []
    optimal_k = session.get('optimal_k')
    data_preview = None
    result_data = None
    cluster_summary = None
    centroid_process = None
    used_k = None
    products_by_cluster = {}

    if 'result_data' in session:
        result_data = pd.DataFrame(session['result_data'])
        cluster_summary = result_data.groupby("cluster").size()
        used_k = len(cluster_summary)
        products_by_cluster = {i: result_data[result_data["cluster"] == i] for i in range(used_k)}
        centroid_process = session.get('centroid_process')

    if request.method == "POST":
        action = request.form.get("action")
        file = request.files.get("dataset")

        try:
            if not file or file.filename == "":
                raise ValueError("File dataset wajib diunggah")

            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(filepath)

            df = read_csv_flexible(filepath)
            df.columns = df.columns.str.lower().str.replace(" ", "_")
            df["harga_satuan"] = clean_numeric_series(df["harga_satuan"])
            df["jumlah_terjual"] = clean_numeric_series(df["jumlah_terjual"])
            df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
            df.dropna(inplace=True)
            df["pendapatan"] = df["harga_satuan"] * df["jumlah_terjual"]

            aggregated = df.groupby("produk", as_index=False).agg(
                penjualan_total=("jumlah_terjual", "sum"),
                pendapatan_total=("pendapatan", "sum"),
                frekuensi=("tanggal", "nunique")
            )

            data_preview = aggregated.head(10)
            cluster_cols = ["penjualan_total", "pendapatan_total", "frekuensi"]
            df_scaled, _, _ = min_max_scaler(aggregated, cluster_cols)
            data_to_cluster = df_scaled[cluster_cols].values.tolist()

            wcss_list = []
            k_range = range(2, min(11, len(data_to_cluster)))

            for k in k_range:
                wcss, _, _ = run_kmeans_internal(data_to_cluster, k)
                wcss_list.append(wcss)
                elbow_data.append({"k": k, "wcss": wcss})

            optimal_k = find_optimal_k(wcss_list, list(k_range))
            session['optimal_k'] = optimal_k

          # 🔹 TAHAP 2 — JALANKAN CLUSTER
            if action == "run_cluster":
                manual_k = request.form.get("manual_k")
                used_k = int(manual_k) if manual_k and manual_k.isdigit() else optimal_k
                
                _, centroids, centroid_process = run_kmeans_internal(data_to_cluster, used_k)
                
                # 2. Tentukan label cluster (angka)
                labels = []
                for point in data_to_cluster:
                    distances = [euclidean(point, c) for c in centroids]
                    labels.append(distances.index(min(distances)))
                aggregated["cluster"] = labels

                # 3. Assign Label Nama (Sangat Laku, dll)
                label_map = auto_cluster_labels(aggregated)
                aggregated["kategori_label"] = aggregated["cluster"].map(label_map)

                # 4. Update Variabel Global & Session agar sinkron
                last_result_df = aggregated.copy()
                session['result_data'] = aggregated.to_dict('records')
                session['centroid_process'] = centroid_process

                # Keperluan dashboard web
                result_data = aggregated
                cluster_summary = aggregated.groupby("cluster").size()
                products_by_cluster = {i: aggregated[aggregated["cluster"] == i] for i in range(used_k)}
                # INTEGRASI GLOBAL END

        except Exception as e:
            error = str(e)
        finally:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)

    return render_template(
        "index.html",
        error=error,
        elbow_data=elbow_data,
        optimal_k=optimal_k,
        data_preview=data_preview,
        result=result_data,
        summary=cluster_summary,
        centroid_process=centroid_process,
        used_k=used_k,
        products_by_cluster=products_by_cluster
    )
def auto_cluster_labels(df):
    """
    Memberi nama cluster otomatis berdasarkan ranking penjualan_total.
    Akan menghasilkan urutan: Sangat Laku, Sedang, Kurang Laku, Lainnya.
    """
    cluster_stats = (
        df.groupby("cluster")
        .agg(
            avg_penjualan=("penjualan_total", "mean")
        )
        .sort_values(by="avg_penjualan", ascending=False)
    )

    # Urutan label disesuaikan dengan dashboard Anda
    base_labels = [
        "Sangat Laku",  # Rank 1 (Tertinggi)
        "Sedang",       # Rank 2
        "Kurang Laku",  # Rank 3
        "Lainnya"       # Rank 4
    ]

    label_map = {}
    for i, cluster_id in enumerate(cluster_stats.index):
        if i < len(base_labels):
            label_map[cluster_id] = base_labels[i]
        else:
            label_map[cluster_id] = f"Cluster {i+1}"
            
    return label_map

# --- Route Download Excel (Diperbarui dengan global last_result_df) ---
@app.route('/download_excel')
def download_excel():
    global last_result_df

    if last_result_df is None or last_result_df.empty:
        flash("Data belum tersedia. Jalankan K-Means terlebih dahulu.", "error")
        return redirect(url_for('index'))

    # Pastikan data yang di-download adalah versi terbaru yang sudah ada labelnya
    output = io.BytesIO()

    # Menyusun urutan kolom agar lebih rapi di Excel
    cols = last_result_df.columns.tolist()
    # Memindahkan 'cluster' dan 'kategori_label' ke bagian depan/belakang jika perlu
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        last_result_df.to_excel(
            writer,
            index=False,
            sheet_name='Hasil Clustering'
        )
        
        # Opsional: Mempercantik tampilan Excel (Auto-fit kolom)
        workbook  = writer.book
        worksheet = writer.sheets['Hasil Clustering']
        for i, col in enumerate(last_result_df.columns):
            column_len = max(last_result_df[col].astype(str).str.len().max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name='hasil_segmentasi_kmeans.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
@app.route('/download_guide')
def download_guide():
    content = """
PANDUAN PENGGUNAAN APLIKASI K-MEANS MINI MARKET

1. Unggah file CSV yang berisi kolom:
   - Produk
   - Harga Satuan
   - Jumlah Terjual
   - Tanggal

2. Data akan dimasukkan dan diproses oleh sistem.

3. Penentuan nilai K:
   - Biarkan kolom K kosong untuk menggunakan nilai K otomatis (Elbow Method).
   - Atau isi nilai K secara manual (1–10).

4. Jalankan proses K-Means.

5. Lihat hasil clustering produk berdasarkan segmentasi penjualan.

6. Unduh hasil segmentasi dalam format Excel.
"""
    return send_file(
        io.BytesIO(content.encode('utf-8')),
        as_attachment=True,
        download_name="Panduan_KMeans_MiniMarket.txt",
        mimetype="text/plain"
    )


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=10000)