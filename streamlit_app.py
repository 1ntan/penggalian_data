import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dashboard Clustering Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_csv(filename):
    path_data = DATA_DIR / filename
    path_root = BASE_DIR / filename

    if path_data.exists():
        return pd.read_csv(path_data)

    if path_root.exists():
        return pd.read_csv(path_root)

    return None


df = load_csv("df_dashboard.csv")
kmeans_results = load_csv("kmeans_results.csv")
agg_results = load_csv("agg_results.csv")
algorithm_comparison = load_csv("perbandingan_algoritma.csv")
feature_results = load_csv("feature_selection_results.csv")
birch_results = load_csv("birch_results.csv")
df_birch = load_csv("df_dashboard_birch.csv")

if df is None:
    st.error("File df_dashboard.csv belum ditemukan. Letakkan file di folder data/df_dashboard.csv.")
    st.stop()

# ============================================================
# VALIDASI KOLOM UTAMA
# ============================================================

required_columns = [
    "Cluster_KMeans",
    "Zone_ID",
    "Image_Type",
    "Action_Suggested",
    "NDVI",
    "NDRE",
    "RGB_Damage_Score",
    "N",
    "P",
    "K",
    "Moisture",
    "pH",
    "Temperature",
    "Humidity",
    "Energy_Consumed_mAh",
    "Latency_ms"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Kolom berikut belum ditemukan di df_dashboard.csv: {missing_columns}")
    st.stop()

numeric_features = [
    "NDVI",
    "NDRE",
    "RGB_Damage_Score",
    "N",
    "P",
    "K",
    "Moisture",
    "pH",
    "Temperature",
    "Humidity",
    "Energy_Consumed_mAh",
    "Latency_ms"
]

# ============================================================
# NILAI DEFAULT EVALUASI
# ============================================================

best_k_kmeans = 2
best_k_agg = 2
best_k_birch = 2

silhouette_kmeans = 0.073278
silhouette_agg = 0.030561
silhouette_birch = 0.025000

# Ambil nilai dari file kmeans_results.csv jika tersedia
if kmeans_results is not None:
    if "Jumlah Cluster (k)" in kmeans_results.columns and "Silhouette Coefficient" in kmeans_results.columns:
        best_row = kmeans_results.loc[kmeans_results["Silhouette Coefficient"].idxmax()]
        best_k_kmeans = int(best_row["Jumlah Cluster (k)"])
        silhouette_kmeans = float(best_row["Silhouette Coefficient"])

# Ambil nilai dari file agg_results.csv jika tersedia
if agg_results is not None:
    if "Jumlah Cluster (k)" in agg_results.columns and "Silhouette Coefficient" in agg_results.columns:
        best_row = agg_results.loc[agg_results["Silhouette Coefficient"].idxmax()]
        best_k_agg = int(best_row["Jumlah Cluster (k)"])
        silhouette_agg = float(best_row["Silhouette Coefficient"])

# AMBIL NILAI BIRCH
if birch_results is not None:
    if (
        "Jumlah Cluster (k)" in birch_results.columns
        and "Silhouette Coefficient" in birch_results.columns
    ):
        best_row = birch_results.loc[
            birch_results[
                "Silhouette Coefficient"
            ].idxmax()
        ]

        best_k_birch = int(
            best_row["Jumlah Cluster (k)"]
        )

        silhouette_birch = float(
            best_row["Silhouette Coefficient"]
        )

# Ambil nilai dari file perbandingan_algoritma.csv jika tersedia
if algorithm_comparison is not None:
    if "Algoritma" in algorithm_comparison.columns and "Silhouette Coefficient" in algorithm_comparison.columns:
        kmeans_match = algorithm_comparison[
            algorithm_comparison["Algoritma"].astype(str).str.contains("K-Means", case=False, na=False)
        ]

        agg_match = algorithm_comparison[
            algorithm_comparison["Algoritma"].astype(str).str.contains("Agglomerative", case=False, na=False)
        ]

        if len(kmeans_match) > 0:
            silhouette_kmeans = float(kmeans_match["Silhouette Coefficient"].iloc[0])

        if len(agg_match) > 0:
            silhouette_agg = float(agg_match["Silhouette Coefficient"].iloc[0])

# ============================================================
# FALLBACK DATA EVALUASI JIKA FILE OPSIONAL BELUM ADA
# ============================================================

if kmeans_results is None:
    kmeans_results = pd.DataFrame({
        "Jumlah Cluster (k)": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Cohesion / WCSS / Inertia": [
            49058.413529,
            46676.926797,
            44883.305454,
            43501.953271,
            42328.668662,
            41297.396553,
            40366.302019,
            39550.829484,
            38776.092785
        ],
        "Silhouette Coefficient": [
            0.073278,
            0.062211,
            0.061682,
            0.059776,
            0.061534,
            0.060035,
            0.062303,
            0.061969,
            0.062727
        ]
    })

if agg_results is None:
    agg_results = pd.DataFrame({
        "Jumlah Cluster (k)": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Cohesion": [
            4279.704152,
            4157.610752,
            4047.514972,
            3946.092028,
            3869.132276,
            3794.702207,
            3733.710785,
            3676.776411,
            3622.151167
        ],
        "Silhouette Coefficient": [
            0.030561,
            0.020107,
            0.015717,
            0.016235,
            0.013242,
            0.013394,
            0.009771,
            0.006716,
            0.007177
        ]
    })

if feature_results is None:
    feature_results = pd.DataFrame({
        "Kombinasi Fitur": [
            "Fitur Lingkungan",
            "Fitur Vegetasi",
            "Vegetasi + Lingkungan",
            "Fitur Tanah",
            "Tanah + Vegetasi",
            "Tanah + Lingkungan",
            "Semua Fitur"
        ],
        "Silhouette Coefficient": [
            0.360958,
            0.262619,
            0.208886,
            0.151636,
            0.116501,
            0.109226,
            0.073278
        ]
    })

# ============================================================
# STYLE
# ============================================================

PRIMARY = "#7A0C0C"
PRIMARY_DARK = "#4F0606"
BLACK = "#252525"
GRAY = "#6B7280"
LIGHT_BG = "#F7F7F7"
CARD_BG = "#FFFFFF"
BORDER = "#E6DADA"
SOFT_RED = "#FFF7F7"

cluster_colors = {
    # paling gelap
    "Cluster 0": "#000000",
    # merah tua
    "Cluster 1": "#4F0606",
    # merah maroon
    "Cluster 2": "#7A0C0C",
    # merah medium
    "Cluster 3": "#A61B1B",
    # merah soft
    "Cluster 4": "#C0392B",
    # coral
    "Cluster 5": "#D35454",
    # salmon
    "Cluster 6": "#E67E7E",
    # pink medium
    "Cluster 7": "#F1948A",
    # pink soft
    "Cluster 8": "#F5B7B1",
    # pink muda
    "Cluster 9": "#FADBD8"
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {LIGHT_BG};
    }}

    /* Jarak atas diperbesar supaya judul tidak ketutup header Streamlit */
    .block-container {{
        padding-top: 4.2rem;
        padding-left: clamp(0.8rem, 1.5vw, 1.5rem);
        padding-right: clamp(0.8rem, 1.5vw, 1.5rem);
        padding-bottom: 1.5rem;
        max-width: 100%;
    }}

    /* Header tetap terlihat agar tombol buka/tutup sidebar tidak hilang */
    [data-testid="stHeader"] {{
        background: rgba(247, 247, 247, 0.92);
        backdrop-filter: blur(6px);
    }}

    [data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding: 1rem 0.85rem;
    }}

    .sidebar-title {{
        background: linear-gradient(135deg, {PRIMARY_DARK}, {PRIMARY});
        color: white;
        padding: 0.85rem;
        border-radius: 12px;
        font-size: clamp(0.9rem, 1vw, 1.05rem);
        font-weight: 900;
        margin-bottom: 0.85rem;
        line-height: 1.25;
        text-align: center;
    }}

    .sidebar-info {{
        background: {SOFT_RED};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.8rem;
        margin-top: 0.85rem;
        font-size: clamp(0.72rem, 0.82vw, 0.88rem);
        line-height: 1.45;
    }}

    [data-testid="stSelectbox"] label {{
        color: #222222;
        font-size: clamp(0.74rem, 0.85vw, 0.9rem);
        font-weight: 800;
    }}

    div[data-baseweb="select"] > div {{
        border-radius: 10px;
        min-height: 38px;
        font-size: clamp(0.75rem, 0.85vw, 0.95rem);
    }}

    .main-title {{
        color: {PRIMARY_DARK};
        text-align: center;
        font-size: clamp(1.35rem, 2.2vw, 2.1rem);
        font-weight: 900;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }}

    .main-subtitle {{
        color: {GRAY};
        text-align: center;
        font-size: clamp(0.72rem, 0.95vw, 0.95rem);
        margin-bottom: 0.9rem;
        line-height: 1.35;
    }}

    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: clamp(0.75rem, 1vw, 1rem);
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        min-height: 95px;
        overflow-wrap: anywhere;
    }}

    .metric-label {{
        color: {GRAY};
        font-size: clamp(0.68rem, 0.78vw, 0.86rem);
        font-weight: 800;
        margin-bottom: 0.25rem;
    }}

    .metric-value {{
        color: #222222;
        font-size: clamp(1.25rem, 2vw, 1.95rem);
        font-weight: 900;
        line-height: 1.05;
    }}

    .metric-caption {{
        color: {GRAY};
        font-size: clamp(0.64rem, 0.75vw, 0.8rem);
        margin-top: 0.25rem;
    }}

    .section-title {{
        color: #222222;
        font-size: clamp(0.95rem, 1.1vw, 1.15rem);
        font-weight: 900;
        margin-bottom: 0.55rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid {BORDER};
    }}

    .note-box {{
        background: {SOFT_RED};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.55rem;
        color: #222222;
        font-size: clamp(0.68rem, 0.78vw, 0.84rem);
        line-height: 1.35;
    }}

    .compact-insight {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-left: 5px solid {PRIMARY};
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.6rem;
        font-size: clamp(0.72rem, 0.82vw, 0.9rem);
        line-height: 1.35;
    }}

    .compact-insight b {{
        color: {PRIMARY_DARK};
    }}

    div[data-testid="stPlotlyChart"] {{
        width: 100%;
        overflow: hidden;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPER
# ============================================================

def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def fmt_float(value, digit=6):
    return f"{float(value):.{digit}f}".replace(".", ",")


def safe_sample(data, n=7000):
    if len(data) == 0:
        return data
    return data.sample(min(n, len(data)), random_state=42)


def apply_chart_style(fig, height=300, showlegend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=18, b=32),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=11, color="#222222"),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            y=-0.23,
            x=0.5,
            xanchor="center",
            font=dict(size=10)
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EFEFEF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EFEFEF", zeroline=False)
    return fig


def prepare_algorithm_data(
    input_df,
    selected_algorithm,
    selected_k
):
    """
    Mengatur data aktif berdasarkan algoritma:
    - Semua: detail cluster memakai K-Means sebagai model utama.
    - K-Means: full data.
    - Agglomerative: sample data.
    - Birch: sample data.
    """

    # ========================================================
    # SEMUA
    # ========================================================

    if selected_algorithm == "Semua":
        cluster_col = f"KMeans_k{selected_k}"
        active = input_df.copy()
        algorithm_name = "K-Means"
        active_silhouette = silhouette_kmeans
        best_k = selected_k

    # ========================================================
    # KMEANS
    # ========================================================

    elif selected_algorithm == "K-Means":

        cluster_col = f"KMeans_k{selected_k}"
        active = input_df.copy()
        algorithm_name = "K-Means"
        if kmeans_results is not None:
            selected_row = kmeans_results[
                kmeans_results[
                    "Jumlah Cluster (k)"
                ] == selected_k
            ]

            if not selected_row.empty:
                active_silhouette = float(
                    selected_row[
                        "Silhouette Coefficient"
                    ].iloc[0]
                )

            else:
                active_silhouette = silhouette_kmeans
        else:
            active_silhouette = silhouette_kmeans
        best_k = selected_k

    # ========================================================
    # AGGLOMERATIVE
    # ========================================================

    elif selected_algorithm == "Agglomerative":
        cluster_col = f"Agglomerative_k{selected_k}"
        if cluster_col not in input_df.columns:
            st.error(
                "Kolom Cluster_Agglomerative belum ditemukan di df_dashboard.csv."
            )
            st.stop()

        active = input_df[
            input_df[cluster_col].notna()
        ].copy()

        if len(active) == 0:
            st.error(
                "Cluster Agglomerative kosong."
            )

            st.stop()

        algorithm_name = "Agglomerative"
        if agg_results is not None:
            selected_row = agg_results[
                agg_results[
                    "Jumlah Cluster (k)"
                ] == selected_k
            ]

            if not selected_row.empty:
                active_silhouette = float(
                    selected_row[
                        "Silhouette Coefficient"
                    ].iloc[0]
                )

            else:
                active_silhouette = silhouette_agg

        else:
            active_silhouette = silhouette_agg
        
        best_k = selected_k

    # ============================================================
    # BIRCH
    # ============================================================
    elif selected_algorithm == "Birch":
        cluster_col = f"Birch_k{selected_k}"
        active = input_df[
            input_df[cluster_col].notna()
        ].copy()
        algorithm_name = "Birch"

        if birch_results is not None:
            selected_row = birch_results[
                birch_results[
                    "Jumlah Cluster (k)"
                ] == selected_k
            ]

            if not selected_row.empty:
                active_silhouette = float(
                    selected_row[
                        "Silhouette Coefficient"
                    ].iloc[0]
                )

            else:
                active_silhouette = silhouette_birch

        else:
            active_silhouette = silhouette_birch
        best_k = selected_k

    active["Cluster_Label"] = (
        active[cluster_col]
        .astype("Int64")
        .astype(str)
        .apply(lambda x: f"Cluster {x}")
    )

    return (
        active,
        cluster_col,
        algorithm_name,
        active_silhouette,
        best_k
    )

# ============================================================
# SIDEBAR FILTER
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">
            🔎 FILTER & INFO DATASET
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_algorithm = st.selectbox(
        "Algoritma",
        [
            "Semua",
            "K-Means",
            "Agglomerative",
            "Birch"
        ],
        index=0
    )

    selected_k = st.selectbox(
        "Nilai K",
        list(range(2, 11)),
        index=0
    )

    active_df, cluster_column, active_algorithm_name, active_silhouette, active_best_k = prepare_algorithm_data(
        df,
        selected_algorithm,
        selected_k
    )

    cluster_options = ["Semua"] + sorted(active_df["Cluster_Label"].unique().tolist())
    zone_options = ["Semua"] + sorted(active_df["Zone_ID"].unique().tolist())
    image_options = ["Semua"] + sorted(active_df["Image_Type"].unique().tolist())
    action_options = ["Semua"] + sorted(active_df["Action_Suggested"].unique().tolist())

    selected_cluster = st.selectbox("Cluster", cluster_options)
    selected_zone = st.selectbox("Zone_ID", zone_options)
    selected_image_type = st.selectbox("Image_Type", image_options)
    selected_action = st.selectbox("Action_Suggested", action_options)

    st.markdown(
        f"""
        <div class="sidebar-info">
            <b>Dataset</b><br>
            Semantic-Aware IoT Dataset for Smart Agriculture<br><br>
            <b>Total dataset:</b> {fmt_int(len(df))}<br>
            <b>Data algoritma aktif:</b> {fmt_int(len(active_df))}<br>
            <b>Fitur final:</b> {len(numeric_features)}<br>
            <b>Algoritma aktif:</b> {selected_algorithm}<br>
            <b>Best K:</b> {active_best_k}
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FILTER DATA
# ============================================================

filtered_df = active_df.copy()

if selected_cluster != "Semua":
    filtered_df = filtered_df[filtered_df["Cluster_Label"] == selected_cluster]

if selected_zone != "Semua":
    filtered_df = filtered_df[filtered_df["Zone_ID"] == selected_zone]

if selected_image_type != "Semua":
    filtered_df = filtered_df[filtered_df["Image_Type"] == selected_image_type]

if selected_action != "Semua":
    filtered_df = filtered_df[filtered_df["Action_Suggested"] == selected_action]

if len(filtered_df) == 0:
    st.warning("Tidak ada data yang sesuai dengan filter.")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="main-title">Dashboard Clustering Smart Agriculture</div>
    <div class="main-subtitle">
        Analisis hasil clustering zona lahan | Algoritma aktif: <b>{selected_algorithm}</b>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# KPI
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

kpi_items = [
    ("Total Dataset", fmt_int(len(df)), "record awal"),
    ("Data Digunakan", fmt_int(len(active_df)), active_algorithm_name),
    ("Data Terfilter", fmt_int(len(filtered_df)), "sesuai filter"),
    ("Fitur Final", str(len(numeric_features)), "atribut numerik"),
    ("Jumlah Cluster (K)", active_best_k, selected_algorithm),
    ("Silhouette", fmt_float(active_silhouette), selected_algorithm)
]

for col, (label, value, caption) in zip([k1, k2, k3, k4, k5, k6], kpi_items):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# ============================================================
# ROW 1: PERBANDINGAN, DISTRIBUSI, PCA
# ============================================================

row1_col1, row1_col2, row1_col3 = st.columns([1.05, 1.05, 1.45])

with row1_col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Perbandingan Algoritma</div>', unsafe_allow_html=True)

        compare_df = pd.DataFrame({
            "Algoritma": [
                "K-Means",
                "Agglomerative",
                "Birch"
            ],
            "Data Digunakan": [
                "Full data",
                "Sample 5000",
                "Sample 5000"
            ],
            "Silhouette": [
                silhouette_kmeans,
                silhouette_agg,
                silhouette_birch
            ]
        })

        fig_compare = px.bar(
            compare_df,
            x="Algoritma",
            y="Silhouette",
            color="Algoritma",
            text=compare_df["Silhouette"].map(lambda x: fmt_float(x)),
            color_discrete_map={
                "K-Means": PRIMARY,
                "Agglomerative": BLACK,
                "Birch": "#9B9292"
            }
        )

        fig_compare.update_traces(textposition="outside")
        fig_compare.update_yaxes(range=[0, max(compare_df["Silhouette"]) * 1.6])
        fig_compare = apply_chart_style(fig_compare, height=270, showlegend=False)
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown(
            """
            <div class="note-box">
                K-Means menjadi model utama.
            </div>
            """,
            unsafe_allow_html=True
        )

with row1_col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Distribusi Cluster</div>', unsafe_allow_html=True)

        cluster_count = filtered_df["Cluster_Label"].value_counts().sort_index()
        cluster_table = cluster_count.reset_index()
        cluster_table.columns = ["Cluster", "Jumlah Data"]
        cluster_table["Persentase (%)"] = (
            cluster_table["Jumlah Data"] / len(filtered_df) * 100
        ).round(2)

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=cluster_table["Cluster"],
                    values=cluster_table["Jumlah Data"],
                    hole=0.55,
                    marker=dict(
                        colors=[
                            cluster_colors.get(label, PRIMARY)
                            for label in cluster_table["Cluster"]
                        ]
                    ),
                    textinfo="percent",
                    sort=False
                )
            ]
        )

        fig_donut.update_layout(
            height=250,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                y=-0.1,
                x=0.5,
                xanchor="center",
                font=dict(size=10)
            )
        )

        st.plotly_chart(fig_donut, use_container_width=True)
        st.dataframe(cluster_table, use_container_width=True, hide_index=True, height=125)

with row1_col3:
    with st.container(border=True):
        st.markdown('<div class="section-title">Visualisasi Cluster PCA</div>', unsafe_allow_html=True)

        if "PCA1" in filtered_df.columns and "PCA2" in filtered_df.columns:
            sample_df = safe_sample(filtered_df, 8000)

            fig_pca = px.scatter(
                sample_df,
                x="PCA1",
                y="PCA2",
                color="Cluster_Label",
                color_discrete_map=cluster_colors,
                opacity=0.72,
                labels={
                    "PCA1": "PCA 1",
                    "PCA2": "PCA 2",
                    "Cluster_Label": "Cluster"
                },
                hover_data=[
                    "Zone_ID",
                    "Image_Type",
                    "Action_Suggested",
                    "N",
                    "P",
                    "K",
                    "Moisture",
                    "pH"
                ]
            )

            fig_pca.update_traces(marker=dict(size=4))
            fig_pca = apply_chart_style(fig_pca, height=360)
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("Kolom PCA1 dan PCA2 belum tersedia di df_dashboard.csv.")

# ============================================================
# ROW 2: EVALUASI K, PROFIL, ATRIBUT PEMBEDA
# ============================================================

row2_col1, row2_col2, row2_col3 = st.columns([1.2, 1.15, 1.25])

with row2_col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Evaluasi Jumlah Cluster</div>', unsafe_allow_html=True)

        if selected_algorithm == "Agglomerative":
            eval_k_df = agg_results.copy()
            cohesion_col = "Cohesion"
        elif selected_algorithm == "Birch":
            eval_k_df = birch_results.copy()
            cohesion_col = "Cohesion"
        else:
            eval_k_df = kmeans_results.copy()
            cohesion_col = "Cohesion / WCSS / Inertia"

        fig_sil = px.line(
            eval_k_df,
            x="Jumlah Cluster (k)",
            y="Silhouette Coefficient",
            markers=True,
            labels={"Silhouette Coefficient": "Silhouette"}
        )
        fig_sil.update_traces(line=dict(color=PRIMARY, width=3), marker=dict(size=7))
        fig_sil = apply_chart_style(fig_sil, height=210, showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)

        if cohesion_col in eval_k_df.columns:
            fig_cohesion = px.line(
                eval_k_df,
                x="Jumlah Cluster (k)",
                y=cohesion_col,
                markers=True,
                labels={cohesion_col: "Cohesion / WCSS"}
            )
            fig_cohesion.update_traces(line=dict(color=BLACK, width=3), marker=dict(size=7))
            fig_cohesion = apply_chart_style(fig_cohesion, height=210, showlegend=False)
            st.plotly_chart(fig_cohesion, use_container_width=True)

with row2_col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Profil Rata-rata Cluster</div>', unsafe_allow_html=True)

        profile_features = ["N", "P", "K", "Moisture", "pH"]

        profile_df = (
            filtered_df
            .groupby("Cluster_Label")[profile_features]
            .mean()
            .round(3)
        )

        st.dataframe(profile_df, use_container_width=True, height=145)

        profile_long = profile_df.reset_index().melt(
            id_vars="Cluster_Label",
            var_name="Atribut",
            value_name="Rata-rata"
        )

        fig_profile = px.bar(
            profile_long,
            x="Atribut",
            y="Rata-rata",
            color="Cluster_Label",
            barmode="group",
            color_discrete_map=cluster_colors,
            labels={
                "Cluster_Label": "Cluster",
                "Rata-rata": "Rata-rata"
            }
        )

        fig_profile = apply_chart_style(fig_profile, height=260)
        st.plotly_chart(fig_profile, use_container_width=True)

with row2_col3:
    with st.container(border=True):
        st.markdown('<div class="section-title">Atribut Pembeda Utama</div>', unsafe_allow_html=True)

        full_profile = active_df.groupby("Cluster_Label")[numeric_features].mean().round(4)

        if len(full_profile.index) >= 2:
            idx0 = full_profile.index[0]
            idx1 = full_profile.index[1]

            diff_series = abs(full_profile.loc[idx0] - full_profile.loc[idx1]).sort_values(ascending=False)
            diff_df = diff_series.head(5).reset_index()
            diff_df.columns = ["Atribut", "Selisih Rata-rata"]

            fig_diff = px.bar(
                diff_df.sort_values("Selisih Rata-rata"),
                x="Selisih Rata-rata",
                y="Atribut",
                orientation="h",
                text="Selisih Rata-rata",
                color="Selisih Rata-rata",
                color_continuous_scale=[[0, "#BBBBBB"], [1, PRIMARY]],
                labels={"Selisih Rata-rata": "Selisih"}
            )

            fig_diff.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_diff.update_layout(coloraxis_showscale=False)
            fig_diff = apply_chart_style(fig_diff, height=320, showlegend=False)
            st.plotly_chart(fig_diff, use_container_width=True)

        if selected_algorithm in [
            "Semua",
            "K-Means"
        ]:
            st.markdown(
                """
                <div class="note-box">
                    Pembeda utama K-Means: <b>P / Fosfor</b>.
                </div>
                """,
                unsafe_allow_html=True
            )
        elif selected_algorithm == "Birch":
            st.markdown(
                """
                <div class="note-box">
                    Pembeda utama Birch cenderung pada <b>Moisture</b> dan <b>N</b>.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="note-box">
                    Pembeda utama Agglomerative cenderung pada <b>Latency_ms</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# ROW 3: FEATURE EXPERIMENT, ZONE, ACTION
# ============================================================

row3_col1, row3_col2, row3_col3 = st.columns([1.2, 1.25, 1.25])

with row3_col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Eksperimen Kombinasi Fitur</div>', unsafe_allow_html=True)

        if "Kombinasi Fitur" in feature_results.columns and "Silhouette Coefficient" in feature_results.columns:
            feat_df = feature_results.copy()
            feat_df = feat_df.sort_values("Silhouette Coefficient", ascending=True)

            fig_feat = px.bar(
                feat_df,
                x="Silhouette Coefficient",
                y="Kombinasi Fitur",
                orientation="h",
                text=feat_df["Silhouette Coefficient"].map(lambda x: fmt_float(x)),
                color="Silhouette Coefficient",
                color_continuous_scale=[[0, "#BBBBBB"], [1, PRIMARY]]
            )

            fig_feat.update_traces(textposition="outside")
            fig_feat.update_layout(coloraxis_showscale=False)
            fig_feat = apply_chart_style(fig_feat, height=320, showlegend=False)
            st.plotly_chart(fig_feat, use_container_width=True)

with row3_col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Distribusi Cluster per Zone_ID</div>', unsafe_allow_html=True)

        zone_cluster = pd.crosstab(
            filtered_df["Zone_ID"],
            filtered_df["Cluster_Label"],
            normalize="index"
        ) * 100

        zone_long = zone_cluster.reset_index().melt(
            id_vars="Zone_ID",
            var_name="Cluster",
            value_name="Persentase"
        )

        fig_zone = px.bar(
            zone_long,
            x="Zone_ID",
            y="Persentase",
            color="Cluster",
            barmode="stack",
            color_discrete_map=cluster_colors,
            labels={"Persentase": "Persentase (%)"}
        )

        fig_zone = apply_chart_style(fig_zone, height=320)
        st.plotly_chart(fig_zone, use_container_width=True)

with row3_col3:
    with st.container(border=True):
        st.markdown('<div class="section-title">Action Suggested per Cluster</div>', unsafe_allow_html=True)

        action_cluster = pd.crosstab(
            filtered_df["Cluster_Label"],
            filtered_df["Action_Suggested"],
            normalize="index"
        ) * 100

        action_long = action_cluster.reset_index().melt(
            id_vars="Cluster_Label",
            var_name="Action Suggested",
            value_name="Persentase"
        )

        fig_action = px.bar(
            action_long,
            x="Persentase",
            y="Cluster_Label",
            color="Action Suggested",
            color_discrete_sequence=[
                "#000000",  # hitam
                "#4F0606",  # merah sangat gelap
                "#7A0C0C",  # maroon
                "#A61B1B",  # merah medium
                "#C0392B",  # merah soft
                "#808080",  # abu
                "#B0B0B0",  # abu muda
                "#F1948A",  # pink soft
                "#F5B7B1"   # pink muda
            ],
            orientation="h",
            text=action_long["Persentase"].map(
                lambda x: f"{x:.1f}%"
            ),
            labels={
                "Cluster_Label": "Cluster",
                "Persentase": "Persentase (%)"
            }
        )

        fig_action.update_traces(textposition="inside")
        fig_action = apply_chart_style(fig_action, height=320)
        st.plotly_chart(fig_action, use_container_width=True)

        st.caption("Action_Suggested hanya informasi pendukung.")

# ============================================================
# ROW 4: DATA DAN RINGKASAN
# ============================================================

row4_col1, row4_col2 = st.columns([1.65, 1])

with row4_col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Data Anggota Cluster</div>', unsafe_allow_html=True)

        display_columns = [
            "Zone_ID",
            "Image_Type",
            "Action_Suggested",
            "NDVI",
            "NDRE",
            "RGB_Damage_Score",
            "N",
            "P",
            "K",
            "Moisture",
            "pH",
            "Temperature",
            "Humidity",
            "Energy_Consumed_mAh",
            "Latency_ms",
            cluster_column
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=320
        )

        st.download_button(
            "Download Data Hasil Filter",
            data=filtered_df[display_columns].to_csv(index=False).encode("utf-8"),
            file_name="hasil_filter_cluster.csv",
            mime="text/csv"
        )

with row4_col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Ringkasan Temuan</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="compact-insight">
                <b>Algoritma aktif</b><br>
                {selected_algorithm}
            </div>
            """,
            unsafe_allow_html=True
        )

        if selected_algorithm == "Agglomerative":
            data_used_text = "Sample 5.000 data"
        elif selected_algorithm == "Birch":
            data_used_text = "Sample 5.000 data"
        elif selected_algorithm == "K-Means":
            data_used_text = "Full data 60.000 record"
        else:
            data_used_text = "Perbandingan algoritma, detail memakai K-Means"
        
        st.markdown(
            f"""
            <div class="compact-insight">
                <b>Data digunakan</b><br>
                {data_used_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="compact-insight">
                <b>Insight utama</b><br>
                K-Means dipilih sebagai model utama.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="compact-insight">
                <b>Catatan</b><br>
                Silhouette rendah, hasil bersifat eksploratif.
            </div>
            """,
            unsafe_allow_html=True
        )

st.caption("Dashboard" \
"rd hasil clustering smart agriculture.")
