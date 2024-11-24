# Mengimpor berbagai pustaka yang diperlukan untuk analisis data dan model regresi linier.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#penyesuaian tema dan mode gelap di Streamlit
st.set_page_config(page_title="Data Penumpang & Kapal Cruise Analysis", layout="wide")
st.markdown("""
    <style>
        body { color: #E8E9EB; background-color: #1C1E21; }
        .css-1aumxhk { background-color: #282C34; color: #F0F0F0; }
        h1, h2, h3, h4 { color: #E8E9EB; }
        .stDataFrame, .stText { color: #E8E9EB; background-color: #1C1E21; }
        .stButton>button { color: #FFFFFF; background-color: #4A90E2; transition: all 0.3s ease; }
        .stButton>button:hover { background-color: #5A99D1; cursor: pointer; }
        .css-1cpxqw2 { background-color: #333; color: #E8E9EB; }
        .box { border-radius: 10px; background-color: #333; padding: 20px; margin-top: 20px; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background-color: #444; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


# Fungsi untuk pra-pemrosesan data
def preprocess_data(df):
    # Menghapus kolom yang tidak diperlukan (Unnamed: 7), jika ada
    df = df.drop(columns=['Unnamed: 7'], errors='ignore')
    
    # Menghapus baris yang memiliki nilai kosong
    df.dropna(inplace=True)
    
    # Mengganti nama kolom agar lebih konsisten
    df.rename(columns={'JUMLAH\nCREW': 'JUMLAH CREW'}, inplace=True)
    
    # Menstandarkan nama lokasi di kolom 'CABANG'
    df['CABANG'] = df['CABANG'].replace({
        'BENOA': 'Benoa', 
        'LEMBAR': 'Lembar', 
        'TANJUNG PERAK': 'Tanjung Perak',
        'SEMARANG': 'Semarang',
        'CELUKAN BAWANG': 'Celukan Bawang',
        'TANJUNG TEMBAGA': 'Tanjung Tembaga',
        'ENDE': 'Ende', 
        'GIILIMAS LEMBAR': 'Giilimas Lembar', 
        'SURABAYA': 'Surabaya', 
        'TANJUNG EMAS': 'Tanjung Emas',
    })
    
    # Menstandarkan variasi nama di kolom 'DARI'
    df['DARI'] = df['DARI'].replace({
        'SINGAPURA': 'SINGAPORE',
        'TG. PERAK': 'TANJUNG PERAK',
        'TJ PERAK': 'TANJUNG PERAK',
        'TJ. PERAK': 'TANJUNG PERAK',
        'TG PERAK': 'TANJUNG PERAK',
        'TJ PRIUK': 'TANJUNG PRIOK',
        'TANJUNG PRIUK': 'TANJUNG PRIOK',
        'TJ EMAS': 'TANJUNG EMAS',
        'TANJUNG EMAS, SEMA': 'TANJUNG EMAS',
        'TANJUNGWANGI': 'TANJUNG WANGI',
        'CLKN BAWANG': 'CELUKAN BAWANG',
        'CELUKANG BAWANG': 'CELUKAN BAWANG',
        'P. SERIBU': 'KEPULAUAN SERIBU',
        'P.SERIBU': 'KEPULAUAN SERIBU',
        'KEP. SERIBU': 'KEPULAUAN SERIBU',
        'KARIMUNJAWA': 'KARIMUN JAWA',
        'GILI NANGGU': 'GILI NANGGU',
        'GILI GENTENG, MADUR': 'GILI GENTENG',
        'GILIMAS': 'GILI MAS',
        'GILI ISLAND': 'GILI ISLANDS',
        'FREEMANTLE': 'FREMANTLE',
        'PROBOLINGO': 'PROBOLINGGO',
        'PULAU KOMODO': 'KOMODO',
        'SUMENEP,MADURA': 'SUMENEP',
        'KOH SAMUI': 'KO SAMUI',
        'KOSAMUI': 'KO SAMUI',
        'SAI GON NEW PORT': 'HO CHI MINH',
        'SENGGIGI/LOMBOK': 'SENGGIGI',
        'KLANG': 'PORT KLANG',
        'PHILIPHINE': 'PHILIPPINES',
    })

    # Menstandarkan variasi nama di kolom 'TUJUAN'
    df['TUJUAN'] = df['TUJUAN'].replace({
        'SINGAPURA': 'SINGAPORE',
        'SONGAPORE': 'SINGAPORE',
        'FREEMANTLE': 'FREMANTLE',
        'FERMANTLE': 'FREMANTLE',
        'PORTHEADLAND': 'PORT HEDLAND',
        'BROME': 'BROOME',
        'PROBOLINGO': 'PROBOLINGGO',
        'CELUKAN BAWANG': 'C. BAWANG',
        'TJ. PERAK': 'TANJUNG PERAK',
        'TJ. EMAS': 'TANJUNG EMAS',
        'P. KARIMUN JAWA': 'KARIMUN JAWA',
        'P.KARIMUN': 'KARIMUN JAWA',
        'KARIMUNJAWA': 'KARIMUN JAWA',
        'P.SERIBU': 'KEPULAUAN SERIBU',
        'KEP. SERIBU': 'KEPULAUAN SERIBU',
        'P KOMODO': 'KOMODO',
        'SLAWY BAY, KOMODO': 'KOMODO',
        'UJUNG PANDANG': 'MAKASSAR',
        'BENOA,BALI': 'BENOA',
        'LOVINA, BALI': 'LOVINA',
        'TG. PRIOK': 'TANJUNG PRIOK',
        'PORTKLANG': 'PORT KLANG',
        'KLANG': 'PORT KLANG',
        'SANDAKAN,SABAH': 'SANDAKAN',
        'HOCHIMIN CITY': 'HO CHI MINH',
        'PHILIPINE': 'PHILIPPINES',
        'PHILIPINA': 'PHILIPPINES',
        'MALAYSIAN': 'MALAYSIA',
        'MARITUNIA': 'MAURITANIA',
    })
    
    # Mengubah kolom tanggal agar memiliki format datetime
    df['TANGGAL KEDATANGAN'] = pd.to_datetime(df['TANGGAL KEDATANGAN'].str.replace('/', '-', regex=False), format='%d-%m-%y', errors='coerce')
    df['TANGGAL KEBERANGKATAN'] = pd.to_datetime(df['TANGGAL KEBERANGKATAN'].str.replace('/', '-', regex=False), format='%d-%m-%y', errors='coerce')
    
    # Mengubah kolom numerik dengan karakter koma menjadi tipe numerik
    df['PENUMPANG TURUN'] = pd.to_numeric(df['PENUMPANG TURUN'].str.replace(',', '', regex=False), errors='coerce')
    df['PENUMPANG NAIK'] = pd.to_numeric(df['PENUMPANG NAIK'].str.replace(',', '', regex=False), errors='coerce')
    df['JUMLAH CREW'] = pd.to_numeric(df['JUMLAH CREW'].str.replace(',', '', regex=False), errors='coerce')
    
    # Menghapus baris yang masih memiliki nilai kosong
    df.dropna(inplace=True)
    
    # Menghitung durasi perjalanan dalam hari
    df['DURASI'] = (df['TANGGAL KEBERANGKATAN'] - df['TANGGAL KEDATANGAN']).dt.days
    
    # Mengonversi data cabang pelabuhan menjadi nilai numerik unik
    unique_ports = df['CABANG'].unique()
    port_to_number = {port: idx for idx, port in enumerate(unique_ports)}
    df['CABANG_NEW'] = df['CABANG'].replace(port_to_number)
    
    # Menghapus kolom 'NAMA KAPAL' karena tidak diperlukan dalam analisis
    df = df.drop('NAMA KAPAL', axis=1)


    return df

# Fungsi model regresi linear
def linear_regression_model(df_processed):
    X = df_processed[['PENUMPANG NAIK']].values  # Menggunakan satu kolom untuk visualisasi
    y = df_processed['JUMLAH CREW'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)  # Menghitung Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # Menghitung nilai R-squared
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  # Membandingkan nilai aktual dan prediksi
    y_pred_full = model.predict(scaler.transform(X))  # Prediksi untuk seluruh dataset
    return model, mse, r2, comparison, X, y, y_pred_full


#visualisasi
def plot_wordcloud(df_processed):
    # Menggabungkan semua lokasi keberangkatan untuk word cloud
    text = ' '.join(df_processed['DARI'])
    
    # Membuat word cloud dari teks yang digabungkan
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    
    # Menampilkan word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud Lokasi Keberangkatan')
    st.pyplot(plt)

def plot_heatmap(df_processed):
    # Membuat heatmap untuk setiap cabang berdasarkan jumlah penumpang naik, turun, dan jumlah kru
    df_heatmap = df_processed.groupby('CABANG')[['PENUMPANG NAIK', 'PENUMPANG TURUN', 'JUMLAH CREW']].sum()

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu', fmt='g')  # Menambahkan anotasi dan memilih palet warna

    plt.title('Heatmap Perbandingan Penumpang Naik, Turun, dan Jumlah Kru per Cabang')
    plt.xlabel('Jenis Data')
    plt.ylabel('Cabang')
    plt.tight_layout()
    st.pyplot(plt)

# Membuat box plot untuk 10 tujuan dengan rata-rata penumpang tertinggi
def plot_top_tujuan(df_processed):
    # Mendapatkan 10 tujuan dengan rata-rata penumpang naik tertinggi
    top_avg_passengers = df_processed.groupby('TUJUAN')['PENUMPANG NAIK'].mean().nlargest(10).index
    # Memfilter data berdasarkan tujuan tersebut
    filtered_df = df_processed[df_processed['TUJUAN'].isin(top_avg_passengers)]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='TUJUAN', y='PENUMPANG NAIK', data=filtered_df)  # Membuat box plot
    plt.title("Box Plot Jumlah Penumpang per 10 Tujuan dengan Rata-rata Penumpang Tertinggi")
    plt.xlabel("Tujuan")
    plt.ylabel("Jumlah Penumpang")
    plt.xticks(rotation=45)  # Memutar label tujuan agar lebih mudah dibaca
    st.pyplot(plt)


# Bubble chart untuk jumlah penumpang vs kru berdasarkan rute
def plot_bubble_chart(df_processed):
    # Membuat plot scatter dengan ukuran gelembung berdasarkan jumlah penumpang
    plt.figure(figsize=(12, 6))
    plt.scatter(
        x=df_processed['CABANG'], 
        y=df_processed['JUMLAH CREW'], 
        s=df_processed['PENUMPANG NAIK']/10,  # Skala ukuran gelembung
        alpha=0.6, 
        color="green", 
        edgecolors="w", 
        linewidth=0.5
    )
    plt.title("Bubble Chart Jumlah Penumpang dan Kru per Rute")
    plt.xlabel("Cabang Keberangkatan")
    plt.ylabel("Jumlah Kru")
    plt.xticks(rotation=45)  # Memutar label cabang untuk keterbacaan
    st.pyplot(plt)

# Pie chart untuk lokasi keberangkatan teratas
def plot_departure_pie_chart(df_processed):
    # Mengelompokkan berdasarkan kolom 'DARI' dan menghitung setiap lokasi keberangkatan
    departure_counts = df_processed['DARI'].value_counts()

    # Mengambil 10 lokasi keberangkatan teratas berdasarkan jumlah
    top_departures = departure_counts.head(10)

    # Mengatur ukuran figure untuk pie chart
    plt.figure(figsize=(8, 8))  # Ukuran lebih besar untuk keterbacaan

    # Membuat pie chart
    top_departures.plot.pie(
        autopct='%1.1f%%',  # Menampilkan persentase dengan satu angka desimal
        startangle=90,      # Memulai dari sudut 90 derajat
        colors=sns.color_palette("Pastel1"),  # Menggunakan palet warna pastel
        wedgeprops={'edgecolor': 'black'}  # Menambahkan garis tepi hitam pada setiap irisan
    )

    plt.title("Pie Chart 10 Lokasi Keberangkatan Teratas")  # Menambahkan judul untuk pie chart
    st.pyplot(plt)  # Menampilkan pie chart di Streamlit

# Bar chart untuk 10 tujuan teratas berdasarkan frekuensi
def plot_tujuan(df_processed):
    # Menghitung frekuensi dari setiap tujuan
    tujuan_counts = df_processed['TUJUAN'].value_counts()

    # Mengambil 10 tujuan dengan frekuensi tertinggi
    top_tujuan = tujuan_counts.head(10)

    # Membuat bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_tujuan.values, y=top_tujuan.index, palette='viridis')
    plt.xlabel('Frekuensi')
    plt.ylabel('Lokasi Tujuan')
    plt.title('Top 10 Lokasi Tujuan Berdasarkan Frekuensi')  # Menambahkan judul
    plt.tight_layout()
    
    # Menampilkan plot menggunakan Streamlit
    st.pyplot(plt)

# Donut chart untuk distribusi 5 tujuan teratas
def plot_donut_chart(df_processed):
    # Menghitung frekuensi tujuan berdasarkan jumlah penumpang naik
    tujuan_counts = df_processed['TUJUAN'].value_counts()

    # Mengambil 5 tujuan terbanyak
    top_5_tujuan = tujuan_counts.head(5)

    # Membuat donut chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        top_5_tujuan, 
        labels=top_5_tujuan.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=sns.color_palette("Pastel1"),
        wedgeprops={'edgecolor': 'black', 'width': 0.3}  # Menambahkan lebar untuk efek donut
    )
    plt.title('Distribusi 5 Tujuan Terbanyak (Donut Chart)')  # Menambahkan judul
    st.pyplot(plt)


# Membuat stacked area chart untuk tren penumpang naik per tujuan
def plot_stacked_area_chart(df_processed):
    # Mengubah kolom tanggal menjadi format datetime
    df_processed['TANGGAL KEBERANGKATAN'] = pd.to_datetime(df_processed['TANGGAL KEBERANGKATAN'])
    
    # Mengelompokkan data berdasarkan tanggal dan tujuan, lalu menghitung total penumpang naik
    df_grouped = df_processed.groupby([df_processed['TANGGAL KEBERANGKATAN'].dt.date, 'TUJUAN'])['PENUMPANG NAIK'].sum().unstack()

    # Menentukan 5 tujuan terbanyak berdasarkan jumlah penumpang naik
    top_5_tujuan = df_grouped.sum().sort_values(ascending=False).head(5).index
    df_grouped = df_grouped[top_5_tujuan]  # Hanya menyimpan 5 tujuan terbanyak

    # Menambahkan kategori 'Lainnya' untuk tujuan selain top 5
    df_grouped['Lainnya'] = df_processed[~df_processed['TUJUAN'].isin(top_5_tujuan)] \
                            .groupby([df_processed['TANGGAL KEBERANGKATAN'].dt.date, 'TUJUAN'])['PENUMPANG NAIK'] \
                            .sum().unstack().sum(axis=1)

    # Membuat stacked area chart
    plt.figure(figsize=(12, 6))
    df_grouped.plot.area(stacked=True, figsize=(12, 6), cmap='coolwarm', alpha=0.6)

    plt.title('Tren Penumpang Naik per Tujuan (Top 5 + Lainnya)')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Penumpang Naik')
    plt.xticks(rotation=45)
    st.pyplot(plt)


# Membuat area chart rata-rata penumpang naik berdasarkan cabang
def plot_average_passengers_by_branch(df_processed):
    # Menghitung rata-rata jumlah penumpang per cabang
    average_passengers_per_branch = df_processed.groupby('CABANG')['PENUMPANG NAIK'].mean().reset_index()

    # Membuat area chart
    plt.figure(figsize=(20, 12))
    sns.lineplot(data=average_passengers_per_branch, x='CABANG', y='PENUMPANG NAIK', marker='o', color='blue')
    plt.fill_between(x=average_passengers_per_branch['CABANG'], y1=average_passengers_per_branch['PENUMPANG NAIK'], color='blue', alpha=0.3)

    # Menambahkan judul dan label
    plt.title('Rata-Rata Penumpang Naik Berdasarkan Cabang')
    plt.xlabel('Cabang')
    plt.ylabel('Rata-Rata Penumpang Naik')
    plt.xticks(rotation=45)

    # Menampilkan plot
    plt.tight_layout()
    st.pyplot(plt)


# Membuat histogram untuk 5 tujuan teratas berdasarkan jumlah perjalanan
def plot_top_5_destinations_histogram(df_processed):
    plt.figure(figsize=(12, 6))

    # Memilih hanya baris yang sesuai dengan 5 tujuan teratas
    top_5_destinations_names = df_processed['TUJUAN'].value_counts().nlargest(5).index
    top_5_data = df_processed[df_processed['TUJUAN'].isin(top_5_destinations_names)]

    # Membuat histogram
    sns.histplot(data=top_5_data, x='TUJUAN', color='skyblue', discrete=True)
    plt.title('Top 5 Destinasi Berdasarkan Jumlah Perjalanan (Histogram)')
    plt.xlabel('Destination')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


# Membuat bar chart untuk 5 rute paling sering dikunjungi
def plot_top_5_routes(df_processed):
    # Menggabungkan kolom 'DARI' dan 'TUJUAN' menjadi kolom 'Route' dan mengambil 5 rute teratas
    top_routes = df_processed.groupby(['DARI', 'TUJUAN']).size().nlargest(5).reset_index(name='Count')
    top_routes['Route'] = top_routes['DARI'] + '-' + top_routes['TUJUAN']  # Membuat kolom gabungan rute

    # Membuat bar chart menggunakan kolom 'Route'
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_routes, x='Count', y='Route', palette='viridis')
    plt.title('Top 5 Rute Paling Sering Dikunjungi')
    plt.xlabel('Number of Trips')
    plt.ylabel('Route (Dari-Tujuan)')
    plt.tight_layout()
    st.pyplot(plt)


# Membuat boxplot distribusi penumpang naik untuk 5 tujuan teratas
def plot_top_5_destinations_passenger_distribution(df_processed):
    # Mengambil 5 tujuan teratas berdasarkan frekuensi
    top_5_destinations = df_processed['TUJUAN'].value_counts().nlargest(5).index

    # Menyaring data agar hanya mencakup 5 tujuan teratas
    top_5_data = df_processed[df_processed['TUJUAN'].isin(top_5_destinations)]

    # Membuat boxplot untuk distribusi penumpang naik pada 5 tujuan teratas
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=top_5_data, x='TUJUAN', y='PENUMPANG NAIK')
    plt.title('Distribusi Penumpang Naik Berdasarkan 5 Tujuan Teratas')
    plt.xlabel('Tujuan')
    plt.ylabel('Jumlah Penumpang Naik')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


# Membuat pie chart untuk durasi rata-rata 5 tujuan teratas
def plot_top_5_destinations_avg_duration(df_processed):
    # Menyiapkan data untuk 5 tujuan teratas berdasarkan durasi rata-rata
    top_5_destinations = df_processed.groupby('TUJUAN')['DURASI'].mean().nlargest(5)

    # Mengambil nilai dan label untuk pie chart
    sizes = top_5_destinations.values
    labels = top_5_destinations.index
    colors = sns.color_palette("pastel", len(sizes))

    # Membuat pie chart dengan efek donat dan bayangan 3D
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    # Menambahkan efek bayangan 3D
    for wedge in wedges:
        wedge.set_edgecolor("black")
        wedge.set_linewidth(1.5)
        wedge.set_alpha(0.85)

    # Menyesuaikan rasio aspek untuk memberikan efek miring
    ax.set(aspect="auto")
    plt.title("Top 5 Tujuan Berdasarkan Durasi Rata-Rata (Pie Chart 3D)")

    st.pyplot(plt)


# Membuat pie chart untuk proporsi total penumpang naik pada 10 tujuan teratas
def plot_top_10_destinations_passengers(df_processed):
    # Menghitung total penumpang untuk setiap tujuan dan mengambil 10 tujuan teratas
    destination_totals = df_processed.groupby('TUJUAN')['PENUMPANG NAIK'].sum().nlargest(10)

    # Membuat pie chart untuk 10 tujuan teratas berdasarkan total penumpang naik
    plt.figure(figsize=(8, 8))
    destination_totals.plot(
        kind='pie', 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("cool", len(destination_totals))
    )
    plt.title('Proporsi Total Penumpang Naik Berdasarkan 10 Tujuan Teratas')
    plt.ylabel('')  # Menghapus label y-axis untuk tampilan yang lebih rapi
    st.pyplot(plt)


# Streamlit interface

def main():
     # Menampilkan judul utama dengan ikon dan warna
    st.markdown(
        """
        <h2 style="text-align: center; color: #F1C40F;"> 
            📊 Aplikasi Sederhana Data Pariwisata🚢
        </h2>
        <p style="text-align: center; font-size: 16px; color: #E5E5E5;">
            Analisis visual dari data perjalanan penumpang dan kapal pesiar yang melibatkan lima pelabuhan utama di Indonesia.
        </p>
        <p style="text-align: center; font-size: 14px; color: #BDC3C7;">
            Dataset ini mencakup jumlah arus penumpang dan kapal pesiar di Pelabuhan Benoa, Tanjung Mas, Tanjung Perak, Tanjung Priok, dan Lembar untuk tahun 2017-2023.
        </p>
        <p style="text-align: center; font-size: 14px; color: #BDC3C7;">
            <a href="https://katalogdata.kemenparekraf.go.id/dataset/data-arus-penumpang-dan-kunjungan-kapal-pesiar-di-pelabuhan-benoa-tahun-2017/resource/fc75db95-10f8-424f-b859-693833470b93" style="text-decoration: none; color: #3498DB;">
                🔗 Link ke Dataset
            </a>
        </p>
        <h3 style="text-align: center; color: #F1C40F;">
            Anggota Kelompok 4
        </h3>
        <ul style="text-align: center; font-size: 14px; color: #E5E5E5; list-style-type: none; padding-left: 0;">
            <li>1. Najla Nadiva - 2211521006</li>
            <li>2. Naufal Adli Dhiaurrahman - 2211521008</li>
            <li>3. Fajrin Putra Pratama - 2211523001</li>
            <li>4. Laura Iffa Razitta - 2211523020</li>
        </ul>
        
        <h4 style="text-align: center; color: #E5E5E5;">
            Gambaran Umum Analisis
        </h4>
        <p style="text-align: center; font-size: 14px; color: #BDC3C7;">
            Model regresi linier digunakan untuk menganalisis hubungan antara dua variabel: jumlah penumpang naik dan jumlah kru kapal. 
        <p style="text-align: center; font-size: 14px; color: #BDC3C7;">
            Hasil analisis ini berguna untuk merencanakan kebutuhan kru kapal dengan lebih akurat dan dapat digunakan untuk memperbaiki perencanaan operasional kapal.
        </p>
        """, 
        unsafe_allow_html=True
    )



    # Continue with the rest of the visualizations
    st.title("📤 Data Penumpang dan Kapal Cruise 2017-2023")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write(df)  # Display the first few rows of the original dataframe

        # Preprocessing and Analysis Button
        if st.button('Run Preprocessing and Analysis'):
            df_processed = preprocess_data(df)
            st.session_state['df_processed'] = df_processed

            # Boxed section for processed data
            with st.container():
                st.subheader("📈 Processed Data (After Preprocessing)")
                st.write(df_processed)

            # Linear regression analysis
            model, mse, r2, comparison, X, y, y_pred = linear_regression_model(df_processed)

            # Boxed section for regression results
            with st.container():
                # Menampilkan hasil analisis regresi linier
                st.subheader("📊 Linear Regression Analysis Results")
                st.write(f"Mean Squared Error (MSE): {mse}")
                st.write(f"R-squared: {r2}")
                
                # Menambahkan dua link setelah hasil analisis
                st.markdown(
                    """
                    🔗 [Hasil Analisis](https://tugasbesarakdatkelompok-4-kelasb.streamlit.app/#31a9def2)  
                    🔗 [Visualisasi Data](https://tugasbesarakdatkelompok-4-kelasb.streamlit.app/#28282ccd)
                    """, 
                    unsafe_allow_html=True
                )
                
                # Menampilkan data perbandingan antara nilai aktual dan prediksi
                st.subheader("Actual vs Predicted")
                st.write(comparison.head(10))

                # Add a hover effect explanation for analysis results
                st.subheader("🔍 Penjelasan Hasil Analisis")
                st.write("""
                Hasil analisis regresi linier pada data jumlah penumpang naik dan jumlah kru kapal menunjukkan bahwa model memiliki kinerja yang sangat baik.
                Nilai Mean Squared Error (MSE) sebesar 175,68 menunjukkan bahwa perbedaan rata-rata antara nilai aktual dan prediksi cukup kecil, sehingga model ini dapat dikatakan memiliki akurasi yang tinggi dalam memprediksi jumlah kru kapal berdasarkan jumlah penumpang naik.
                
                Selain itu, nilai R-squared (R²) sebesar 0,999 mengindikasikan bahwa sekitar 99,9% variabilitas jumlah kru kapal dapat dijelaskan oleh jumlah penumpang naik. Hal ini menunjukkan adanya hubungan yang sangat kuat antara jumlah penumpang naik dan jumlah kru kapal, serta kecocokan yang tinggi dari model dengan data yang dianalisis.

                Analisis perbandingan nilai aktual dan prediksi semakin memperkuat hasil ini. Sebagai contoh, pada data dengan jumlah penumpang naik sebesar 2201, prediksi jumlah kru adalah 2159,46, yang sangat mendekati nilai aktual. Hal serupa juga terlihat pada sampel lain, seperti pada penumpang naik sebanyak 425 dengan prediksi 425,48. 

                Secara keseluruhan, model regresi linier ini menunjukkan performa yang sangat baik dan dapat diandalkan untuk memprediksi jumlah kru kapal berdasarkan data penumpang naik.
                """)


                # Visualization Boxed Section
                st.subheader("📊 Visualisasi Analisis Regresi Linier")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X, y, color='blue', label='Actual Data')  # Data points
                ax.plot(X, y_pred, color='red', label='Regression Line')  # Regression line
                ax.set_xlabel('PENUMPANG NAIK')
                ax.set_ylabel('JUMLAH CREW')
                ax.set_title('Linear Regression Visualization')
                ax.legend()
                st.pyplot(fig)

                # Add explanation of visualization inside a container
                with st.container():
                    st.write("""
                        **1. Gambaran Umum**  
                        Grafik di atas menampilkan hasil dari model regresi linier yang diterapkan untuk melihat hubungan antara dua variabel, yaitu:  
                        - **Jumlah Penumpang Naik** (sumbu X)  
                        - **Jumlah Kru Kapal** (sumbu Y)  

                        Grafik ini dibuat untuk menganalisis seberapa besar pengaruh jumlah penumpang yang naik terhadap jumlah kru yang dibutuhkan pada suatu kapal.

                        **2. Interpretasi Garis Regresi**  
                        Garis merah pada grafik adalah garis regresi yang merepresentasikan model prediksi linier terbaik yang dihasilkan dari analisis ini. Garis ini menunjukkan kecenderungan linier positif antara kedua variabel, yang berarti:  
                        - Semakin banyak penumpang yang naik, semakin banyak kru yang diperlukan.  

                        Dalam konteks operasional kapal, ini masuk akal karena peningkatan jumlah penumpang biasanya diikuti dengan peningkatan kebutuhan akan kru untuk menjaga kualitas layanan dan keselamatan.

                        **3. Kesesuaian Model dengan Data (Goodness of Fit)**  
                        Banyak titik data (titik biru) yang berada di sekitar garis regresi merah menunjukkan bahwa model regresi ini mampu memprediksi jumlah kru dengan cukup akurat berdasarkan jumlah penumpang yang naik.  

                        Untuk mengukur kecocokan model, kita bisa melihat dua metrik penting:  
                        - **Koefisien Determinasi (R²):** Model ini memiliki nilai **R² sekitar 0,999**. Ini artinya, model dapat menjelaskan sekitar 99,9% variabilitas dalam jumlah kru berdasarkan jumlah penumpang naik. Semakin tinggi nilai **R²**, semakin baik model dalam menjelaskan hubungan antara variabel dependen (jumlah kru) dan variabel independen (jumlah penumpang).  
                        - **Mean Squared Error (MSE):** Nilai **MSE** untuk model ini adalah sekitar **175,68**. MSE yang rendah menunjukkan bahwa perbedaan rata-rata antara prediksi model dan data aktual cukup kecil. Artinya, model ini tidak memiliki kesalahan yang signifikan dalam memprediksi jumlah kru berdasarkan jumlah penumpang.

                        **4. Outlier dalam Data**  
                        Selain titik-titik yang dekat dengan garis regresi, terdapat beberapa **outlier** yang tampak jauh dari garis. **Outlier** adalah titik data yang tidak sesuai dengan pola umum, atau dalam kasus ini, titik yang tidak cocok dengan prediksi model.  

                        Penyebab adanya **outlier** ini bisa beragam, misalnya:  
                        - **Jenis Kapal yang Berbeda:** Beberapa kapal mungkin memiliki konfigurasi atau kapasitas kru yang berbeda tergantung jenis dan ukuran kapalnya.  
                        - **Rute yang Berbeda:** Rute yang lebih panjang atau lebih berbahaya mungkin memerlukan lebih banyak kru, terlepas dari jumlah penumpang.  
                        - **Kondisi Operasional:** Misalnya, kapal yang beroperasi di musim tertentu atau di bawah peraturan tertentu mungkin memerlukan lebih banyak kru.  

                        Kehadiran **outlier** ini mengindikasikan bahwa selain jumlah penumpang, mungkin ada faktor lain yang mempengaruhi jumlah kru kapal.

                        **5. Kesimpulan Analisis**  
                        Berdasarkan hasil regresi linier ini, kita dapat menyimpulkan beberapa hal:  
                        - Terdapat hubungan **linier positif** yang kuat antara jumlah penumpang naik dan jumlah kru kapal. Hubungan ini dapat digunakan untuk merencanakan atau memperkirakan kebutuhan kru berdasarkan perkiraan jumlah penumpang.  
                        - Tingginya nilai **R²** dan rendahnya nilai **MSE** menunjukkan bahwa model ini cocok untuk prediksi dalam konteks variabel yang dianalisis, yaitu jumlah kru sebagai fungsi dari jumlah penumpang.  
                        - Kehadiran beberapa **outlier** menandakan bahwa meskipun jumlah penumpang adalah indikator utama, faktor lain mungkin juga berperan dalam menentukan jumlah kru yang dibutuhkan.

                        **6. Rekomendasi**  
                        Berdasarkan hasil analisis, beberapa rekomendasi yang dapat diberikan adalah:  
                        - **Perencanaan Kebutuhan Kru:** Model ini dapat digunakan sebagai alat bantu perencanaan kebutuhan kru di kapal. Misalnya, manajer operasional kapal dapat memperkirakan kebutuhan kru dengan cukup akurat berdasarkan prediksi jumlah penumpang.  
                        - **Analisis Lanjutan:** Sebaiknya dilakukan analisis lebih lanjut terhadap **outlier** untuk mengetahui faktor-faktor tambahan yang mungkin berpengaruh terhadap jumlah kru. Misalnya, variabel jenis kapal, rute, atau musim dapat ditambahkan dalam analisis lanjutan.  
                        - **Peningkatan Layanan:** Dengan memperkirakan kebutuhan kru secara lebih akurat, perusahaan kapal dapat memastikan layanan yang optimal bagi penumpang, sekaligus meminimalkan biaya operasional.
                    """)

            # Display the main title with an icon and color
            st.markdown(
                """
                <h1 style="text-align: center; color: #FF5733;">
                    📊 Visualisasi Data
                </h1>
                """, 
                unsafe_allow_html=True
            )
            # Menampilkan Word Cloud untuk lokasi keberangkatan penumpang
            with st.container():
                st.subheader("🗣️ Word Cloud Lokasi Keberangkatan")
                plot_wordcloud(df_processed)

            # Menampilkan Heatmap yang menunjukkan hubungan antara Penumpang Naik, Turun, dan Jumlah Kru per Cabang
            with st.container():
                st.subheader("🌡️ Heatmap Penumpang Naik, Turun, dan Jumlah Kru per Cabang")
                plot_heatmap(df_processed)

            # Box Plot untuk visualisasi distribusi jumlah penumpang per tujuan
            with st.container():
                st.subheader("📍 Box Plot Jumlah Penumpang per Tujuan")
                plot_top_tujuan(df_processed)

            # Menampilkan Bubble Chart untuk menggambarkan hubungan antara jumlah penumpang dan kru per rute
            with st.container():
                st.subheader("💥 Bubble Chart Jumlah Penumpang dan Kru per Rute")
                plot_bubble_chart(df_processed)

            # Menampilkan pie chart untuk menunjukkan proporsi keberangkatan berdasarkan lokasi
            st.subheader('🍴 Proporsi Keberangkatan Berdasarkan Lokasi')
            plot_departure_pie_chart(df_processed)

            # Menampilkan grafik frekuensi tujuan penumpang berdasarkan lokasi
            st.subheader('📍 Top 10 Lokasi Tujuan Berdasarkan Frekuensi')
            plot_tujuan(df_processed)

            # Menampilkan Donut Chart untuk menunjukkan distribusi 5 tujuan terbanyak
            st.subheader('🍩 Distribusi 5 Tujuan Terbanyak dengan Donut Chart')
            plot_donut_chart(df_processed)

            # Menampilkan Stacked Area Chart untuk menggambarkan tren waktu perjalanan penumpang
            st.subheader('📅 Stacked Area Chart (Untuk Menampilkan Tren Waktu)')
            plot_stacked_area_chart(df_processed)

            # Menampilkan rata-rata jumlah penumpang naik berdasarkan cabang
            st.subheader('📊 Rata-Rata Penumpang Naik Berdasarkan Cabang')
            plot_average_passengers_by_branch(df_processed)

            # Menampilkan histogram untuk visualisasi 5 tujuan dengan perjalanan terbanyak
            st.subheader('🗺️ Top 5 Destinasi Berdasarkan Jumlah Perjalanan (Histogram)')
            plot_top_5_destinations_histogram(df_processed)

            # Menampilkan top 5 rute yang paling sering dikunjungi
            st.subheader('🚀 Top 5 Rute Paling Sering Dikunjungi')
            plot_top_5_routes(df_processed)

            # Menampilkan distribusi penumpang naik berdasarkan 5 tujuan teratas
            st.subheader('🌍 Distribusi Penumpang (Naik) berdasarkan 5 Tujuan Teratas')
            plot_top_5_destinations_passenger_distribution(df_processed)

            # Menampilkan bagan lingkaran dengan efek 3D untuk durasi rata-rata perjalanan penumpang ke 5 tujuan teratas
            st.subheader('🌟 5 Tujuan Teratas berdasarkan Durasi Rata-Rata (Bagan Lingkaran Efek 3D)')
            plot_top_5_destinations_avg_duration(df_processed)

            # Menampilkan proporsi total penumpang naik berdasarkan 10 tujuan teratas dalam bentuk pie chart
            st.subheader('🛳️ Proporsi Total Penumpang (Naik) berdasarkan 10 Tujuan Teratas')
            plot_top_10_destinations_passengers(df_processed)

if __name__ == "__main__":
    main()
