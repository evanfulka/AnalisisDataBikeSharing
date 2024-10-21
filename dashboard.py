import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt 
from sklearn.preprocessing import MinMaxScaler

# Judul Dashboard
st.title("Analisis Data pada Bike Sharing Dataset Tahun 2011-2012")

# Tab untuk berbagai bagian
tabs = st.tabs(["Identitas", "Pertanyaan Bisnis", "EDA", "Visualization & Explanatory Analysis", "Analisis Lanjutan", "Kesimpulan"])

# Tab 1: Identitas
with tabs[0]:
    st.header("Identitas")
    st.write(""" 
        **Nama:** Evan Fulka Bima \n
        **Email:** efulkabima0407@gmail.com \n
        **ID Dicoding:** evan_fulka
    """)

# Tab 2: Pertanyaan Bisnis
with tabs[1]:
    st.header("Pertanyaan Bisnis")
    st.markdown(""" 
    1. **Apa pola penyewaan sepeda berdasarkan waktu dalam sehari (misalnya, pagi, siang, malam) dan apakah pola tersebut berbeda antara pengguna kasual dan terdaftar?**
    2. **Bagaimana hubungan antara kondisi cuaca seperti hujan ringan atau hujan lebat dengan tingkat penyewaan sepeda?**
    """)

# Gathering Data
df_day = pd.read_csv('data/day.csv')
df_hour = pd.read_csv('data/hour.csv')

#Assessing Data
day_missing_values = df_day.isnull().sum()
hour_missing_values = df_hour.isnull().sum()

day_duplicates = df_day.duplicated().sum()
hour_duplicates = df_hour.duplicated().sum()

day_statistics = df_day.describe()
hour_statistics = df_hour.describe()

#Cleaning Data
scaler = MinMaxScaler()

cols_to_normalize_day = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
cols_to_normalize_hour = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

df_day[cols_to_normalize_day] = scaler.fit_transform(df_day[cols_to_normalize_day])
df_hour[cols_to_normalize_hour] = scaler.fit_transform(df_hour[cols_to_normalize_hour])

# Tab 3: EDA
with tabs[2]:
    # Exploratory Data Analysis (EDA)
    st.header("Exploratory Data Analysis (EDA)")

    # Bagian 1: Visualisasi Distribusi Variabel Numerik
    st.subheader("Visualisasi Distribusi Variabel Numerik")

    # Visualisasi distribusi variabel 'cnt' di df_day dan df_hour
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(df_day['cnt'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Distribusi Jumlah Rental Harian')

    sns.histplot(df_hour['cnt'], bins=30, kde=True, ax=ax[1])
    ax[1].set_title('Distribusi Jumlah Rental Per Jam')

    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini menunjukkan distribusi jumlah rental sepeda untuk data harian dan per jam. Pada plot kiri, distribusi jumlah rental harian cenderung berbentuk bimodal, dengan dua puncak di sekitar nilai tengah distribusi yang menunjukkan dua kelompok signifikan dari hari-hari dengan volume rental yang lebih tinggi dan lebih rendah. Sementara itu, distribusi per jam di plot kanan menunjukkan distribusi yang sangat miring ke kiri, di mana sebagian besar rental terjadi pada rentang yang lebih rendah.")

    # Visualisasi variabel numerik lain seperti 'temp', 'atemp', 'hum', dan 'windspeed'
    fig, ax = plt.subplots(figsize=(10, 8))
    df_day[['temp', 'atemp', 'hum', 'windspeed']].hist(bins=20, ax=ax)
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini menampilkan distribusi variabel numerik lainnya seperti temperatur, kelembaban, dan kecepatan angin. Distribusi temperatur (temp dan atemp) menunjukkan pola yang relatif merata dengan beberapa variasi kecil, sementara kelembaban (hum) cenderung berkumpul di kisaran menengah hingga tinggi. Distribusi kecepatan angin (windspeed) cenderung miring ke kiri, dengan banyak data terkonsentrasi pada kecepatan angin yang rendah.")

    # Bagian 2: Visualisasi Hubungan Antar Variabel
    st.subheader("Visualisasi Hubungan Antar Variabel")

    # Filter hanya kolom numerik
    numerical_columns = df_day.select_dtypes(include=['float64', 'int64'])

    # Korelasi antara variabel numerik
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Korelasi Variabel di Dataset Day')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Pada heatmap ini, kita dapat melihat bahwa variabel suhu (temp dan atemp) memiliki korelasi yang cukup kuat dengan jumlah rental sepeda (cnt), sementara kecepatan angin (windspeed) menunjukkan korelasi negatif yang lemah. Di sisi lain, jumlah pengguna terdaftar (registered) sangat berkorelasi dengan jumlah total rental, jauh lebih tinggi dibandingkan dengan pengguna tidak terdaftar (casual).")

    # Untuk df_hour juga bisa dilakukan hal yang sama
    numerical_columns_hour = df_hour.select_dtypes(include=['float64', 'int64'])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_columns_hour.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Korelasi Variabel di Dataset Hour')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Heatmap dari dataset per jam menunjukkan pola serupa, dengan korelasi kuat antara suhu dan jumlah rental. Namun, ada variasi tambahan pada variabel waktu (hr) yang menunjukkan hubungan penting dengan jumlah rental sepeda, yang masuk akal mengingat pola penggunaan sepeda kemungkinan bervariasi sepanjang hari.")

    # Bagian 3: Analisis Kategori
    st.subheader("Analisis Kategori")

    # Analisis jumlah rental berdasarkan season dan weather
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='season', y='cnt', data=df_day, ax=ax)
    plt.title('Jumlah Rental Berdasarkan Season')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Pada plot ini, terlihat bahwa musim ketiga (musim panas) memiliki median jumlah rental tertinggi dibandingkan musim lain, sementara musim pertama (musim dingin) menunjukkan jumlah rental yang paling rendah.")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='weathersit', y='cnt', data=df_hour, ax=ax)
    plt.title('Jumlah Rental Berdasarkan Weather Situation')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Plot ini menunjukkan bahwa kondisi cuaca yang lebih buruk (kategori 3 dan 4) terkait dengan jumlah rental yang jauh lebih rendah dibandingkan cuaca yang lebih baik.")

    # Analisis jumlah rental berdasarkan workingday dan holiday
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='workingday', y='cnt', data=df_day, ax=ax)
    plt.title('Jumlah Rental Berdasarkan Working Day')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Pada plot ini, hari kerja (workingday) tidak menunjukkan perbedaan signifikan terhadap jumlah rental dibandingkan hari non-kerja.")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='holiday', y='cnt', data=df_hour, ax=ax)
    plt.title('Jumlah Rental Berdasarkan Holiday')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Plot ini memperlihatkan bahwa pada hari libur (holiday), jumlah rental cenderung lebih rendah dibandingkan hari biasa, yang mungkin mencerminkan pola penggunaan sepeda yang lebih rendah saat liburan.")

    # Bagian 4: Tren Waktu
    st.subheader("Tren Waktu")

    # Visualisasi tren rental per bulan
    fig, ax = plt.subplots(figsize=(10, 6))
    df_day.groupby('mnth')['cnt'].mean().plot(kind='bar', ax=ax)
    plt.title('Rata-rata Jumlah Rental Per Bulan')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini menggambarkan tren rata-rata jumlah rental sepeda per bulan. Terlihat bahwa jumlah rental mencapai puncaknya pada bulan Mei hingga September, yang mencerminkan meningkatnya aktivitas bersepeda di musim semi dan musim panas. Sebaliknya, bulan Desember dan Januari menunjukkan angka rental yang paling rendah, kemungkinan karena cuaca yang kurang mendukung aktivitas luar ruangan.")

    # Visualisasi tren rental per jam
    fig, ax = plt.subplots(figsize=(10, 6))
    df_hour.groupby('hr')['cnt'].mean().plot(kind='bar', ax=ax)
    plt.title('Rata-rata Jumlah Rental Per Jam')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Pada visualisasi ini, tren jumlah rental per jam memperlihatkan puncak signifikan pada jam 8 pagi dan jam 5 sore, yang kemungkinan besar terkait dengan jam sibuk komuter. Selain itu, aktivitas rental juga terlihat menurun pada jam-jam malam dan dini hari, mencerminkan sedikitnya pengguna yang menggunakan sepeda di luar jam kerja.")

# Tab 4: Visualization & Explanatory Analysis
with tabs[3]:
    # Visualization & Explanatory Analysis
    st.header("Visualization & Explanatory Analysis")

    # Bagian 1: Pertanyaan 1 - Pola Penyewaan Sepeda Berdasarkan Waktu
    st.subheader("Pertanyaan 1: Apa pola penyewaan sepeda berdasarkan waktu dalam sehari dan apakah pola tersebut berbeda antara pengguna kasual dan terdaftar?")

    # Membuat kategori waktu berdasarkan jam
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return 'Pagi'
        elif 12 <= hour < 18:
            return 'Siang'
        elif 18 <= hour < 24:
            return 'Sore'
        else:
            return 'Malam'

    df_hour['waktu'] = df_hour['hr'].apply(categorize_hour)

    # Plot pola penyewaan antara pengguna kasual dan terdaftar
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='waktu', y='casual', data=df_hour, ax=ax)
    ax.set_title('Pola Penyewaan Sepeda oleh Pengguna Kasual Berdasarkan Waktu Sehari')
    st.pyplot(fig)  # Menyertakan objek figura

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='waktu', y='registered', data=df_hour, ax=ax)
    ax.set_title('Pola Penyewaan Sepeda oleh Pengguna Terdaftar Berdasarkan Waktu Sehari')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi di atas menunjukkan perbedaan pola penyewaan sepeda antara pengguna kasual dan pengguna terdaftar berdasarkan waktu dalam sehari. Untuk pengguna kasual, puncak penyewaan terjadi pada siang hari, dengan distribusi yang lebih lebar dan beberapa outlier pada sore hari, sedangkan pada malam hari penyewaan sangat sedikit. Sebaliknya, pengguna terdaftar menunjukkan pola yang lebih seragam dengan puncak yang lebih tinggi pada pagi hari, yang kemungkinan besar mencerminkan pengguna yang bersepeda untuk pergi ke tempat kerja, dan sedikit menurun pada sore hari. Pada malam hari, penyewaan oleh pengguna terdaftar juga sangat rendah, mirip dengan pengguna kasual. Pola ini mencerminkan bagaimana pengguna terdaftar cenderung memanfaatkan sepeda untuk kebutuhan komuter, sementara pengguna kasual lebih banyak menggunakan sepeda di siang hari untuk aktivitas rekreasi.")

    # Bagian 2: Pertanyaan 2 - Hubungan Antara Kondisi Cuaca dan Tingkat Penyewaan Sepeda
    st.subheader("Pertanyaan 2: Bagaimana hubungan antara kondisi cuaca seperti hujan ringan atau hujan lebat dengan tingkat penyewaan sepeda?")

    # Mengganti kode weathersit dengan deskripsi cuaca
    weather_labels = {
        1: 'Cerah/Berawan',
        2: 'Berkabut/Berawan',
        3: 'Hujan Ringan/Salju Ringan',
        4: 'Hujan Lebat/Badai'
    }

    df_hour['cuaca'] = df_hour['weathersit'].map(weather_labels)

    # Visualisasi penyewaan berdasarkan cuaca
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='cuaca', y='cnt', data=df_hour, ax=ax)
    ax.set_title('Hubungan Antara Kondisi Cuaca dan Tingkat Penyewaan Sepeda')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini menunjukkan hubungan antara kondisi cuaca dan tingkat penyewaan sepeda. Dari boxplot terlihat bahwa cuaca cerah atau berawan menghasilkan tingkat penyewaan sepeda yang lebih tinggi, dengan distribusi yang lebih lebar dan median yang lebih tinggi dibandingkan kondisi cuaca lainnya. Ketika cuaca mulai berkabut atau berawan, jumlah penyewaan sepeda sedikit menurun. Penyewaan sepeda berkurang drastis saat terjadi hujan ringan atau salju ringan, dan kondisi terburuk adalah hujan lebat atau badai, di mana jumlah penyewaan sepeda berada pada tingkat yang paling rendah. Ini menunjukkan bahwa semakin buruk kondisi cuaca, semakin sedikit orang yang menyewa sepeda, dengan penurunan yang signifikan saat cuaca berubah menjadi ekstrem.")

# Tab 5: Analisis Lanjutan
with tabs[4]:
    # Analisis Lanjutan (Clustering Menggunakan Teknik Binning)
    st.header("Analisis Lanjutan (Clustering Menggunakan Teknik Binning)")

    # Bagian 1: Binning untuk Jumlah Penyewaan Sepeda
    st.subheader("Binning untuk Jumlah Penyewaan Sepeda")

    # Melihat distribusi dari data cnt
    st.write("Deskripsi Jumlah Penyewaan (cnt):")
    st.write(df_hour['cnt'].describe())

    # Membuat bins manual yang sesuai dengan distribusi data
    bins_manual = [0, 0.15, 0.3, 1]  # Sesuaikan dengan distribusi cnt
    bin_labels = ['Rendah', 'Sedang', 'Tinggi']

    # Membagi data 'cnt' ke dalam beberapa kategori menggunakan binning manual
    df_hour['cnt_binned'] = pd.cut(df_hour['cnt'], bins=bins_manual, labels=bin_labels, include_lowest=True)

    # Visualisasi distribusi kategori binning
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='cnt_binned', data=df_hour, ax=ax)
    ax.set_title('Distribusi Jumlah Rental Berdasarkan Kategori Binning (Manual)')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini menunjukkan distribusi jumlah rental sepeda yang dikelompokkan ke dalam tiga kategori menggunakan metode binning manual, yaitu rendah, sedang, dan tinggi. Kategori rendah mendominasi distribusi dengan jumlah penyewaan yang jauh lebih tinggi dibandingkan dengan kategori sedang dan tinggi. Berdasarkan hasil statistik cnt, sebagian besar nilai berada di bawah median, yang mengarah pada klasifikasi lebih banyak data ke dalam kategori rendah. Hanya sebagian kecil data yang termasuk dalam kategori sedang dan tinggi, menunjukkan bahwa jumlah rental yang tinggi adalah kejadian yang lebih jarang terjadi, sementara rental dengan jumlah rendah lebih umum dalam dataset ini.")

    # Bagian 2: Binning untuk Suhu
    st.subheader("Binning untuk Suhu")

    # Membagi suhu 'temp' ke dalam beberapa kategori menggunakan binning
    temp_labels = ['Dingin', 'Sejuk', 'Hangat']
    df_hour['temp_binned'] = pd.cut(df_hour['temp'], bins=[0, 0.3, 0.6, df_hour['temp'].max()], labels=temp_labels, include_lowest=True)

    # Visualisasi distribusi kategori suhu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='temp_binned', data=df_hour, ax=ax)
    ax.set_title('Distribusi Suhu Berdasarkan Kategori Binning')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini menunjukkan distribusi suhu yang telah dikelompokkan ke dalam tiga kategori: dingin, sejuk, dan hangat. Kategori Sejuk mendominasi distribusi dengan jumlah observasi terbanyak, menunjukkan bahwa sebagian besar suhu dalam dataset berada pada kisaran sedang. Kategori Hangat juga cukup banyak terwakili, meskipun tidak sebanyak Sejuk. Sementara itu, kategori Dingin memiliki jumlah observasi paling sedikit, menunjukkan bahwa suhu rendah jarang terjadi di dataset ini. Secara keseluruhan, grafik ini mengilustrasikan bahwa kondisi suhu yang lebih moderat lebih umum dibandingkan suhu ekstrem dingin atau panas.")

    # Bagian 3: Hubungan Jumlah Rental dan Suhu
    st.subheader("Hubungan Jumlah Rental dan Suhu")

    # Visualisasi hubungan antara kategori jumlah rental dan kategori suhu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='cnt_binned', hue='temp_binned', data=df_hour, ax=ax)
    ax.set_title('Hubungan Antara Kategori Jumlah Rental dan Suhu')
    st.pyplot(fig)  # Menyertakan objek figura

    st.write("Visualisasi ini memperlihatkan hubungan antara kategori jumlah rental sepeda dan suhu. Untuk kategori jumlah rental yang rendah, mayoritas terjadi pada suhu Sejuk, diikuti oleh suhu Dingin dan Hangat. Namun, saat jumlah rental meningkat ke kategori Sedang dan Tinggi, suhu Hangat mulai lebih dominan, terutama pada kategori rental tinggi, yang menunjukkan bahwa suhu yang lebih hangat cenderung mendorong lebih banyak penyewaan sepeda. Sebaliknya, suhu Dingin secara konsisten mendukung jumlah rental yang lebih rendah, dan hampir tidak ada kontribusi suhu dingin untuk kategori rental yang tinggi. Ini menunjukkan adanya keterkaitan antara suhu yang lebih hangat dengan peningkatan aktivitas bersepeda.")

# Tab 6: Kesimpulan
with tabs[5]:
    # Bagian Conclusion
    st.header("Kesimpulan")

    # Kesimpulan Pertanyaan 1
    st.subheader("Kesimpulan Pertanyaan 1")
    st.write("Pengguna terdaftar cenderung menggunakan sepeda pada jam-jam sibuk pagi dan sore, mengindikasikan bahwa mereka lebih memanfaatkan sepeda sebagai sarana transportasi harian, seperti perjalanan ke dan dari tempat kerja. Sebaliknya, pengguna kasual lebih banyak menyewa sepeda di siang dan sore hari, kemungkinan besar untuk aktivitas santai atau rekreasi. Pola ini menunjukkan perbedaan yang signifikan antara kedua kelompok dalam cara mereka menggunakan layanan penyewaan sepeda.")

    # Kesimpulan Pertanyaan 2
    st.subheader("Kesimpulan Pertanyaan 2")
    st.write("Kondisi cuaca yang baik, seperti cerah atau sedikit berawan, sangat mendukung tingginya tingkat penyewaan sepeda. Namun, saat cuaca memburuk, terutama saat hujan ringan hingga hujan lebat, jumlah penyewaan sepeda menurun tajam. Ini menunjukkan bahwa penyewaan sepeda sangat sensitif terhadap perubahan cuaca, dan layanan ini lebih efektif ketika cuaca mendukung aktivitas luar ruangan.")

    # Kesimpulan Analisis Lanjutan
    st.subheader("Kesimpulan Analisis Lanjutan")
    st.write("Suhu memainkan peran penting dalam memengaruhi aktivitas penyewaan sepeda. Suhu yang hangat secara jelas mendorong lebih banyak penyewaan, menunjukkan bahwa cuaca yang nyaman memotivasi orang untuk menggunakan sepeda, baik untuk transportasi maupun rekreasi. Sementara itu, pada suhu sejuk, aktivitas penyewaan tersebar lebih merata, mencerminkan kondisi cuaca yang seimbang dan cocok untuk bersepeda. Sebaliknya, pada suhu dingin, terjadi penurunan drastis dalam jumlah penyewaan, yang menegaskan bahwa cuaca dingin cenderung menghalangi orang untuk bersepeda, terutama dalam kategori dengan volume penyewaan tinggi. Hal ini memberikan wawasan bahwa perencanaan operasional dan pemasaran layanan berbagi sepeda harus memperhitungkan faktor cuaca, terutama suhu, untuk mengoptimalkan penggunaan sepeda.")