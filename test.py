import streamlit as st
import pandas as pd
import numpy as np

from itertools import combinations
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import time

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import os




# Judul = "SISTEM REKOMENDASI PENAWARAN KATEGORI PRODUK PADA PLATFORM ECOMMERCE WARUNG DENGAN METODE CLUSTERING DAN ASOSIASI"
# st.sidebar.title(Judul)


option = st.sidebar.selectbox(
    'Silakan pilih proses:',
    #('Home','Profile Grosir','Data Preparation','Create Association Rules','FP-GROWTH','Chart','Coba Text')
    ('Home','Profile Grosir','Data Preparation','Create Association Rules','Simulasi Rekomendasi')
    #('Home','Profile Grosir','Data Preparation','Create Association Rules')
)

if option == 'Home' or option == '':
    st.write(".") #menampilkan halaman utama
    
   
    #---------- HALAMAN JUDUL NIH.. ------------------------------------------
    # image = "D:\PYTHON_PROJECT\REKOMENDASI\LOGO_TP.png"
    st.write("<h3 style='text-align: center; line-height: 1;'>SISTEM REKOMENDASI PENAWARAN KATEGORI PRODUK PADA PLATFORM ECOMMERCE WARUNG DENGAN METODE CLUSTERING DAN ASOSIASI</h3>", unsafe_allow_html=True)    
    # left_co, cent_co,last_co = st.columns(3)
    #  with cent_co:
    #      st.image(image, caption="..logo..", use_column_width=True)
    st.write("<h4 style='text-align: center;'>PROGRAM STUDI MAGISTER ILMU KOMPUTER&nbsp</h4>", unsafe_allow_html=True)
    #----------------------------------------------------------------------------------------------------------------------AKHIR HALAMAN JUDUL


elif option == 'Profile Grosir':
    st.write("""## Profile Grosir""") #menampilkan judul halaman dataframe
    st.write("Load dari : D:\PYTHON_PROJECT\Data Profile Grosir") 
    dataawal = pd.read_csv(r'D:\PYTHON_PROJECT\ProfileGrosir_ALL_JATIM_2022.csv',sep=';')
    st.dataframe(dataawal.tail())

    # Mengambil kolom yang akan digunakan untuk clustering
    features = dataawal[['umr', 'cust_latitude','cust_longitude']]


    lsOpt_JmlCluster = ["----","10 Cluster"]
    cmbx_JmlCluster = st.selectbox("Cluster Yang akan di Proses :",lsOpt_JmlCluster, index=0)

    # Menampilkan opsi yang dipilih
    if cmbx_JmlCluster != "----":

        # Membuat objek KMeans dengan jumlah cluster yang diinginkan
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(features)

        # # Mendapatkan label dari setiap data
        labels = kmeans.labels_

        st.write("Hasil Clustering Profile Grosir") 
        HASIL = dataawal.copy()
        HASIL.loc[:,'kab'] = dataawal['kab']
        HASIL.loc[:,'wholesaler_id'] = dataawal['wholesaler_id']
        HASIL.loc[:,'cluster'] = kmeans.labels_
        HASIL
        

        HASIL.to_csv("D:\PYTHON_PROJECT\REKOMENDASI\HasilProfileGrosir\hasil_kmeans_10a_cluster.csv",index=False)
        st.write("Export Selesai") 


        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features['umr'], features['cust_latitude'], features['cust_longitude'], c=labels)

        ax.set_xlabel('UMR')
        ax.set_ylabel('cust_latitude')
        ax.set_zlabel('cust_longitude')

        plt.title('Clustering Result')
        #plt.show()
        st.pyplot(fig)


	
elif option == 'Data Preparation':
    
    st.write("""## Data Preparation""") #menampilkan judul halaman dataframe
    st.write("DATA CLUSTERING : (Load dari: D:\PYTHON_PROJECT\REKOMENDASI\HasilProfileGrosir\hasil_kmeans_10a_cluster.csv)") 

    
    #df_cluster = pd.read_csv(r'D:\PYTHON_PROJECT\REKOMENDASI\HasilProfileGrosir\hasil_kmeans_10a_cluster.csv',sep=',')
    df_cluster = pd.read_csv(r'D:\PYTHON_PROJECT\REKOMENDASI\HasilProfileGrosir\hasil_kmeans_10a_cluster.csv',sep=',')
    st.dataframe(df_cluster.tail())

    st.write("DATA TRANSAKSI 12 BULAN :") 
    st.write("Load dari : D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\ jan 2022 - dec 2022") 

    # Daftar file CSV yang akan di-load
    files = [
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_01_jan.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_02_feb.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_03_mar.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_04_apr.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_05_may.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_06_jun.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_07_jul.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_08_aug.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_09_sep.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_10_oct.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_11_nov.csv',
        r'D:\PYTHON_PROJECT\REKOMENDASI\Load_DataTRX\order_12_dec.csv'
    ]

    # Membuat progress bar
    progress_bar = st.progress(0)

    # Load data dari setiap file CSV dan merge (union) menjadi satu dataframe
    df_union = pd.DataFrame()  # Dataframe untuk menyimpan hasil union


    # Load data dari setiap file CSV
    for i, file in enumerate(files):
        # Load data dari file CSV
        df = pd.read_csv(file, sep=',')
        
        # Union (merge) data ke dataframe utama
        df_union = pd.concat([df_union, df], ignore_index=True)

        # Mengupdate nilai progress bar setelah setiap proses load data selesai
        progress_bar.progress((i + 1) / (len(files)+1))

    st.dataframe(df_union.tail())
    progress_bar.progress((i + 1) / len(files))   


    st.write("DATA TRANSAKSI di Join Dengan Cluster") 
    def simulate_long_process():    
        #Join Tabel Trx dan Tabel Cluster
        df_trx_cluster_Apriori = pd.merge(df_union,df_cluster,on='wholesaler_id',how='inner')
        
        #Pemilihan Attribute yang akan digunakan
        data_Apriori = df_trx_cluster_Apriori[['order_no', 'katagori','pcode','cluster']]
        
        # Melakukan operasi SELECT, GROUP BY, dan SUM pada DataFrame
        df_result = data_Apriori.groupby(['order_no', 'katagori','cluster']).agg(order=('pcode', 'count')).reset_index()
        return df_result
    
    with st.spinner('Sedang memproses...'):
        progress_bar = st.progress(0)
        df_result = simulate_long_process()
        progress_bar.progress(100)

    
    st.dataframe(df_result.tail())

    ##eksport hasil ke file csv
    HASIL = df_result.copy()
    HASIL.to_csv("D:\PYTHON_PROJECT\REKOMENDASI\HasilJoin_Cluster\hasil_join_cluster.csv",index=False)
    st.write("Export Selesai") 
   
	
	
	
elif option == 'Create Association Rules':	
	
    # Membuat objek kosong untuk membersihkan layar
    placeholder = st.empty()

    
    df_datajoin = pd.read_csv(r'D:\PYTHON_PROJECT\REKOMENDASI\HasilJoin_Cluster\hasil_join_cluster.csv',sep=',')
    st.dataframe(df_datajoin.tail(200))
    st.write("Jumlah Data : ",len(df_datajoin))
    
    df = df_datajoin.groupby(['cluster']).agg(JmlCluster=('order_no', 'count')).reset_index()
    # st.dataframe(df)



    # Membuat combo box untuk memilih opsi dengan pemilihan ganda
    #cmbxCluster = st.multiselect("Cluster Yang akan di Proses :", ["Opsi 1", "Opsi 2", "Opsi 3"])
    lsOption = ["----","Cluster 0", "Cluster 1", "Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7","Cluster 8","Cluster 9"]
    cmbxCluster = st.selectbox("Cluster Yang akan di Proses :",lsOption, index=0)
    
    # Menampilkan opsi yang dipilih
    if cmbxCluster != "----":
        
        #Filter sesuai dengan Clusternya
        PilihCluster=int(cmbxCluster.split()[-1])
        
        #PilihCluster=Angka
        df_tc = df_datajoin.loc[:,['order_no',
                                      'katagori','order','cluster']]

        df_selected = df_tc[df_tc['cluster'] == PilihCluster]

        # Mengubah DataFrame menjadi list dari transaksi
        transactions = df_selected.groupby('order_no')['katagori'].apply(list).tolist()

        # Melakukan one-hot encoding pada transaksi
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        st.dataframe(df_encoded.tail())

        df_encoded.to_csv("D:\PYTHON_PROJECT\REKOMENDASI\HasilJoin_Cluster\hasil_encode.csv")

        # Membuat kotak teks untuk input data
        txtMinSupport = st.text_input("Masukkan Nilai Minimum Support:")
        txtConfidence = st.text_input("Masukkan Nilai Tracehold Confidence:")
        
        # Menampilkan tombol
        button_clicked = st.button("PROSES ASSOCIATION RULES")

        # Menangani aksi setelah tombol diklik
        if button_clicked:
            xMinSupport = float(txtMinSupport.strip('%'))
            xConfidence = float(txtConfidence.strip('%'))

            # Menggunakan algoritma Apriori untuk mendapatkan frequent patterns
            frequent_patterns= apriori(df_encoded, min_support=xMinSupport/100, use_colnames=True)

            # Mendapatkan aturan asosiasi berdasarkan frequent patterns
            rule_APRIORI = association_rules(frequent_patterns, metric="confidence", min_threshold= xConfidence/100)
            hasil_APRIORI = rule_APRIORI[['antecedents', 'consequents','support','confidence','lift']]

            hasil_APRIORI['antecedents'] = hasil_APRIORI['antecedents'].apply(lambda x: '| '.join(x))
            hasil_APRIORI['consequents'] = hasil_APRIORI['consequents'].apply(lambda x: '| '.join(x))
            hasil_APRIORI['jumlah_antecedents'] = hasil_APRIORI['antecedents'].apply(lambda x: len(x.split('|')))
            hasil_APRIORI['jumlah_consequents'] = hasil_APRIORI['consequents'].apply(lambda x: len(x.split('|')))


            st.dataframe(hasil_APRIORI)
        
            nama_file = f"association_rules_C{PilihCluster}_Sup_{txtMinSupport}_Conf_{txtConfidence}.csv"
            folder_path = r"D:\PYTHON_PROJECT\REKOMENDASI\HasilAsosiasi"
            file_path = os.path.join(folder_path, nama_file)

            hasil_APRIORI.to_csv(file_path,index=False)
            st.write("Jumlah Rules",len(hasil_APRIORI))
            # st.write("Export Selesai") 

            # ---------------------- memfilter hanya yang lift ratio >=1 ---------------------------------------------
            hasil_APRIORI_filtered = hasil_APRIORI[hasil_APRIORI['lift'] >= 1]
            st.dataframe(hasil_APRIORI_filtered)
            st.write("Jumlah Rules Setelah Filter Lift Ratio >= 1 : ",len(hasil_APRIORI_filtered))
            
            # ----------------------- Development PROSES RANGKING
            #st.write("antecedents & consequents 11")
            hasil_ac_11 = (hasil_APRIORI_filtered[(hasil_APRIORI_filtered['jumlah_antecedents'] == 1) & (hasil_APRIORI_filtered['jumlah_consequents'] == 1)])
            #st.dataframe(hasil_ac_11)
            #st.write("Jumlah antecedents & consequents 11",len(hasil_ac_11))
            max_lift_rows_11 = hasil_ac_11.loc[hasil_ac_11.groupby('antecedents')['lift'].idxmax()]
            #st.dataframe(max_lift_rows_11)
            #st.write("Jumlah Max Lift 11",len(max_lift_rows_11))



            #st.write("antecedents & consequents 21")
            hasil_ac_21 = (hasil_APRIORI_filtered[(hasil_APRIORI_filtered['jumlah_antecedents'] == 2) & (hasil_APRIORI_filtered['jumlah_consequents'] == 1)])
            #st.dataframe(hasil_ac_21)
            #st.write("Jumlah antecedents & consequents 21",len(hasil_ac_21))
            max_lift_rows_21 = hasil_ac_21.loc[hasil_ac_21.groupby('antecedents')['lift'].idxmax()]
            #st.dataframe(max_lift_rows_21)
            #st.write("Jumlah Max Lift 21",len(max_lift_rows_21))


            #st.write("antecedents & consequents 31")
            hasil_ac_31 = (hasil_APRIORI_filtered[(hasil_APRIORI_filtered['jumlah_antecedents'] == 3) & (hasil_APRIORI_filtered['jumlah_consequents'] == 1)])
            #st.dataframe(hasil_ac_31)
            #st.write("Jumlah antecedents & consequents 31",len(hasil_ac_31))
            max_lift_rows_31 = hasil_ac_31.loc[hasil_ac_31.groupby('antecedents')['lift'].idxmax()]
            #st.dataframe(max_lift_rows_31)
            #st.write("Jumlah Max Lift 31",len(max_lift_rows_31))
            
            hasil_APRIORI_ranking = pd.concat([max_lift_rows_11, max_lift_rows_21, max_lift_rows_31], ignore_index=True)
            
            #st.dataframe(hasil_APRIORI_ranking)
            st.write("Jumlah Rules Hasil Ranking",len(hasil_APRIORI_ranking))
            
            nama_file = f"mapping_rekomendasi_C{PilihCluster}_Sup_{txtMinSupport}_Conf_{txtConfidence}.csv"
            folder_path = r"D:\PYTHON_PROJECT\REKOMENDASI\HasilMapping"
            file_path = os.path.join(folder_path, nama_file)

            hasil_APRIORI_ranking.to_csv(file_path,index=False)
            st.write("Export Selesai , hasil pada foleder : D:\PYTHON_PROJECT\REKOMENDASI\HasilMapping") 

            hasil_APRIORI_ranking = hasil_APRIORI_ranking.rename(columns={"antecedents": "beli", "consequents": "rekomendasi"}) # Mengganti nama kolom dalam DataFrame
            hasil_APRIORI_ranking_sorted = hasil_APRIORI_ranking.sort_values(by=["jumlah_antecedents", "lift"], ascending=[True, False]) # Menyortir DataFrame berdasarkan kolom "jumlah_anteceents" secara ascending dan "lift" secara descending
            hasil_APRIORI_selected = hasil_APRIORI_ranking_sorted.iloc[:, [0, 1, 2, 3, 4]]
            st.dataframe(hasil_APRIORI_selected)
            
            # --------------------------- Menampilkan tabel dalam format html
            # Mengubah DataFrame menjadi format yang sesuai
            # df_table = hasil_APRIORI_selected.to_html(index=False)
            # Menampilkan tabel HTML menggunakan st.markdown
            # st.markdown(df_table, unsafe_allow_html=True)

            # Menentukan lebar kolom dalam CSS styling
            # table_style = f"""
            # <style>
            #     table th:nth-child(1), table td:nth-child(1) {{ width: 30%; }}
            #     table th:nth-child(2), table td:nth-child(2) {{ width: 30%; }}
            #     table th:nth-child(3), table td:nth-child(3) {{ width: 10%; }}
            #     table th:nth-child(4), table td:nth-child(4) {{ width: 15%; }}
            #     table th:nth-child(5), table td:nth-child(5) {{ width: 15%; }}
            # </style>
            # """
            # # Menampilkan tabel HTML dengan CSS styling
            # st.markdown(table_style + df_table, unsafe_allow_html=True)
            # --------------------------- Menampilkan tabel dalam format html
            



            # ----------------------------------------------------------------------------------------Akhir Development PROSES RANGKING

elif option == 'Simulasi Rekomendasi':
    st.write('Simulasi Rekomendasi Kategori Produk')
    lsOption = ["----","Cluster 0", "Cluster 1", "Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7","Cluster 8","Cluster 9"]
    cmbxCluster = st.selectbox("Cluster Yang akan di Proses :",lsOption, index=0)
    
    # Menampilkan opsi yang dipilih
    if cmbxCluster != "----":
        st.write('Mapping ',cmbxCluster)
        df_hasilmapping = pd.read_csv(r'D:\PYTHON_PROJECT\REKOMENDASI\HasilMapping\mapping_rekomendasi_C1_Sup_8%_Conf_30%.csv',sep=',')
        
        lsOptKat = ["----","Makanan Kemasan", "Makanan Ringan", "Minuman","Perlengkapan Rumah Tangga","Permen dan Coklat","Produk Kopi","Produk Mie Instan","Rokok","Sembako"]
        cmbxKat = st.selectbox("Pilih Kategori Pertama di Keranjang Anda :",lsOptKat, index=0)

        if cmbxKat != "----":
            st.write ('Rekomendasi Kategori Yang di Tawarkan')
            df_1kategori = df_hasilmapping[(df_hasilmapping['jumlah_antecedents'] == 1) & (df_hasilmapping['antecedents'] == cmbxKat) ]
            df_1kategori = df_1kategori.rename(columns={"antecedents": "beli", "consequents": "rekomendasi"}) # Mengganti nama kolom dalam DataFrame
            df_1kategori_selected = df_1kategori.iloc[:, [1, 2, 3, 4]]
            st.dataframe(df_1kategori_selected)

            # row_index = 1 
            # col_name = 'consequents'  # Nama kolom yang ingin Anda ambil nilai dari dataframe
            # selected_value = df_1kategori.loc[row_index, col_name]  
            

        
            # #lsOptKat2 = ["----","Makanan Kemasan", "Makanan Ringan", "Minuman","Perlengkapan Rumah Tangga","Permen dan Coklat","Produk Kopi","Produk Mie Instan","Rokok","Sembako"]
            # cmbxKat2 = st.selectbox("Pilih Kategori Berikutnya di Keranjang Anda :",lsOptKat, index=0)
            # if cmbxKat2 != "----":
            #     st.write ('Rekomendasi Kategori Yang di Tawarkan')
            #     df_2kategori = df_hasilmapping[(df_hasilmapping['jumlah_antecedents'] == 2) & (df_hasilmapping['antecedents'] == cmbxKat2) ]
            #     df_2kategori

            
elif option == 'FP-GROWTH':
    st.write("""## FP-GROWTH""") 
     # Membuat objek kosong untuk membersihkan layar
    placeholder = st.empty()

    
    df_datajoin = pd.read_csv(r'D:\PYTHON_PROJECT\REKOMENDASI\HasilJoin_Cluster\hasil_join_cluster.csv',sep=',')
    st.dataframe(df_datajoin.tail())
    
    df = df_datajoin.groupby(['cluster']).agg(JmlCluster=('order_no', 'count')).reset_index()
    st.dataframe(df)



    # Membuat combo box untuk memilih opsi dengan pemilihan ganda
    #cmbxCluster = st.multiselect("Cluster Yang akan di Proses :", ["Opsi 1", "Opsi 2", "Opsi 3"])
    lsOption = ["----","Cluster 0", "Cluster 1", "Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7","Cluster 8","Cluster 9"]
    cmbxCluster = st.selectbox("Cluster Yang akan di Proses :",lsOption, index=0)
    
    # Menampilkan opsi yang dipilih
    if cmbxCluster != "----":
        
        #Filter sesuai dengan Clusternya
        PilihCluster=int(cmbxCluster.split()[-1])
        
        #PilihCluster=Angka
        df_tc = df_datajoin.loc[:,['order_no',
                                      'katagori','order','cluster']]

        df_selected = df_tc[df_tc['cluster'] == PilihCluster]

        # Mengubah DataFrame menjadi list dari transaksi
        transactions = df_selected.groupby('order_no')['katagori'].apply(list).tolist()

        # Melakukan one-hot encoding pada transaksi
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        st.dataframe(df_encoded.tail())

        df_encoded.to_csv("D:\PYTHON_PROJECT\REKOMENDASI\HasilJoin_Cluster\hasil_encode.csv")

        # Membuat kotak teks untuk input data
        txtMinSupport = st.text_input("Masukkan Nilai Minimum Support:")
        txtConfidence = st.text_input("Masukkan Nilai Tracehold Confidence:")
        
        # Menampilkan tombol
        button_clicked = st.button("PROSES ASSOCIATION RULES")

        # Menangani aksi setelah tombol diklik
        if button_clicked:
            xMinSupport = float(txtMinSupport.strip('%'))
            xConfidence = float(txtConfidence.strip('%'))

            # Menggunakan algoritma Apriori untuk mendapatkan frequent patterns
            frequent_patterns= fpgrowth(df_encoded, min_support=xMinSupport/100, use_colnames=True)
           
            # Mendapatkan aturan asosiasi berdasarkan frequent patterns
            rule_FPG = association_rules(frequent_patterns, metric="confidence", min_threshold= xConfidence/100)
            hasil_FPG = rule_FPG[['antecedents', 'consequents','support','confidence','lift']]

            hasil_FPG['antecedents'] = hasil_FPG['antecedents'].apply(lambda x: '| '.join(x))
            hasil_FPG['consequents'] = hasil_FPG['consequents'].apply(lambda x: '| '.join(x))

            st.dataframe(hasil_FPG)
        
            nama_file = f"association_rules_fpg_C{PilihCluster}_Sup_{txtMinSupport}_Conf_{txtConfidence}.csv"
            folder_path = r"D:\PYTHON_PROJECT\REKOMENDASI\HasilAsosiasi"
            file_path = os.path.join(folder_path, nama_file)

            hasil_FPG.to_csv(file_path,index=False)
            st.write("Jumlah Rules",len(hasil_FPG))
            st.write("Export Selesai") 
	
elif option == 'Chart':
    st.write("""## Draw Charts""") #menampilkan judul halaman 

    #membuat variabel chart data yang berisi data dari dataframe
    #data berupa angka acak yang di-generate menggunakan numpy
    #data terdiri dari 2 kolom dan 20 baris
    chart_data = pd.DataFrame(
        np.random.randn(20,2), 
        columns=['a','b']
    )
    #menampilkan data dalam bentuk chart
    st.line_chart(chart_data)
    #data dalam bentuk tabel
    chart_data

elif option == 'Coba Text':
    st.write("""## Coba Text""") #menampilkan judul halaman 
    # Header dengan ukuran font besar
    st.markdown("# Ini adalah header dengan ukuran font besar")

    # Subheader dengan ukuran font sedang
    st.markdown("## Ini adalah subheader dengan ukuran font sedang")

    # Teks dengan ukuran font kecil
    st.markdown("Ini adalah teks dengan ukuran font kecil")

    # Teks dengan ukuran font khusus
    st.markdown("### Ini adalah teks dengan ukuran font khusus", unsafe_allow_html=True)

    # Menggunakan sintaks markdown
    st.markdown("Ini adalah teks baris pertama\n\nIni adalah teks baris kedua")

    # Menggunakan tag HTML <br> untuk spasi antar baris
    st.markdown("Ini adalah teks baris pertama<br><br>Ini adalah teks baris kedua", unsafe_allow_html=True)

    # Menggunakan tag HTML <p> untuk paragraf
    st.markdown("<p style='margin: 0; padding: 0;'>Ini adalah teks baris pertama</p> <p style='margin: 0; padding: 0;'>Ini adalah teks baris kedua</p>", unsafe_allow_html=True)




    
