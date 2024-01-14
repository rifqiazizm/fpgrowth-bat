import streamlit as st
import pandas as pd  
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


class Strimlit(object):

    def __init__(self):
        self.title = 'Implementasi BAT Algorithm pada FP-Growth'
        self.sub = 'Upload File transaksi anda'


    def ObjectFunction(self,params,df_trans):
        min_support = params[0] 
        min_confidence = params[1]
    
        frequent_patterns= fpgrowth(df_trans, min_support=min_support, use_colnames=True)
        if frequent_patterns.empty:
            return 0
        rule_ = association_rules(frequent_patterns, metric="confidence", min_threshold= min_confidence)
        
        return  rule_['lift'].sum()


    def bat_algorithm(self,num_bats, num_iterations,df_masuk):
        # Inisialisasi parameter Bat Algorithm
        num_dimensions = 2  # Jumlah parameter yang ingin dioptimalkan (min_support, min_confidence)
        lower_bound = np.array([0.6, 0.85])  # Batas bawah untuk setiap parameter
        upper_bound = np.array([1.5, 1.9])  # Batas atas untuk setiap parameter
        loudness = 0.5  # Inisialisasi nilai loudness
        pulse_rate = 0.3  # Inisialisasi nilai pulse rate
        lower_freq = 0.2983
        highest_freq = 0.923

        # Inisialisasi posisi kelelawar secara acak di dalam batas pencarian
        bats = np.random.uniform(lower_bound, upper_bound, (num_bats, num_dimensions))
        # Inisialisasi kecepatan dan posisi awal
        velocities = np.zeros((num_bats, num_dimensions))
        frequencies = np.zeros(num_bats)

        # Inisialisasi posisi terbaik
        best_solution_index = 0
        best_solution = bats[best_solution_index]

        # Iterasi Bat Algorithm
        for _ in range(num_iterations):
            # Update posisi kelelawar
            for i in range(num_bats):
                # Pembaruan frekuensi dan posisi berdasarkan pulsasi
                # if np.random.random() > frequencies[i]:
                    # velocities[i] += (bats[i] - bats.mean(axis=0)) * loudness
                    # frequencies[i] = pulse_rate * (1 - np.exp(-1 * np.random.random()))  # Update frekuensi
                frequencies[i] = lower_freq + (highest_freq - lower_freq) * np.random.uniform(0,1)
                velocities[i] = velocities[i] + (bats[i] - best_solution) * frequencies[i]

                if np.random.random() > pulse_rate:
                    alpha = 0.03
                    bats[i] = best_solution + alpha * np.random.uniform(0,1)



                # Batasan posisi kelelawar dalam batas pencarian
                bats[i] = np.clip(bats[i] + velocities[i], lower_bound, upper_bound)

            # Evaluasi setiap kelelawar
            for i in range(num_bats):
                current_evaluation = self.ObjectFunction(bats[i],df_masuk)
                # Jika nilai evaluasi lebih baik, update posisi terbaik
                if current_evaluation > self.ObjectFunction(best_solution,df_masuk):
                    best_solution = bats[i]

        # Mengembalikan posisi terbaik (parameter yang dioptimalkan)
        return best_solution

        

    def main(self):
        st.set_page_config(page_title="Association Rule ",layout='wide')
        st.header(self.title)
        st.subheader(self.sub)
        st.markdown("""
        <style>
        div.stButton {text-align:center}
        </style>""", unsafe_allow_html=True)


        file_upload = st.file_uploader('')
        st.markdown("***")
        
        
        if file_upload is not None:
            
        
            df = pd.read_csv(file_upload)
            st.dataframe(df.head(10),use_container_width=True)
            st.text("")
            st.toast('File berhasil di upload, sedang convert ke dataframe')

           
        with st.spinner('Creating Graph from dataframe'):

            if st.button("Tampilkan Deskriptif data",type='primary' ):
                jml_uniq = df['id_produk'].nunique()
                jml_transaksi = df['id_produk'].count()
                kategori = df['katagori'].nunique()
                kab = df['kabupaten'].nunique()

                metric1, metric2, metric3,metric4 = st.columns(4)
                metric1.metric("Jml Item", jml_uniq, "item unik")
                metric2.metric("Jml Transaksi", jml_transaksi, "data")
                metric3.metric("kategori", kategori, "jenis item")
                metric4.metric("daerah", kab, "kabupaten")

                col1,col2 = st.columns(2)
                st.text('')
                st.text('')

                with col1:
                    st.bar_chart(data=df.groupby('katagori').sum('qty_sales_order')['qty_sales_order'])
                    # st.write(.plot(kind='bar', figsize=(15,8)))   

                with col2:
                    # st.write(df.groupby('kabupaten').sum('qty_sales_order')['qty_sales_order'].plot(kind='pie', figsize=(15,8),shadow=True,autopct='%1.0f%%'))
                    # st.bar_chart(data=df.groupby('katagori').sum('qty_sales_order')['qty_sales_order'])
                    df_temp = df.groupby('kabupaten').sum('qty_sales_order').reset_index()
                    labels = df_temp['kabupaten'].tolist()
                    sizes = df_temp['qty_sales_order'].tolist()

                    fig1, ax1 = plt.subplots()
                    ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                    st.pyplot(fig1)

        st.markdown("***")
        st.markdown("***")
        st.markdown("""
                        <h2 style='text-align:center;' > 
                            Bat Optimization Algorithm
                        </h2>
                    """,unsafe_allow_html=True)
        st.text('')
        col1,col2 = st.columns(2)

        with col1:
            st.image('bat.jpg',caption="")

        with col2:
            st.markdown(""" 
                            <h4
                                style='text-align:justify;'
                            >
                            The Bat algorithm is a population-based metaheuristics algorithm for solving continuous optimization problems. It’s been used to optimize solutions in cloud computing, feature selection, image processing, and control engineering problems.
                            </h4>
                        """,unsafe_allow_html=True)
        
        df_result = df.groupby(['order_no', 'id_produk']).agg(order=('id_produk', 'count')).reset_index()
        transactions = df.groupby('order_no')['id_produk'].apply(list).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        df_encoded

obj = Strimlit()
obj.main()