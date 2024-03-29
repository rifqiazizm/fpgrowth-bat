import streamlit as st
import pandas as pd  
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
import asyncio 
import time





 


class Strimlit(object):




    def __init__(self):
        self.title = 'Pembentukan Aturan Aosiasi Algoritma FP Growth'
        self.sub = 'Upload File transaksi anda'
        self.df_bat = pd.DataFrame()
        self.df_result = pd.DataFrame()
        self.session_clicked = False
        st.session_state['batExec'] = 0
        self.bat_func = []
        self.df_encoded = pd.DataFrame()
        self.out = {}
        self.best_sol = [0,0]
        self.objFunc = []
        self.frequent_patterns = pd.DataFrame()



    def fpGrowth(self,min_support,min_confidence,df_input ):
        freqP= fpgrowth(df_input, min_support=min_support, use_colnames=True)
        self.frequent_patterns = freqP
        if freqP.empty:
            return 0
        
        rule_ = association_rules(freqP, metric="confidence", min_threshold= min_confidence)
        
        return  rule_





    def ObjectFunction(self,params,df_trans):
        min_support = float(params[0])
        min_confidence = float(params[1])
        
        frequent_patterns= fpgrowth(df_trans, min_support=min_support, use_colnames=True)
        # self.objFunc.append([len(df_trans)])
        # time.sleep(0.1)
        if frequent_patterns.empty:
            return 0
        rule_ = association_rules(frequent_patterns, metric="confidence", min_threshold= min_confidence)
        
        # return  rule_['confidence'].mean() / rule_['support'].mean()
        return rule_['lift'].mean()





    def bat_algorithm(self,num_bats, num_iterations,lower_sup,upper_sup,lower_conf,upper_conf,loudness,pulse_rate,df_masuk,progress):
        # Inisialisasi parameter Bat Algorithm
        num_dimensions = 2  # Jumlah parameter yang ingin dioptimalkan (min_support, min_confidence)
        lower_bound = np.array([float(lower_sup), float(lower_conf)])  # Batas bawah untuk setiap parameter
        upper_bound = np.array([float(upper_sup), float(upper_conf)])  # Batas atas untuk setiap parameter
        loudness = float(loudness)  # Inisialisasi nilai loudness
        pulse_rate = float(pulse_rate)  # Inisialisasi nilai pulse rate
        lower_freq = 0.0002983
        highest_freq = 0.000923

        # Inisialisasi posisi kelelawar secara acak di dalam batas pencarian
        bats = np.random.uniform(lower_bound, upper_bound, (int(num_bats), num_dimensions))
        # Inisialisasi kecepatan dan posisi awal
        velocities = np.zeros((int(num_bats), num_dimensions))
        frequencies = np.zeros(int(num_bats))
        trig=0
        # Inisialisasi posisi terbaik
        # best_solution_index = 0
        best_solution = [0,0]
        outputFunc = np.zeros(int(num_bats))
        proses = 0


        # Iterasi Bat Algorithm
        for _ in range(int(num_iterations)):
            # Update posisi kelelawar
            for i in range(int(num_bats)):
                # Pembaruan frekuensi dan posisi berdasarkan pulsasi
                # if np.random.random() > frequencies[i]:
                    # velocities[i] += (bats[i] - bats.mean(axis=0)) * loudness
                    # frequencies[i] = pulse_rate * (1 - np.exp(-1 * np.random.random()))  # Update frekuensi
                frequencies[i] = lower_freq + (highest_freq - lower_freq) * np.random.random()
                velocities[i] = velocities[i] + (bats[i] - best_solution) * frequencies[i]

                if np.random.random() > float(pulse_rate):
                    alpha = 0.03
                    bats[i] = best_solution + alpha * np.random.uniform(0,1,size=2)
                

                progress.progress(( ((proses+ 1) / (int(num_iterations) * int(num_bats))) )  )
                proses+=1

                # Batasan posisi kelelawar dalam batas pencarian
                batsTemp = np.clip(bats[i] + velocities[i], lower_bound, upper_bound)
                # bats[i] = bats[i] + velocities[i]
                    
                
                Ftemp =  self.ObjectFunction(params=batsTemp,df_trans=df_masuk)
                # time.sleep(0.1)
                print("output dari Ftemp" + str(Ftemp))
                print("output dari outputFunc" + str(outputFunc[i] ))
                print(Ftemp > outputFunc[i] and np.random.rand() < loudness)
                

                if Ftemp > outputFunc[i] and np.random.rand() < loudness:
                    
                    print('HIT DISINI')
                    
                    print('Ftemp > max(outputFunc)' + str(Ftemp > max(outputFunc)))
                    print('max(outputFunc)' + str(max(outputFunc)))
                    if Ftemp > max(outputFunc):
                        best_solution = batsTemp
    
                        print('hit disana')
                        trig+=1
                    outputFunc[i] = Ftemp
                    bats[i] = batsTemp
                print('-----------------------------------------------------------------')
                self.bat_func.append({
                    'val' : bats[i],
                    'trig' : trig,
                    'val2' : batsTemp,
                    'output' : self.ObjectFunction(params=batsTemp,df_trans=df_masuk)
                })

            # Evaluasi setiap kelelawar
            # for i in range(int(num_bats)):
            #     current_evaluation = self.ObjectFunction(bats[i],df_masuk)
            #     self.bat_func.append({
            #         'val' : bats[i],
            #         'output' : current_evaluation
            #     })
            #     # Jika nilai evaluasi lebih baik, update posisi terbaik
            #     if current_evaluation > self.ObjectFunction(best_solution,df_masuk):
            #         best_solution = bats[i]

        # Mengembalikan posisi terbaik (parameter yang dioptimalkan)
        print(outputFunc)
        return {
                    'best solution' : best_solution,
                    'parameter' : [num_bats,num_iterations,lower_bound,upper_bound,loudness,pulse_rate],
                    'Nilai Kelelawar' : bats ,
                }
    



    def onClickBtn(self):
        self.session_clicked = True


    def main(self):
        # df=pd.DataFrame()
        
        st.set_page_config(page_title="Association Rule ",layout='wide')
        
        st.markdown("""
            <style>
            div.stButton {text-align:center}
            </style>""", unsafe_allow_html=True)

        st.markdown("""
                        <h2 style='text-align:center;' > 
                            Implementasi BAT Algorithm pada FP-Growth
                        </h2>
                    """,unsafe_allow_html=True)
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')

        file_upload = st.file_uploader('Upload File Transaksi Anda')
        # st.markdown("***")
        st.text('')
        
        
        if file_upload is not None:
            
            
            st.text('')
            st.text('')
            df = pd.read_csv(file_upload)
            st.dataframe(df.head(10),use_container_width=True)
            st.text("")
            st.toast('File berhasil terdeteksi, sedang convert ke dataframe')

           
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


                cekUniq = df[df[['order_no','order_date','amount_sales_order','id_produk','qty_sales_order']].duplicated(keep=False)]
                cekTrx = df[df['qty_sales_order'] == 0 ]


                st.warning('Jumlah Data duplikat ='+ str(len(cekUniq)))
                st.warning('Jumlah Data Transaksi Tidak Bernilai = ' +str(len(cekTrx)))


                with st.spinner("Data sedang di cleaning"):
                    if st.button("Bersihkan Data ",type='secondary'):
                        df.drop_duplicates(inplace=True,keep='first')
                        st.toast("Data Duplikat Berhasil Dihapus")
                

        st.markdown("***")
       
        st.markdown("""
                        <h2 style='text-align:center;' > 
                            Bat Optimization Algorithm
                        </h2>
                        
                    """,unsafe_allow_html=True)
        st.markdown("***")
        
        st.text('')

        
        

        
        st.markdown(""" 
                            <h4
                                style='text-align:justify;'
                            >
                            The Bat algorithm is a population-based metaheuristics algorithm for solving continuous optimization problems. It’s been used to optimize solutions in cloud computing, feature selection, image processing, and control engineering problems.
                            </h4>
                        """,unsafe_allow_html=True)
        st.text('')
        st.text('')
        col1,col2 = st.columns(2)

        with col1:
            st.image('bat2.jpg',caption="")

        with col2:
            st.image('bat3.jpg',caption="")
            
        st.text('')
        st.text('')

        kol1,kol2,kol3 = st.columns(3)

        with kol2:
            st.button('Ambil Data untuk BAT',type='primary',key='button1',use_container_width=True)
            st.text('')
            st.text('')
        # st.write(st.session_state.button)
        with st.spinner('Transforming data from pandas'):
            if st.session_state.button1:
                # st.text('berhasilk')
                self.df_result = df.groupby(['order_no', 'id_produk']).agg(order=('id_produk', 'count')).reset_index()
                transactions = df.groupby('order_no')['id_produk'].apply(list).tolist()
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                self.df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                st.session_state['df_encoded'] = pd.DataFrame(te_ary, columns=te.columns_)
                st.toast('Data Berhasil diambil',)
                

                st.text('')
                st.text('')
                st.text('')

        col1,col2 = st.columns(2)

        with col2: 
            st.text('data per item')
            st.dataframe(self.df_result.head() ,use_container_width=True)
        with col1:
            st.text('data item hasil encode')
            st.dataframe(self.df_encoded.head() ,use_container_width=True)


                
        st.text('')
        st.markdown("***")
        
                
        lol1,lol2 = st.columns(2)
        
        st.session_state['Exec'] = False
        with lol1:
            with st.form(key='bat'):
                bat = st.text_input("Input jml kelelawar")
                iterat = st.text_input("Input jml iterasi")
                minSupp = st.text_input("Input Minimal Support")
                maxSupp = st.text_input('Input Max Support')
                minConf = st.text_input('Input Minimal Confidence ')
                maxConf = st.text_input('Input Max Confidence ')
                loudness= st.text_input('Input Loudness ')
                pulse   = st.text_input('Input Pulse Rate')
                button = st.form_submit_button(label='Eksekusi BAT algorithm',type='primary')
                # st.write(st.session_state['df_encoded'])
                prog = st.progress(value=0, text="progress eksekusi BAT")
                if button:
                    st.text('okeee')
                    with st.spinner("Mengeksekusi algorithma BAT"):
                        transactions = df.groupby('order_no')['id_produk'].apply(list).tolist()
                        te = TransactionEncoder()
                        te_ary = te.fit(transactions).transform(transactions)
                        df_input = pd.DataFrame(te_ary, columns=te.columns_)
                        st.toast('executing bat with '+str(len(df_input)))
                        self.out  = self.bat_algorithm(
                            df_masuk=df_input,
                            num_bats=bat,
                            num_iterations=iterat,
                            lower_sup=minSupp,
                            upper_sup=maxSupp,
                            lower_conf=minConf,
                            upper_conf=maxConf,
                            loudness=loudness,
                            pulse_rate=pulse,
                            progress = prog
                        )
                 
                        

        with lol2:
            st.subheader('Nilai Optimal: ')
            if len(self.out) > 1:
                st.write(self.out) 
                st.success('Bat Algorithm Sudah dieksekusi' )  
            else:
                st.warning('Bat algorithm belum dieksekusi')

            xSupport    =  [a['val2'][0] for a in self.bat_func]
            xConfidence =  [a['val2'][1] for a in self.bat_func]
            yOutput = [a['output'] for a in self.bat_func]


            tt1,tt2 = st.columns(2)

            
            dfOut = pd.DataFrame({
                'support' : xSupport,
                'confidence' : xConfidence,
                'output' : yOutput

            })
            
            with tt1:

                st.markdown(" #### Grafik nilai support terhadap obj function")
                st.line_chart(dfOut,x='support',y='output',use_container_width=True)     

            with tt2:
                st.markdown(" #### Grafik nilai Confidence terhadap obj function")
                st.line_chart(dfOut,x='confidence',y='output',use_container_width=True)          
            
                 

            st.text("")
            st.text("")  

            
       
        
        st.markdown("***")
       
        st.markdown("""
                        <h2 style='text-align:center;' > 
                            FP-Growth Algorithm
                        </h2>
                        
                    """,unsafe_allow_html=True)
        st.markdown("***") 

        # kolom1,kolom2 = st.columns(2)
        # st.text("")
        # st.text("")  
        

        # with kolom1:
        #     st.image('fp-growth.png',caption="",use_column_width=True)


        # with kolom2:
        #     st.text("")
        #     st.text("")  
        st.text("")
        st.text("")
        # st.text("")  
        # st.text("")
        # st.text("")  
        #     st.markdown(' ##### The FP-Growth Algorithm is an alternative way to find frequent item sets without using candidate generations, thus improving performance. For so much, it uses a divide-and-conquer strategy. The core of this method is the usage of a special data structure named frequent-pattern tree (FP-tree), which retains the item set association information.')

        

        
        
     
        fp = st.form('fp-growth')
        minSupport = fp.text_input('Minimal Support FP Growth')
        minConfidence = fp.text_input('Minimal Confidence FP Growth')
        st.text("")
        submitFP = fp.form_submit_button('Eksekusi FP Growth' , type='primary')
            

  
            
      
        st.subheader('Hasil Algoritma FP Growth: ')
        outFP = pd.DataFrame()
        if submitFP:
            with st.spinner('Eksekusi FP Growth'):
                transactions = df.groupby('order_no')['id_produk'].apply(list).tolist()
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                self.df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                st.toast('FP Growth Sedang Dieksekusi dengan dataframe' +str(len(self.df_encoded)))
                outFP = self.fpGrowth(float(minSupport), float(minConfidence), self.df_encoded).copy()

                    # viz = st.bar_chart(self.frequent_patterns.sort_values(by='support',ascending=False).head(10))
                self.frequent_patterns['itemsets'] = self.frequent_patterns['itemsets'].apply(lambda ss: [x for x in ss])
                st.success('Association Rule Berhasil dibentuk!')


        koll1,koll2 = st.columns(2)
        with koll1:
                
            st.write(self.frequent_patterns,use_container_width=True)
        
        with koll2:                   
            st.write(outFP,use_container_width=True)

            # st.write(outFP.columns)

        st.text("")
        st.text("")  
        st.text("")
        st.text("")   



        if len(self.df_encoded) > 0 :
            qtySales = df[['area','id_produk']].groupby('id_produk').count().reset_index().copy()
            lowSales =  qtySales[qtySales['area'] == 1]['id_produk'].tolist()
            highSales = qtySales.sort_values(by='area',ascending=False)['id_produk'].head(50).tolist()

            if st.button("Tampilkan item Consequent dengan Penjualan Terendah",type='primary'):

                outFP['conseqConvert'] = outFP['consequents'].apply(lambda x: [a for a in x ])
                st.write(outFP[ outFP['conseqConvert'].apply(lambda x: True if True in [ True if a in lowSales else False for a in x ] else False) ])
                
            if st.button("Tampilkan item Consequent dengan Penjualan Tertinggi",type='primary'):
                st.write(outFP[ outFP['conseqConvert'].apply(lambda x: True if True in [ True if a in highSales else False for a in x ] else False) ])
            
                    
        
                    


st.session_state['df_encoded'] = pd.DataFrame()
obj = Strimlit()
obj.main()



