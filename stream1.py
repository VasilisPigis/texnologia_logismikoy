import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Ρύθμιση αισθητικής των γραφημάτων
sns.set_theme(style="darkgrid", palette="pastel")

# Τίτλος εφαρμογής
st.markdown("# 📊 Μοριακή Βιολογία - Ανάλυση Δεδομένων")
st.markdown("Ανεβάστε ένα αρχείο CSV και εξερευνήστε τα δεδομένα!")

# Upload αρχείου CSV
uploaded_file = st.file_uploader("📂 Επιλέξτε αρχείο CSV ή Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')  # Χρήση openpyxl για ανάγνωση Excel
    
    st.write("## 🔍 Προεπισκόπηση δεδομένων:")
    st.dataframe(df.head())
    
    # Sidebar επιλογές
    st.sidebar.header("⚙️ Ρυθμίσεις")
    column = st.sidebar.selectbox("📌 Επιλέξτε στήλη για στατιστική ανάλυση", df.columns)
    num_components = st.sidebar.slider("🎚️ Επιλέξτε αριθμό συνιστωσών PCA", 1, min(df.shape[1], 10), 2)
    
    # Χρήση Tabs για καλύτερη οργάνωση
    tab1, tab2, tab3 = st.tabs(["📈 Στατιστικά", "📊 Οπτικοποίηση", "🧬 PCA Ανάλυση"])
    
    with tab1:
        st.write("### 📋 Βασικά στατιστικά στοιχεία")
        st.write(df[column].describe())
    
    with tab2:
        st.write("### 📊 Ιστόγραμμα της επιλεγμένης στήλης")
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.write("### 🟦 Boxplot για ανίχνευση outliers")
        fig, ax = plt.subplots()
        sns.boxplot(y=df[column], ax=ax)
        st.pyplot(fig)
        
        if df.select_dtypes(include=['float64', 'int64']).shape[1] >= 2:
            with st.expander("📌 Δείτε το Heatmap Συσχετίσεων"):
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
    
    with tab3:
        if df.select_dtypes(include=['float64', 'int64']).shape[1] < 2:
            st.error("Δεν υπάρχουν αρκετά αριθμητικά δεδομένα για PCA!")
        else:
            st.write("### ⚡ PCA Ανάλυση")
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
            pca = PCA(n_components=num_components)
            principal_components = pca.fit_transform(df_scaled)
            pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(num_components)])
            st.dataframe(pca_df.head())
            
            st.write("### 📍 PCA Διάγραμμα")
            fig, ax = plt.subplots()
            sns.scatterplot(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)

# Sidebar - Πληροφορίες Ομάδας
st.sidebar.title("👥 Πληροφορίες Ομάδας")
st.sidebar.write("🔹 Όνομα 1 - Συνεισφορά")
st.sidebar.write("🔹 Όνομα 2 - Συνεισφορά")
st.sidebar.write("🔹 Όνομα 3 - Συνεισφορά")
