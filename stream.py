import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os

# Εγκατάσταση εξαρτήσεων αν δεν υπάρχουν
try:
    import streamlit
    import pandas
    import matplotlib
    import seaborn
    import sklearn
except ModuleNotFoundError:
    os.system(f"{sys.executable} -m pip install streamlit pandas matplotlib seaborn scikit-learn")
    os.execl(sys.executable, sys.executable, *sys.argv)

# Τίτλος εφαρμογής
st.title("Μοριακή Βιολογία - Ανάλυση Δεδομένων")

# Upload αρχείου CSV
uploaded_file = st.file_uploader("Επιλέξτε αρχείο CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Προεπισκόπηση δεδομένων:")
    st.write(df.head())
    
    # Επιλογή στήλης για ανάλυση
    column = st.selectbox("Επιλέξτε στήλη για στατιστική ανάλυση", df.columns)
    
    # Οπτικοποίηση δεδομένων
    st.write("### Ιστόγραμμα της επιλεγμένης στήλης")
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Βασικά στατιστικά
    st.write("### Βασικά στατιστικά στοιχεία")
    st.write(df[column].describe())
    
    # Heatmap για correlation matrix
    st.write("### Heatmap Συσχετίσεων")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # PCA για μείωση διαστάσεων
    st.write("### PCA Ανάλυση")
    num_components = st.slider("Επιλέξτε αριθμό συνιστωσών", 1, min(df.shape[1], 10), 2)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(df_scaled)
    
    pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(num_components)])
    st.write("### PCA Αποτελέσματα")
    st.write(pca_df.head())
    
    # Οπτικοποίηση PCA
    st.write("### PCA Διάγραμμα")
    fig, ax = plt.subplots()
    sns.scatterplot(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

# Πληροφορίες για την ομάδα
st.sidebar.title("Πληροφορίες Ομάδας")
st.sidebar.write("Όνομα 1 - Συνεισφορά")
st.sidebar.write("Όνομα 2 - Συνεισφορά")
st.sidebar.write("Όνομα 3 - Συνεισφορά")
