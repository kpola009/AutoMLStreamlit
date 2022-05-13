import streamlit as st
from streamlit_pages.streamlit_pages import MultiPage
import pandas as pd
from sklearn.model_selection import train_test_split
from regression import autoregression
st.title("Space.ai")
st.header("AutoML")


## Tabular Data Type function
def tabular():
    st.subheader("Tabular")
    st.header('Upload Dataset')

    # File uploader
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    # File show
    if data_file is not None:
        st.write(type(data_file))
        df = pd.read_csv(data_file)
        st.dataframe(df)

        # Algorithm Selection
        st.subheader('Select AI Algorithm to build')
        algoselect = st.selectbox('Options', ['','Machine Learning', 'Deep Learning'])

        # if algoselect is Machine Learning
        if algoselect == 'Machine Learning':
            target_variable = st.selectbox('Select the Target Variable', df.columns)

            if target_variable is not None:
                y = df[target_variable]
                X = df.loc[:, df.columns != target_variable]

                st.write("You selected target variable ")
                st.dataframe(y)

                min_time = st.number_input('Minimum Time for Training in seconds')

                if min_time:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
                    st.write("Models are getting Trained")
                    models = autoregression(X_train, X_test, y_train, y_test, min_time)
                    st.dataframe(models)


        ## TODO Data Visualization
        ## TODO Featuring Engineering, Data Cleaning
        ## TODO AutoML (Auto-Sklearn)


        # if algoselect is Deep Learning
        elif algoselect == 'Deep Learning':
            target_variable = st.selectbox('Select the Target Variable', df.columns)

            if target_variable is not None:
                y = df[target_variable]
                X = df.loc[:, df.columns != target_variable]

                st.write("You selected target variable ")
                st.dataframe(y)

        ## TODO Data Visualization
        ## TODO Featuring Engineering, Data Cleaning
        ## TODO AutoML


## Image Data Type function
def image():
    st.subheader("Image")
    st.header('Upload Dataset')
    data_file = st.file_uploader("Upload CSV", type=['csv'])

## Text Data Type function
def text():
    st.subheader("Text")
    st.header('Upload Dataset')
    data_file = st.file_uploader("Upload CSV", type=['csv'])

## Timeseries Data Type function
def timeseries():
    st.subheader("Timeseries")
    st.header('Upload Dataset')
    data_file = st.file_uploader("Upload CSV", type=['csv'])

# call app class object
app = MultiPage()
# Add pages
app.add_page("Tabular",tabular)
app.add_page("Image",image)
app.add_page("Text",text)
app.add_page("Timeseries",timeseries)
app.run()