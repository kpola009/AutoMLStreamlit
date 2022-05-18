import os
import streamlit as st
import numpy as np
from PIL import Image

from multipage import MultiPage
from pages import tabular, image, text, timeseries

app = MultiPage()

st.title("Space.ai")
st.header("AutoML")

app.add_page("Tabular", tabular.app)
app.add_page("Image", image.app)
app.add_page("Text", text.app)
app.add_page("Time Series", timeseries.app)

app.run()
