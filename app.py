import streamlit as st

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, match_pair
from lightglue import viz2d
from pathlib import Path
import torch
from lightglue import LightGlue, SuperPoint, DISK
import os
import uuid
from PIL import Image


# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1000).eval()  # load the extractor
matcher = LightGlue(pretrained='superpoint').eval() # load the matcher

# or DISK+LightGlue
extractor = DISK(max_num_keypoints=1000).eval() # load the extractor
matcher = LightGlue(pretrained='disk').eval() # load the matcher



@st.cache_data
def save_uploadedfile(uploadedfile, image_name):
     with open(os.path.join("assets/images",image_name),"wb") as f:
         f.write(uploadedfile.getbuffer())


extractor = SuperPoint(max_num_keypoints=2048, nms_radius=3).eval()  # load the extractor
match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}

# make any grid with a function
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid



st.set_page_config(layout="wide")
st.header("LightGlue: A Smarter, Faster Image Matching Technique ")


st.write(" ")
st.write(" ")

st.write("Please select two images")

extractor = SuperPoint(max_num_keypoints=2048, nms_radius=3).eval()  # load the extractor
match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval()





# Generate a unique key for each widget
key1 = str(uuid.uuid4())
key2 = str(uuid.uuid4())



# Define the column layout
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("Upload your image 1", type=['png', 'jpeg', 'jpg'], key='key1')

    if uploaded_file1 is not None:
        st.image(uploaded_file1)
        save_uploadedfile(uploaded_file1, "image0.jpg")
        
with col2:
    uploaded_file2 = st.file_uploader("Upload your image 2", type=['png', 'jpeg', 'jpg'], key='key2')

    if uploaded_file2 is not None:
        st.image(uploaded_file2)
        save_uploadedfile(uploaded_file2, "image1.jpg")

mygrid0 = make_grid(1,3)

if uploaded_file2 is not None and uploaded_file2 is not None:
    mybutton = mygrid0[0][1].button('Matching')

    if mybutton:
      with st.spinner("Matching ..."):

        # SuperPoint+LightGlue
        extractor = SuperPoint(max_num_keypoints=1000).eval()  # load the extractor
        matcher = LightGlue(pretrained='superpoint').eval() # load the matcher

        # or DISK+LightGlue
        extractor = DISK(max_num_keypoints=1000).eval() # load the extractor
        matcher = LightGlue(pretrained='disk').eval() # load the matcher

        # load images to torch and resize to max_edge=1024
        image0, scales0 = load_image("assets/images/image0.jpg")
        image1, scales1 = load_image("assets/images/image1.jpg")

        # extraction + matching + rescale keypoints to original image size
        pred = match_pair(extractor, matcher, image0, image1,
                        scales0=scales0, scales1=scales1)    

        kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]




        axes = viz2d.plot_images([image0.permute(1, 2, 0), image1.permute(1, 2, 0)])
        viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)

        viz2d.save_plot('assets/save1.jpg')

        st.image('assets/save1.jpg')


        

st.write(" ")
st.write(" ")
st.write(" ")

st.write("---")
st.write("References:")
st.write("[1] Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). LightGlue: Local Feature Matching at Light Speed. ArXiv PrePrint.")

