import streamlit as st

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, match_pair
from lightglue import viz2d
from pathlib import Path
import torch
from lightglue import LightGlue, SuperPoint, DISK



from pathlib import Path
from setuptools import setup

description = ['LightGlue']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

extra_dependencies = ['torch', 'kornia', 'numpy', 'einops']

setup(
    name='lightglue',
    version='0.0',
    packages=['lightglue'],
    python_requires='>=3.6',
    extras_require={'extra': extra_dependencies},
    author='Philipp Lindenberger, Paul-Edouard Sarlin',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/cvg/LightGlue/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)



extractor = SuperPoint(max_num_keypoints=2048, nms_radius=3).eval()  # load the extractor
match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval()


st.write('Hello')


# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=1000).eval()  # load the extractor
matcher = LightGlue(pretrained='superpoint').eval() # load the matcher

# or DISK+LightGlue
extractor = DISK(max_num_keypoints=1000).eval() # load the extractor
matcher = LightGlue(pretrained='disk').eval() # load the matcher

# load images to torch and resize to max_edge=1024
image0, scales0 = load_image("assets/Lille_Fladre0.jpg")
image1, scales1 = load_image("assets/Lille_Fladre1.jpeg")

# extraction + matching + rescale keypoints to original image size
pred = match_pair(extractor, matcher, image0, image1,
                  scales0=scales0, scales1=scales1)    

kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]




axes = viz2d.plot_images([image0.permute(1, 2, 0), image1.permute(1, 2, 0)])
viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)

viz2d.save_plot('assets/save1.jpg')

st.image('assets/save1.jpg')