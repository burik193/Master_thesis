
<div align="center">

# Master thesis
## Fuzzy name matching using neural network approach

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="Jellyfish" src="https://jamesturk.github.io/jellyfish/assets/white-jellyfish.svg" width="20"></a>
[![Discord](https://dcbadge.vercel.app/api/server/3wYvAaz3Ck?style=flat)](https://discord.gg/3wYvAaz3Ck)
</div>


## 📌&nbsp;&nbsp;Introduction
This thesis addresses the persistent problem faced by businesses across diverse fields, including B2B, B2C,
E-commerce, service-based, healthcare, NPOs, financial institutions, telecommunications, retail, real estate,
and transportation. Specifically, nearly all companies in these fields use one or more Customer Relationship
Management (CRM) systems to manage their internal and external relationships, as well as store and
protect various types of data, such as product information, customer data, and shareholder information.
This data can be used to develop prediction or suggestion models to better understand customer behavior,
market trends, and pricing development.

However, despite the availability of numerous tools to solve these problems, their effectiveness is contingent
upon the availability of valid and reliable data. To obtain insights from the data, companies must first
collect and prepare it for use. The collection and processing of data constitute the foundation of any
application that relies on this data over its lifetime.
Consider a company that hosts various events, both in-person and online, and collects participant data
through a registration form that includes the participant’s name. While the company can use it for the
variety of internal tasks, such as advertising or product suggestions, the data is not directly useful. One
potential issue with this kind of data is the validity, which requires matching the data from the registration
sheet to the data in the CRM system. While the CRM data can be assumed to be accurate, there is no
guarantee of the quality of the registration data, which may contain errors such as misspellings, typos, or
even conscious misspellings, such as using a nickname instead of the real name. Moreover, the data may be
incomplete and contain sparse information, especially when the event includes participants from outside
the company. This problems, that are faced by big companies on everyday basis, serves as inspiration for
this theses.

We define a set of problems that arise from matching two large datasets of character strings without
contextual information or semantic sense, which include proper names and common errors. We refer to this
set of problems as Fuzzy Name Matching (FNM). Our goal is to develop a mapping solution that matches
misspelled names to their valid counterparts, create an effective solution architecture, and perform the
matching task in feasible time.

##<img alt="HoTo" src="question-mark.png" width=12> How to start

Install python 3.9 and proceed with

    pip install -r requirements.txt

Note, that to train the models and to reduce the spectrogram generation time we recommend using CUDA.
You can find the proper installation guide here: https://pytorch.org/get-started/locally/. The resulting command should 
look like following:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

##<img alt="HoTo" src="struct.png" width=50>  Code structure

The code is wrapped inside the jupyter notebook interface for exploration purposes and equipped with additional links 
for better navigation using Table of Contents.

The purpose of each notebook is described in the Appendix B.



