import os
import streamlit as st

st.set_page_config(page_title='PulmoAID Info', layout='wide', initial_sidebar_state='auto', page_icon='ℹ️')

st.title("Advancing Lung Cancer Diagnostics through Multimodal Deep Learning")
st.subheader("Optimal Fusion of 3D Radiological Imaging and Clinical Data with LLM-Driven Interpretability - Tushar Mehta")

st.markdown("""
This research showcases the power of multimodal fusion and LLM-driven interpretability in enhancing both diagnostic accuracy and clinical trust in lung cancer detection.

**Key Contributions:**

* **Novel Multimodal Algorithm:** Developed a cutting-edge algorithm that intelligently integrates CT imaging data with crucial demographic information and comprehensive clinical reports. This fusion aims to achieve superior lung cancer detection capabilities.
* **Advanced Feature Extraction:** Leveraged the strengths of transfer learning by employing 2D (VGG16) and 3D (MedicalNet, 3DResNet) models to extract meaningful features from radiological images.
* **Optimized Fusion Strategy:** Identified the optimal fusion model, which strategically combines features extracted by VGG16 with relevant clinical predictors and sentiment analysis derived from clinical reports using Large Language Models (LLMs). This combined information is then fed into a Logistic Regression classifier for final prediction.
* **Validation on Large-Scale Dataset:** The developed model was rigorously trained and evaluated on the extensive National Lung Screening Trial (NLST) dataset, ensuring the robustness and generalizability of the findings.
* **Enhanced Diagnostic Accuracy and Clinical Trust:** The research demonstrates that this multimodal approach, coupled with LLM-driven interpretability, leads to significant improvements in diagnostic accuracy, ultimately fostering greater clinical trust in the system's recommendations.
""")

st.markdown(""" 
# **PulmoAID**
*Enabling AI-based diagnostics for lung cancer using advanced multimodal feature fusion approach.*
""".strip())

st.subheader('Data Summary Statistics')
st.code(body=""" 
DATA SUMMARY STATISTICS 

Training and Testing Dataset
1A (Positive Patients) - 310
1G (Negative Patients) - 269

Validation (Field Testing) Dataset - 
2A (Positive Patients) - 312
2G (Negative Patients) - 184
    """.strip(), language=None)
st.markdown('# **App Demonstration**')
st.video("https://youtu.be/7Bp_GRH7F_Q")
st.subheader('Code Repository')
st.markdown("[GitHub](https://github.com/TMRmehta/PulmoAID)")
 
st.subheader('PulmoAID App')
st.markdown("[Streamlit.io](https://pulmoaid.streamlit.app/)")

# st.subheader('Model Architecture Summary')
# st.image(image=os.path.join('images', 'architecture.png'))
# st.write(""" 
# This model integrates multimodal feature fusion to detect lung cancer from CT scan images. 
# It employs a pretrained VGG-16 network for feature extraction, capturing deep spatial representations from the input images. 
# These extracted features are then processed through a fully connected neural network (FCNN), 
# serving as a fusion layer to integrate and refine the learned representations. 
# Finally, the fused features are passed to a logistic regression classifier, which performs binary classification to 
# predict the likelihood of lung cancer. This architecture effectively combines deep learning-based feature
# extraction with traditional classification techniques to enhance diagnostic accuracy.
# """.strip())

# st.subheader('LLM Integration')
# st.image(os.path.join('images', 'gemini_logo.jpg'))
# st.write(''' 
# Gemini 1.5 Flash's speed and efficiency allows rapid analysis of patient data. 
# Its large context window allows for the simultaneous processing of comprehensive patient histories, potentially identifying subtle patterns and correlations that might be missed by human clinicians. 
# This could lead to faster diagnoses, especially in time-sensitive situations like emergency medicine, and enable cost-effective screening for widespread conditions. 
# The model's multimodal reasoning capabilities could also integrate diverse data sources, such as genetic information and lifestyle factors, to provide more holistic and personalized diagnostic insights, improving accuracy and efficiency in healthcare workflows.

# The model was used to induce a synthetic varible (llm_sentiment) to act as a doctor's sentiment/score for cancer likeliness to improve model accuracy.
# It was prompted to generate a score between 0-10 based on patient's cancer history to add a third modality to the classifier's prediction.
# This LLM also serves as a context-aware question answering chatbot/virtual doctor to interact with the clinical data.
#     ''')	

# st.subheader('System Architecture')
# st.image(os.path.join('images', 'system_architecture.jpeg'))

# st.subheader('Multimodal Fusion Model Metrics')
# st.image(os.path.join('images', 'fusion_metrics.png'))

st.subheader('Citations and Sources')
st.markdown("[National Lung Screening Trial (NLST)](https://www.cancer.gov/types/lung/research/nlst)")
st.markdown("[VGG-16 (PyTorch)](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)")
