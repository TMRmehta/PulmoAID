import joblib
import random
import torch
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from database import DBManager
from manager import DataManager, LungCancerVGG16Fusion
from gemini import LLM
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import json, io
import pytz
EST_TIME = pytz.timezone('US/Eastern')

import warnings
warnings.simplefilter('ignore')

# INIT PATHS
train_csv = os.path.join('data', 'selected_data.csv')
val_csv = os.path.join('data', 'selected_data.csv')
logo_img = os.path.join('images', 'logo.png')
arch_img = os.path.join('images', 'architecture.png')

modelpaths = {
	"Logistic Regression": os.path.join('classifiers', 'FusionModel LR_97.41.pkl'),
	"KNN": os.path.join('classifiers', 'FusionModel KNN_73.28.pkl'),
	"Naive Bayes": os.path.join('classifiers', 'FusionModel NB_78.45.pkl'),
	"Random Forest": os.path.join('classifiers', 'FusionModel RFC_91.38.pkl'),
	"XGBoost": os.path.join('classifiers', 'FusionModel XGB_92.24.pkl')
}


# INIT SLICES
feature_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
demographic_cols = ['age', 'ethnic', 'gender', 'height', 'race', 'weight']
smoking_hist = ['age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday', 'smokelive', 'smokework', 'smokeyr']
llm_sent = ['llm_sentiment']
clinical = ['biop0', 'bioplc', 'proclc', 'can_scr', 'canc_free_days']
treatments = ['procedures', 'treament_categories', 'treatment_types', 'treatment_days']


if 'layout' not in st.session_state: st.session_state.layout = 'centered'
if 'login' not in st.session_state: st.session_state.login = False
if 'scans' not in st.session_state: st.session_state.scans = []
if 'pil_images' not in st.session_state: st.session_state.pil_images = []
if 'subject' not in st.session_state: st.session_state.subject = 'N/A'

st.set_page_config(page_title='PulmoAID', 
				   layout=st.session_state.layout,
				   page_icon='🫁')


def info_tab():
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

	st.subheader('Model Architecture Summary')
	st.image(image=arch_img)
	st.write(""" 
This model integrates multimodal feature fusion to detect lung cancer from CT scan images. 
It employs a pretrained VGG-16 network for feature extraction, capturing deep spatial representations from the input images. 
These extracted features are then processed through a fully connected neural network (FCNN), 
serving as a fusion layer to integrate and refine the learned representations. 
Finally, the fused features are passed to a logistic regression classifier, which performs binary classification to 
predict the likelihood of lung cancer. This architecture effectively combines deep learning-based feature
extraction with traditional classification techniques to enhance diagnostic accuracy.
	""".strip())

	st.subheader('LLM Integration')

	st.image(os.path.join('images', 'gemini_logo.jpg'))
	st.write(''' 
Gemini 1.5 Flash's speed and efficiency allows rapid analysis of patient data. 
Its large context window allows for the simultaneous processing of comprehensive patient histories, potentially identifying subtle patterns and correlations that might be missed by human clinicians. 
This could lead to faster diagnoses, especially in time-sensitive situations like emergency medicine, and enable cost-effective screening for widespread conditions. 
The model's multimodal reasoning capabilities could also integrate diverse data sources, such as genetic information and lifestyle factors, to provide more holistic and personalized diagnostic insights, improving accuracy and efficiency in healthcare workflows.

The model was used to induce a synthetic varible (llm_sentiment) to act as a doctor's sentiment/score for cancer likeliness to improve model accuracy.
It was prompted to generate a score between 0-10 based on patient's cancer history to add a third modality to the classifier's prediction.
This LLM also serves as a context-aware question answering chatbot/virtual doctor to interact with the clinical data.
		''')	
	st.subheader('Multimodal Fusion Model Metrics')
	st.image(os.path.join('images', 'fusion_metrics.png'))

	st.subheader('Citations and Sources')
	st.markdown("[National Lung Screening Trial (NLST)](https://www.cancer.gov/types/lung/research/nlst)")
	st.markdown("[VGG-16 (PyTorch)](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)")


@st.cache_resource
def load_model_from_chunks(model_class, chunks_dir, device='cpu'):
	"""
	Load a split model directly into memory for Streamlit
	Uses st.cache_resource to prevent reloading on every rerun
	
	Args:
		model_class: The PyTorch model class
		chunks_dir (str): Directory containing the model chunks
		device (str): Device to load the model to
	"""
	# Read metadata
	with open(os.path.join(chunks_dir, 'metadata.json'), 'r') as f:
		metadata = json.load(f)
	
	# Create a bytes buffer
	buffer = io.BytesIO()
	
	# Join chunks directly in memory
	for i in range(metadata['num_chunks']):
		chunk_filename = f'chunk_{i:03d}.bin'
		with open(os.path.join(chunks_dir, chunk_filename), 'rb') as f:
			buffer.write(f.read())
	
	# Reset buffer position
	buffer.seek(0)
	
	# Load model
	model = model_class().to(device)
	model.load_state_dict(torch.load(buffer, map_location=device))
	
	return model


@st.cache_resource
def initialize_model():
	try:
		device = torch.device('cpu')
		model = load_model_from_chunks(
			model_class=LungCancerVGG16Fusion,
			chunks_dir="model_chunks",  # This should be in your repo
			device=device
		)
		model.eval()
		return model
	
	except Exception as e:
		# st.error(f"Error loading model: {str(e)}")
		print(e)
		return None


@st.cache_resource
def load_llm():
	loaded_llm = LLM(st.secrets["keys"]["api"])
	return loaded_llm


@st.cache_resource
def utilloader(utility:str):
	if utility == 'llm':
		loaded_llm = LLM(st.secrets["keys"]["api"])
		return loaded_llm
	
	if utility == 'manager':
		torch.manual_seed(0)
		# device = torch.device('cpu')
		# VGG_16 = LungCancerVGG16Fusion().to(device)
		# modelpath = os.path.join("models", "best_vgg16.pth")
		# VGG_16.load_state_dict(torch.load(modelpath, weights_only=True, map_location=device))
		# VGG_16.eval()

		# From chunks
		VGG_16 = initialize_model()
		VGG_16.eval()

		return DataManager(VGG_16)
	
	if utility == 'subject_list':
		data = pd.read_csv(train_csv)
		return list(data['Subject'])

	if utility == 'classifier_csv':
		return pd.read_csv(os.path.join("data", "selected_data.csv")) 
	
	if utility == 'llm_csv':
		return pd.read_csv(os.path.join("data", "selected_descriptive_data.csv"))
	
	if utility == 'database':
		return DBManager()
	
	if utility == 'doctor_notes':
		return pd.read_csv(os.path.join('data', 'doctor_notes.csv'))
	
	if utility == 'patient_notes':
		return pd.read_csv(os.path.join('data', 'patient_notes.csv'))

@st.cache_resource
def load_classifier(name:str):
	return joblib.load(modelpaths[name])


@st.cache_resource
def generate_outcome(features=[], subject='', classifier='', full_row=None) -> str:
    global csvdata, feature_cols
    
    row = csvdata[csvdata['Subject'] == int(subject)]
    newrow = row[feature_cols + demographic_cols + smoking_hist + llm_sent].values.flatten().tolist()
    model = load_classifier(classifier)
    
    # Determine which row to use for prediction
    prediction_row = full_row if full_row is not None else [newrow]
    
    try:
        # Try to get probability prediction
        outcome = model.predict_proba(prediction_row)
        probability_negative = outcome[0][0] * 100
        probability_positive = outcome[0][1] * 100
        
        result = f"**Subject {subject}** has tested **{'🔴 _Positive_' if probability_positive > probability_negative else '🟢 _Negative_'}** with a confidence of **{max(probability_positive, probability_negative):.2f}%**\n"
        result += '\n' + f"**Probability distribution:** Positive: **{probability_positive:.2f}** | Negative: **{probability_negative:.2f}%**"
        
    except AttributeError:
        # Fall back to binary prediction
        outcome = model.predict(prediction_row)
        is_positive = int(outcome[0]) == 1
        
        result = f"**Subject {subject}** has tested **{'🔴 _Positive_' if is_positive else '🟢 _Negative_'}**\n"
        result += '\n' + f"**Probability distribution:** Not available for this model type"
    
    return result


# @st.cache_resource
def generate_shap_plot(base: pd.DataFrame, subject: str):
	# np.random.seed(0)
	# model = load_classifier('XGBoost')

	# features = ['n1', 'n2', 'n3', 'n4',
	# 			'age', 'ethnic', 'gender', 'height', 'race', 'weight',
	# 			'age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday',
	# 			'smokelive', 'smokework', 'smokeyr']
	# X = base[features]
	# y = base['lung_cancer']

	# # Fit model before SHAP calculation
	# model.fit(X, y)
	
	# subject_index = base[base['Subject'] == int(subject)].index
	# explainer = shap.TreeExplainer(model)
	# shap_values = explainer.shap_values(X)
	# subject_shap_values = shap_values[subject_index]

	# plt.figure(figsize=(12, 8))
	# shap.summary_plot(shap_values, X, max_display=20, show=False)

	# # Get feature importance order
	# mean_abs_shap = np.abs(shap_values).mean(axis=0)
	# feature_importance_order = np.argsort(-mean_abs_shap)[:20]
	
	# # Plot subject points
	# for i, idx in enumerate(feature_importance_order):
	# 	plt.scatter(subject_shap_values[0][idx], i, color='black', edgecolor='white', s=50, zorder=3)

	# plt.title("SHAP Summary Plot with Subject Highlighted")
	# plt.tight_layout()
	
	# return plt

	shap_path = os.path.join('shap_plots', f'{subject}.png')
	image = Image.open(shap_path)

	return image


def history_writer(df: pd.DataFrame, imagelist: list):
	placeholder = os.path.join('images', 'notfound.jpg')
	ts = df['Timestamp'][0]
	dt_obj = datetime.fromisoformat(ts)
	dt_obj_est = dt_obj.astimezone(EST_TIME)
	formatted_time = dt_obj_est.strftime("%m/%d/%Y - %I:%M %p")
	
	st.subheader(formatted_time)
	st.markdown(df['Result'][0])
	
	df_slice = df[['Subject'] + demographic_cols + smoking_hist]
	st.dataframe(df_slice, use_container_width=True, hide_index=True)

	col1, col2, col3, col4 = st.columns(4)
	cols = [col1, col2, col3, col4]

	for i in range(4):
		with cols[i]:
			img_path = os.path.join('ct_scans', imagelist[i]) if i < len(imagelist) else placeholder
			if not os.path.exists(img_path):  # Check if file exists
				img_path = placeholder

			st.image(img_path, use_container_width=True, caption=os.path.basename(img_path).split('.')[0])
	
	st.divider()


def metadata_tab():
	st.markdown(
''' 
## **Variable Encoding Metadata**  

### **Gender**  
- **1** – Male  
- **2** – Female  

### **Ethnicity**  
- **1** – Hispanic or Latino  
- **2** – Neither Hispanic nor Latino  
- **7** – Participant refused to answer  
- **95** – Missing data form (not expected to ever be completed)  
- **98** – Missing (form submitted, but answer left blank)  
- **99** – Unknown  

### **Race**  
- **1** – White  
- **2** – Black or African-American  
- **3** – Asian  
- **4** – American Indian or Alaskan Native  
- **5** – Native Hawaiian or Other Pacific Islander  
- **6** – More than one race  

### **Smoked Cigar/Pipe**  
### **Lived/Worked with Smoker**
- **0** – No  
- **1** – Yes  
'''
	)


# New Doctor Page
def doctor_page():
	global csvdata, llmdata, db
	
	# Initialize session state variables if they don't exist
	if 'edited_data' not in st.session_state: st.session_state.edited_data = {}
	if 'pil_images' not in st.session_state: st.session_state.pil_images = []
	if 'current_prediction' not in st.session_state: st.session_state.current_prediction = None
	if 'uploaded_files_list' not in st.session_state: st.session_state.uploaded_files_list = []
	
	with st.sidebar:
		st.header(body='Doctor\'s Portal')
		st.write(datetime.now(EST_TIME).strftime("%d %b %Y %I:%M %p") + ' EST')
		st.write(f'Welcome, Tushar Mehta')

		logout = st.button(label='Logout', use_container_width=True)
		
		if logout:
			st.session_state.messages = []
			st.session_state.edited_data = {}
			st.session_state.pil_images = []
			st.session_state.current_prediction = None
			st.session_state.uploaded_files_list = []
			st.session_state.login = False
			st.rerun()

		st.session_state.subject_selection = st.selectbox(label='Patient ID', options=st.session_state.subject_list)
		st.session_state.model_selection = st.selectbox(label='Classifier', options=list(modelpaths.keys()))

		clinical_data = st.toggle(label='Clinical Data')
		demographic_data = st.toggle(label='Demographic Data')
		smoking_history = st.toggle(label='Smoking History')

	st.image(image=logo_img, use_container_width=True)

	information, images_clinical, diagnostics, historaical_data, ai, metadata = st.tabs(['Information', 'Images and Clinical', 'Diagnostics', 'Patient History', 'Talk To Virtual Doctor', 'Metadata'])

	with information:
		info_tab()

	with images_clinical:
		uploaded_files = st.file_uploader(label='Upload Scans', accept_multiple_files=True, type=["jpg", "jpeg", "png"])
		
		# Store uploaded files in session state
		if uploaded_files:
			st.session_state.uploaded_files_list = uploaded_files
			st.session_state.pil_images = []  # Clear previous images
			
			tmpset = set()
			for file in uploaded_files:
				name = file.name
				tmpset.add(name.split('_')[0])

			if len(tmpset) > 1:
				st.warning('Input files are of different subjects, please give images for one subject only.')
			
			if tmpset:
				tmp_current = str(tmpset.pop())
				# Update the selected subject based on uploaded files
				st.session_state.selected_subject = tmp_current
				
				# Warn if selected subject doesn't match images
				if str(tmp_current) != str(st.session_state.subject_selection):
					st.warning('Warning! Uploaded image ID(s) do not match with selected subject. Please choose correct subject in sidebar. Unmatched images may lead to incorrect prediction.')

		else:
			# If no files uploaded, use the sidebar selection
			st.session_state.selected_subject = st.session_state.subject_selection

		edited_data = {}
		original_columns = csvdata.columns.tolist()
		c1, c2, c3 = st.columns(3)

		original = csvdata[csvdata['Subject'] == int(st.session_state.selected_subject)]

		def process_data(section_name, columns):
			"""Handles editing and storing modified data while preserving structure."""
			display_names = {
				'age': 'Patient Age',
				'ethnic': 'Ethnicity',
				'gender': 'Gender',
				'height': 'Height (in)',
				'race': 'Race',
				'weight': 'Weight (lbs)',
				'age_quit': 'Quit Smoking Age',
				'cigar': 'Smoked Cigar',
				'pipe': 'Smoked Pipe',
				'pkyr': 'Pack Years',
				'smokeage': 'Smoking Onset Age',
				'smokeday': 'Avg Days Smoked',
				'smokelive': 'Lives With Smoker(s)',
				'smokework': 'Works With Smoker(s)',
				'smokeyr': 'Total Smoking Years',
				'biop0': 'Biopsy Type',
				'bioplc': 'Cancer Biopsy',
				'can_scr': 'Screened for Cancer',
				'canc_free_days': 'Days Without Cancer',
				'proclc' : 'Cancer Procedure',
				'cigsmok' : 'Currently Smoking'
			}
			
			slice_df = csvdata[['Subject'] + columns]
			data_df = slice_df[slice_df['Subject'] == int(st.session_state.selected_subject)].T
			data_df.columns = ['Value']
			
			# Apply display names to the index
			data_df.index = data_df.index.map(lambda x: display_names.get(x, x))
			
			# Editable dataframe
			edited_df = st.data_editor(data_df, use_container_width=True)
			
			# Convert display names back to original column names
			reverse_mapping = {v: k for k, v in display_names.items()}
			edited_df.index = edited_df.index.map(lambda x: reverse_mapping.get(x, x))
			
			# Store edited values while avoiding duplicate 'Subject' columns
			edited_df = edited_df.T
			edited_df = edited_df.drop(columns=['Subject'], errors='ignore')
			edited_df.insert(0, 'Subject', st.session_state.selected_subject)  # Ensure 'Subject' is the first column
			
			edited_data[section_name] = edited_df
			# Store in session state for access in other tabs
			st.session_state.edited_data[section_name] = edited_df


		with c1:
			if clinical_data:
				st.write('Clinical Data')
				process_data('Clinical', clinical)

		with c2:
			if demographic_data:
				st.write('Demographic Data')
				process_data('Demographic', demographic_cols)

		with c3:
			if smoking_history:
				st.write('Smoking History')
				process_data('Smoking History', smoking_hist)

		# DOCTORS NOTES AND PATIENT OBSERVATIONS

		st.session_state.dc_notes = st.text_area(label='Doctor\'s Notes', 
										         value=doc_notes[doc_notes['Subject'] == int(st.session_state.selected_subject)]['comments'].values[0])

		notes = st.file_uploader(label='Upload New Notes', type=['pdf', 'txt', 'docx'], accept_multiple_files=False)
		save = st.button('Save/Update Notes', use_container_width=True)

		st.session_state.pt_notes = st.text_area(label='Patient\'s Observations',
											     value=pat_notes[pat_notes['Subject'] == int(st.session_state.selected_subject)]['Remark'].values[0])
		
		save_obs = st.button(label='Save Observations', use_container_width=True)


	with diagnostics:
		st.write(""" 
		Comparison of current analysis with the last diagnostics in terms of
		probability key factors that are different.
		""".strip())
		
		st.subheader(f"Patient ID: {st.session_state.selected_subject}")

		submit = st.toggle(
			label='Generate Fusion Model Prediction (Please upload CT Scans First)' if not uploaded_files else "Generate Fusion Model Prediction", 
			disabled=not uploaded_files)

		if 0 < len(uploaded_files) < 4:
			st.warning('It is recommended to upload at least 4 images for the subject.')

		if uploaded_files and submit:
			nameset = set()

			for file in uploaded_files:
				name = file.name
				nameset.add(name.split('_')[0])
				try:
					image = Image.open(file).convert("RGB")
					st.session_state.pil_images.append(image)
				except Exception as e:
					st.error(f"Error processing '{file.name}': {e}")
				
			if len(nameset) > 1:
				st.warning('Input files are of different subjects, please give images for one subject only.')
			else:
				current_subject = nameset.pop()
				st.session_state.selected_subject = current_subject

				with st.spinner(text='Running Model...'):
					features = Manager.extract_features(imagelist=st.session_state.pil_images)
					outcome = generate_outcome(features, current_subject, st.session_state.model_selection)
					st.markdown(outcome)
					# Store the prediction in session state
					st.session_state.current_prediction = outcome
		

		# Merge all edited data from session state
		final_edited_df = None
		
		if st.session_state.edited_data:
			dfs_to_concat = list(st.session_state.edited_data.values())
			if dfs_to_concat:
				final_edited_df = pd.concat(dfs_to_concat, axis=1)
				
				# Remove duplicate columns (keeping the first occurrence)
				final_edited_df = final_edited_df.loc[:, ~final_edited_df.columns.duplicated()]
				
				# Get original data for the selected subject to fill any missing values
				original = csvdata[csvdata['Subject'] == int(st.session_state.selected_subject)]
				
				# Make sure we have all columns from the original dataframe
				final_edited_df = final_edited_df.reindex(columns=original_columns, fill_value=None)
				
				# Fill missing values with original data
				if not original.empty:
					final_edited_df = final_edited_df.fillna(original.set_index('Subject').loc[int(st.session_state.selected_subject)])
				
				new_pred = st.toggle('Generate New Prediction')
				
				if new_pred:
					# Make sure we have all necessary feature columns
					new_X = final_edited_df[feature_cols + demographic_cols + smoking_hist + llm_sent]
					
					new_results = generate_outcome(
						subject=st.session_state.selected_subject, 
						classifier=st.session_state.model_selection, 
						full_row=new_X
					)
					
					st.markdown(new_results)
					st.session_state.current_prediction = new_results

					save_results = st.button('Save New Results', use_container_width=True)

					if save_results:
						tmp_df = final_edited_df[['Subject'] + demographic_cols + smoking_hist].iloc[0].copy()
						tmp_df['Result'] = new_results
						tmp_df['Timestamp'] = datetime.now().isoformat()
						
						# Get image names from uploaded files
						image_names = []
						if st.session_state.uploaded_files_list:
							image_names = [file.name for file in st.session_state.uploaded_files_list]
						
						tmp_df['Images'] = image_names
						df_dict = pd.DataFrame(tmp_df).T.to_dict(orient='records')[0]
						
						save_success = db.save(df_dict)

						if save_success:
							st.toast('Saved diagnostics to database.')

						else:
							st.toast('Datbase upload failed!')
		else:
			st.info("No edits have been made to patient data. Make changes in the 'Images and Clinical' tab first to generate new predictions.")

		cl1, cl2 = st.columns(2)

		with cl1:
			show_scan_disabled = len(st.session_state.uploaded_files_list) == 0
			# show_scan = st.toggle(
			# 	label='Show CT Scan (Upload CT Scans First)' if show_scan_disabled else "Show CT Scan",
			# 	disabled=show_scan_disabled)
			
			show_scan = st.toggle(
				label='Show Saliency Map (Please Upload CT Scans First)' if show_scan_disabled else "Show Saliency Map",
				disabled=show_scan_disabled)

			if show_scan and st.session_state.uploaded_files_list:
				# st.image(st.session_state.uploaded_files_list[0], caption='CT Scan')
				plot_path = os.path.join('saliency_plots', f'{st.session_state.subject_selection}.png')

				if os.path.exists(plot_path):
					st.image(image=plot_path, caption=os.path.basename(plot_path).split('.')[0])

				else:
					st.image(image=os.path.join('images', 'notfound.jpg'), caption='Failed to Load')

		with cl2:
			tmp_current = st.session_state.selected_subject
			show_shap_disabled = len(st.session_state.uploaded_files_list) == 0
			
			show_shap = st.toggle(
				label='Generate SHAP Plot (Upload CT Scans First)' if show_shap_disabled else "Generate SHAP Plot", 
				disabled=show_shap_disabled)

			if show_shap:
				# Check if SHAP plot exists for the current subject
				shap_path = os.path.join('shap_plots', f'{tmp_current}.png')
				if os.path.exists(shap_path):
					st.image(shap_path)
				else:
					st.warning(f"SHAP plot for subject {tmp_current} not found.")


		# st.session_state.dc_notes = st.text_area(label='Doctor\'s Notes', 
		# 								         value=doc_notes[doc_notes['Subject'] == int(st.session_state.selected_subject)]['comments'].values[0])

		# notes = st.file_uploader(label='Upload New Notes', type=['pdf', 'txt', 'docx'], accept_multiple_files=False)
		# save = st.button('Save/Update Notes', use_container_width=True)

		# st.session_state.pt_notes = st.text_area(label='Patient\'s Observations',
		# 									     value=pat_notes[pat_notes['Subject'] == int(st.session_state.selected_subject)]['Remark'].values[0])
		
		# save_obs = st.button(label='Save Observations', use_container_width=True)


	with historaical_data:
		show_history = st.toggle('Load Patient History')

		if show_history:
			records = db.fetch(subject_id=int(st.session_state.selected_subject))

			if len(records) > 0:
				for entry in records:
					df = pd.DataFrame([entry])
					imagelist = entry.get("Images", [])  # Extract image list or default to empty list
					history_writer(df, imagelist)

			else:
				st.write('No older records found for this patient.')

	with ai:
		if 'llm' not in st.session_state:
			st.session_state.llm = load_llm()


		st.session_state.llm.set_prompt(fr'''
You are an intelligent AI mdeical assistant.
Refer to the patient data given below (patient is referred to as "Subject"). It is related to a lung cancer study.

{llmdata[llmdata['Subject'] == int(st.session_state.subject_selection)][demographic_cols + smoking_hist + clinical + llm_sent + ['lung_cancer']].to_dict(orient='records')}

Some fields that do have a clear description are described for your understanding below - 
bioplc - Had a biopsy related to lung cancer?
biop0 - Had a biopsy related to positive screen?
proclc - Had any procedure related to lung cancer?
can_scr - Result of screen associated with the first confirmed lung cancer diagnosis Indicates whether the cancer followed a positive negative, or missed screen, or whether it occurred after the screening years.
0="No Cancer", 1="Positive Screen", 2="Negative Screen", 3="Missed Screen", 4="Post Screening"
canc_free_days - Days until the date the participant was last known to be free of lung cancer. 
llm_sentiment - AI generated sentiment variable for cancer likeliness from 0 - 10.
lung_cancer - Actual clinical test outcome for lung cancer (0 = negative, 1 = positive)

The doctor has also added some notes upon examining the CT scan data.
Doctor's Notes - {st.session_state.dc_notes}

The patient has also made some observations on their own, which they have noted down below
Patient's Remarks/Observations - {st.session_state.pt_notes}

Based on this data, another doctor will be interacting with you and ask you some questions. Answer these questions. 
Answer them as per your knowledge and understanding. Keep your answers highly verbose and descriptive.
If any question is unrelated to lung cancer or the medical field in general, respectfully decilne to answer that question.
''')
		
		if "messages" not in st.session_state: st.session_state.messages = []
		prompt = st.chat_input(placeholder='Summarize this patient...')

		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		if prompt:
			st.chat_message("user").markdown(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
			response = st.session_state.llm.ask(prompt)
			st.session_state.messages.append({"role": "assistant", "content": response})

			with st.chat_message("assistant"):
				st.markdown(response)

	with metadata:
		metadata_tab()


def patient_page(patient_id:str):
	global csvdata, llmdata, doc_notes
	if 'edited_data' not in st.session_state: st.session_state.edited_data = {}
	if 'pil_images' not in st.session_state: st.session_state.pil_images = []
	if 'current_prediction' not in st.session_state: st.session_state.current_prediction = None
	if 'uploaded_files_list' not in st.session_state: st.session_state.uploaded_files_list = []

	st.session_state.subject = patient_id
	st.image(image=logo_img, use_container_width=True)

	with st.sidebar:
		st.header(body='Patient\'s Portal')
		st.write(datetime.now(EST_TIME).strftime("%d %b %Y %I:%M %p") + ' EST')
		st.write(f'Welcome {st.session_state.subject}')
		st.session_state.model_selection = st.selectbox(label='Classifier', options=list(modelpaths.keys()))

		# show_hist = st.toggle('Show History')
		clinical_data = st.toggle(label='Clinical Data')
		demographic_data = st.toggle(label='Demographic Data')
		smoking_history = st.toggle(label='Smoking History')
		st.divider()
		show_reports = st.toggle('View Diagnostic Plots')
		show_preds = st.toggle('View Prediction History')

		logout = st.button(label='Logout', use_container_width=True)

		if logout:
			st.session_state.edited_data = {}
			st.session_state.pil_images = []
			st.session_state.current_prediction = None
			st.session_state.uploaded_files_list = []
			st.session_state.chat_history = []
			st.session_state.login = False
			st.rerun()

	st.title('Patient Dashboard')

	info, diagnostics, history, ai, metadata = st.tabs(['Information', 'Diagnostics', 'My History and Results', 'Talk To Virtual Doctor', 'Metadata'])

	with info:
		info_tab()


	with diagnostics:
		edited_data = {}
		original_columns = csvdata.columns.tolist()
		c1, c2, c3 = st.columns(3)

		original = csvdata[csvdata['Subject'] == int(patient_id)]

		def process_data(section_name, columns):
			"""Handles editing and storing modified data while preserving structure."""
			display_names = {
				'age': 'Patient Age',
				'ethnic': 'Ethnicity',
				'gender': 'Gender',
				'height': 'Height (in)',
				'race': 'Race',
				'weight': 'Weight (lbs)',
				'age_quit': 'Quit Smoking Age',
				'cigar': 'Smoked Cigar',
				'pipe': 'Smoked Pipe',
				'pkyr': 'Pack Years',
				'smokeage': 'Smoking Onset Age',
				'smokeday': 'Avg Days Smoked',
				'smokelive': 'Lives With Smoker(s)',
				'smokework': 'Works With Smoker(s)',
				'smokeyr': 'Total Smoking Years',
				'biop0': 'Biopsy Type',
				'bioplc': 'Cancer Biopsy',
				'can_scr': 'Screened for Cancer',
				'canc_free_days': 'Days Without Cancer',
				'proclc' : 'Cancer Procedure',
				'cigsmok' : 'Currently Smoking'
			}
			
			slice_df = csvdata[['Subject'] + columns]
			data_df = slice_df[slice_df['Subject'] == int(patient_id)].T
			data_df.columns = ['Value']
			
			# Apply display names to the index
			data_df.index = data_df.index.map(lambda x: display_names.get(x, x))
			
			# Editable dataframe
			edited_df = st.data_editor(data_df, use_container_width=True)
			
			# Convert display names back to original column names
			reverse_mapping = {v: k for k, v in display_names.items()}
			edited_df.index = edited_df.index.map(lambda x: reverse_mapping.get(x, x))
			
			# Store edited values while avoiding duplicate 'Subject' columns
			edited_df = edited_df.T
			edited_df = edited_df.drop(columns=['Subject'], errors='ignore')
			edited_df.insert(0, 'Subject', patient_id)  # Ensure 'Subject' is the first column
			
			edited_data[section_name] = edited_df


		with c1:
			if clinical_data:
				st.write('Clinical Data')
				process_data('Clinical', clinical)

		with c2:
			if demographic_data:
				st.write('Demographic Data')
				process_data('Demographic', demographic_cols)

		with c3:
			if smoking_history:
				st.write('Smoking History')
				process_data('Smoking History', smoking_hist)

		if edited_data:
			final_edited_df = pd.concat(edited_data.values(), axis=1)

			# Remove duplicate columns (keeping the first occurrence)
			final_edited_df = final_edited_df.loc[:, ~final_edited_df.columns.duplicated()]
			final_edited_df = final_edited_df.reindex(columns=original_columns, fill_value=None)
			final_edited_df = final_edited_df.fillna(original.set_index('Subject').loc[int(patient_id)])

		st.markdown('Doctor\'s Notes')
		st.session_state.dc_notes = doc_notes[doc_notes['Subject'] == int(patient_id)]['comments'].values[0]
		st.code(body=st.session_state.dc_notes)

		st.session_state.pt_notes = st.text_area(label='My Observations',
											     value=pat_notes[pat_notes['Subject'] == int(patient_id)]['Remark'].values[0])
		
		save_obs = st.button(label='Save Observations', use_container_width=True)


	with history:
		# new_pred = st.toggle('Generate My Prediction Results')
		if clinical_data and demographic_data and smoking_history:
			new_X = final_edited_df[feature_cols + demographic_cols + smoking_hist + llm_sent]
			new_results = generate_outcome(subject=patient_id, 
										classifier=st.session_state.model_selection, 
										full_row=new_X)
			st.markdown(new_results)
			st.divider()

		else:
			st.info('Please enable CSV data to view results.')

		if show_reports:
			col1, col2 = st.columns(2)

			with col1:
				st.image(image=os.path.join('shap_plots', f'{patient_id}.png'), caption='SHAP Summary')

			with col2:
				plot_path = os.path.join('saliency_plots', f'{patient_id}.png')

				if os.path.exists(plot_path):
					st.image(image=plot_path, caption=os.path.basename(plot_path).split('.')[0])

				else:
					st.image(image=os.path.join('images', 'notfound.jpg'), caption='Failed to Load')

			st.divider()

		if show_preds:
			st.subheader('Diagnostic Prediction History')
			records = db.fetch(subject_id=int(patient_id))

			if len(records) > 0:
				for entry in records:
					df = pd.DataFrame([entry])
					imagelist = entry.get("Images", [])  # Extract image list or default to empty list
					history_writer(df, imagelist)

			else:
				st.info('No older records are currently present.')


	with ai:
		if 'llm' not in st.session_state:
			st.session_state.llm = load_llm()

		st.session_state.llm.set_prompt(f'''
You are a helpful AI assistant. Your task is to respond to the patient's queries to the best of your knowledge.
Refer to patient info given below.
{llmdata[llmdata['Subject'] == int(patient_id)][demographic_cols + smoking_hist + clinical + llm_sent + treatments + ['lung_cancer']].to_dict(orient='records')}
Answer them as per your knowledge and understanding. Keep your answers highly descriptive.
If any question is unrelated to the data, respectfully decilne to answer that question stating your reason.
Give suggestions for diagnosis and treatment based on these columns - treament_categories, treatment_types, treatment_days

Some fields that do have a clear description are described for your understanding below - 
bioplc - Had a biopsy related to lung cancer?
biop0 - Had a biopsy related to positive screen?
proclc - Had any procedure related to lung cancer?
can_scr - Result of screen associated with the first confirmed lung cancer diagnosis Indicates whether the cancer followed a positive negative, or missed screen, or whether it occurred after the screening years.
0="No Cancer", 1="Positive Screen", 2="Negative Screen", 3="Missed Screen", 4="Post Screening"
canc_free_days - Days until the date the participant was last known to be free of lung cancer. 
llm_sentiment - AI generated sentiment variable for cancer likeliness from 0 - 10.
lung_cancer - Actual clinical test outcome for lung cancer (0 = negative, 1 = positive)

A doctor has also added some comments/notes after examining the patient's CT scans
Doctor's Notes - {st.session_state.dc_notes}

The patient has also made some observations on their own, which they have noted down below
Patient's Remarks/Observations - {st.session_state.pt_notes}

If any patient tested negative (lung_cancer == 0), that means they do no need any further treatment.
Do not refer to the patient by their number, instead use natural language by referring to them as 'You'.
Also, from the given data always assume that the patient is yet to receive the treatment and/or diagnosis, and respond accordingly.
Eg. It is suggested that you ... so on.
'''.strip())

		if "messages" not in st.session_state: st.session_state.messages = []
		prompt = st.chat_input(placeholder='What treatment is suggested?')

		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		if prompt:
			st.chat_message("user").markdown(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
			response = st.session_state.llm.ask(prompt)
			st.session_state.messages.append({"role": "assistant", "content": response})

			with st.chat_message("assistant"):
				st.markdown(response)

	with metadata:
		metadata_tab()

def main():
	
	if 'login' not in st.session_state: st.session_state.login = False
	if 'user' not in st.session_state: st.session_state.user = None
	
	if not st.session_state.login:
		st.image(image=logo_img, use_container_width=True)
		st.title('Login')
		
		username = st.text_input(label='Username/Patient ID')
		password = st.text_input(label='Password', type='password')
		col1, col2 = st.columns(2)
		
		with col1:
			if st.button("Doctor", use_container_width=True):
				st.session_state.user = "Doctor"
		
		with col2:
			if st.button("Patient", use_container_width=True):
				st.session_state.user = "Patient"
		
		if username and password and st.session_state.user:
			if st.session_state.user == "Doctor" and username == st.secrets["keys"]["username"] and password == st.secrets["keys"]["password"]:
				st.session_state.login = True
				st.session_state.user = "Doctor"
				# st.rerun()

			elif st.session_state.user == "Patient" and username.strip().isnumeric():
				tmp = int(username.strip())
				
				if tmp in st.session_state.subject_list and password == st.secrets["keys"]["password"]:
					st.session_state.login = True
					st.session_state.subject = username.strip()
					# st.rerun()

				else:
					st.error("Invalid Patient ID or password")
			else:
				st.error("Invalid credentials or user type selection!")
	
	elif st.session_state.login and st.session_state.user == "Doctor":
		if 'chat_history' not in st.session_state: st.session_state.chat_history = []
		doctor_page()
	
	elif st.session_state.login and st.session_state.user == "Patient":
		if 'chat_history' not in st.session_state: st.session_state.chat_history = []
		# st.session_state.scans = load_image(str(st.session_state.subject))
		patient_page(st.session_state.subject)


if __name__ == "__main__":
	csvdata = utilloader('classifier_csv')
	llmdata = utilloader('llm_csv')
	doc_notes = utilloader('doctor_notes')
	pat_notes = utilloader('patient_notes')
	Manager = utilloader('manager')
	db = utilloader('database')

	st.session_state.subject_list = list(csvdata['Subject'])
	
	# st.session_state.login = True
	# st.session_state.user = 'Doctor'
	# patient_page('100158')
	
	main()