import re
import google.generativeai as genai
import streamlit as st
import pandas as pd
from time import time, sleep

API = st.secrets["keys"]["api"]
genai.configure(api_key=API)

class LLM():
	def __init__(self, api_key:str):
		print('Init LLM')
		self.api = api_key
		self.limit = 8000
		self.temp = 1
		self.sys_prompt = """
		You are an intelligent medical report diagnostics AI. 
		Based on the provided information about the patient, give a conclusion for the outcome.
		"""
		self.last_invoke = 0
		self.context = []
		self.subject_id = None

		self.config = {
			"temperature": self.temp,
			"top_p": 0.95,
			"top_k": 40,
			"max_output_tokens": self.limit,
			"response_mime_type": "text/plain",
		}
		
		self.model = genai.GenerativeModel(model_name="gemini-1.5-flash", 
							  system_instruction=self.sys_prompt, 
							  generation_config=self.config)
		
	def truncate_context(self):
		new = self.context[0] + self.context[-10:]
		self.context = new

	
	def set_prompt(self, prompt:str):
		self.sys_prompt = prompt
		self.model = genai.GenerativeModel(model_name="gemini-1.5-flash", 
							  system_instruction=self.sys_prompt, 
							  generation_config=self.config)


	def set_temp(self, temp:float):
		if temp > 1:
			temp == 1

		elif temp < 0:
			temp == 0

		self.temp = temp


	def ask(self, question:str):
		sleep_duration = 60/12 - (time() - self.last_invoke)
		# print(f'sleep_duration {max(sleep_duration, 0)}')
		sleep(max(sleep_duration, 0))

		if len(self.context) > 1_000_000/self.limit:
			self.truncate_context()
		
		chat = self.model.start_chat(
		history=self.context
		)

		response = chat.send_message(question, generation_config=self.config)
		self.last_invoke = time()
		self.context.append({"role": "user", "parts": question})
		self.context.append({"role": "model", "parts": response.text})
		
		# print('LLM Response', response.text)
		return response.text
	
	def debug_vitals(self):
		print(rf'''SYS PROMPT {self.sys_prompt}''')


class ScoreGen():
	def __init__(self):
		sys_prompt = """
		You are a helpful and intelligent medical report generation AI. 
		The user will provide you with a list of paramters that correspond to a subject in the form of a Python dict. 
		This data is related to a study of potential lung cancer patients.
		Your task is to share your sentiment in a numerical format. Based on the list of parameters, give a score between 0 to 10 for cancer positivity likeliness.
		Note that all subjects have a smoking history, but not all of them tested positive. So give your score carefully.
		Units are as follows - age (years), height (inches), weight (pounds), age_quit (years, never quit smoking if -1)
		If you there are any other variables in the user input that you are not aware of, ignore them during your score generation.
		Refer to the examples given below.

		For Example: 
		Score = 10 if {
		'age': 59,
		'gender': 'Male',
		'height': 66.0,
		'weight': 178.0,
		'age_quit': -1.0,
		'cigar': 'Subject did not smoke cigars',
		'cigsmok': 1,
		'pipe': 'Subject smoked pipe',
		'pack_years': 60.0,
		'smokeage': 'Subject started smoking at age 19.0',
		'smokeday': 'Subject smoked 30 cigarettes per day on average',
		'smokelive': 'Subject lives/lived with smoker',
		'smokework': 'Subject works/worked with exposure to smokers',
		'smokeyr': 'Subject has been smoking for 40 years.',
		'procedures': "{'Radionuclide scan - Bone', 'MRI - Brain', 'Thoracotomy', 'CT - Abdomen and pelvis', 'Biopsy - Open Surgical', 'Clinical Evaluation', 'Radionuclide scan - FDG-PET scan', 'Resection'}"}

		Score = 7 if  {
		'age': 63,
		'gender': 'Male',
		'height': 76.0,
		'weight': 260.0,
		'age_quit': 60.0,
		'cigar': 'Subject did not smoke cigars',
		'cigsmok': 0,
		'pipe': 'Subject did not smoke pipe',
		'pack_years': 152.0,
		'smokeage': 'Subject started smoking at age 22.0',
		'smokeday': 'Subject smoked 80 cigarettes per day on average',
		'smokelive': 'Subject lives/lived with smoker',
		'smokework': 'Subject works/worked with exposure to smokers',
		'smokeyr': 'Subject has been smoking for 38 years.',
		'procedures': "{'MRI - Brain', 'Pulmonary function tests/spirometry', 'Thoracotomy', 'Clinical Evaluation', 'Cytology - Percutaneous transthoracic', 'CT - Diagnostic chest', 'CT - Chest, limited thin section of nodule', 'Resection'}"}

		Score = 5 if  {
		'age': 69,
		'gender': 'Male',
		'height': 70.0,
		'weight': 160.0,
		'age_quit': -1.0,
		'cigar': 'Subject did not smoke cigars',
		'cigsmok': 1,
		'pipe': 'Subject did not smoke pipe',
		'pack_years': 98.0,
		'smokeage': 'Subject started smoking at age 18.0',
		'smokeday': 'Subject smoked 40 cigarettes per day on average',
		'smokelive': 'Subject lives/lived with smoker',
		'smokework': 'Subject works/worked with exposure to smokers',
		'smokeyr': 'Subject has been smoking for 49 years.',
		'procedures': "{'Cytology - Bronchoscopic', 'Biopsy - Other', 'MRI - Brain', 'Pulmonary function tests/spirometry', 'Echocardiography', 'CT - Brain', 'Biopsy - Transbronchial', 'Clinical Evaluation', 'Biopsy - Lymph node - scalene nodes', 'Biopsy - Endobronchial', 'Radionuclide scan - Fusion PET/CT scan', 'CT - Diagnostic chest', 'CT - Chest, abdomen, and pelvis', 'Radiograph - Chest', 'Radionuclide scan - FDG-PET scan'}"}

		Score = 3 if  {
		'age': 56,
		'gender': 'Male',
		'height': 71.0,
		'weight': 170.0,
		'age_quit': -1.0,
		'cigar': 'Subject did not smoke cigars',
		'cigsmok': 1,
		'pipe': 'Subject did not smoke pipe',
		'pack_years': 72.0,
		'smokeage': 'Subject started smoking at age 20.0',
		'smokeday': 'Subject smoked 40 cigarettes per day on average',
		'smokelive': 'Subject lives/lived with smoker',
		'smokework': 'Subject works/worked with exposure to smokers',
		'smokeyr': 'Subject has been smoking for 36 years.',
		'procedures': nan}

		Score = 1 if {
		'age': 58,
		'gender': 'Male',
		'height': 74.0,
		'weight': 165.0,
		'age_quit': -1.0,
		'cigar': 'Subject did not smoke cigars',
		'cigsmok': 1,
		'pipe': 'Subject smoked pipe',
		'pack_years': 61.5,
		'smokeage': 'Subject started smoking at age 17.0',
		'smokeday': 'Subject smoked 30 cigarettes per day on average',
		'smokelive': 'Subject lives/lived with smoker',
		'smokework': 'Subject works/worked with exposure to smokers',
		'smokeyr': 'Subject has been smoking for 41 years.',
		'procedures': nan}

		Check for 'smokeday' the lower the cigarettes per day the lower the score.
		Check 'cigar' and 'pipe' if the subject smokes cigars or pipe the score should be higher.
		Check 'smokeyr' the higher the years the higher the score.
		Check all the other given parameters and give a appropriate score.
		Give your answer in the following format - 
		Reason : a single line explaining your reason in short
		Score : and integer between 0 and 10 (0 and 10 included)
		"""

		self.llm = LLM(api_key=API)
		self.llm.set_prompt(sys_prompt)
		print('-- Score LLM')

	def generate_score(self, df_row:pd.DataFrame) -> int:

		try:
			response = self.llm.ask(str(df_row.to_dict()))
			score_line = re.findall(r'score.*', response, flags=re.IGNORECASE)[0]
			number = re.findall(r'\d+', score_line)[0].strip()

			if number.isnumeric():
				return min(int(number) + 1, 10)
			
			return -1
		
		except Exception as e:
			print(f'Error in score gen, response = {response}')
			return -1
