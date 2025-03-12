import google.generativeai as genai
import streamlit as st
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
		print(f'sleep_duration {max(sleep_duration, 0)}')
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
