import pandas as pd
import streamlit as st

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


class DBManager():
	def __init__(self):
		username = st.secrets['keys']['mongouser']
		password = st.secrets['keys']['mongoauth']
		uri = f"mongodb+srv://{username}:{password}@cluster0.iygkqrj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

		self.connected = False
		self.client = MongoClient(uri, server_api=ServerApi('1'))

		try:
			self.client.admin.command('ping')
			self.connected = True
			print('Ping to DB success!')

		except Exception as e:
			print(e, 'Failed to connect remote DB')

		if self.connected:
			self.db = self.client['pulmoaid_db'] 
			self.collection = self.db['pulmoaid_collection']


	def check_status(self):
		return self.connected


	def retry_connection(self):
		try:
			self.client.admin.command('ping')
			self.connected = True
			print('Ping to DB success!')

		except Exception as e:
			print(e, 'Failed to connect remote DB')

		if self.connected:
			self.db = self.client['pulmoaid_db'] 
			self.collection = self.db['pulmoaid_collection']

		return self.connected


	def fetch(self, subject_id:int):
		if self.connected:
			records = list(
					self.collection.find({"Subject": subject_id}, {"_id": 0})
					.sort("Timestamp", 1)  # Sort by timestamp in ascending order
				)
			
			return records
		return []


	def save(self, data:dict):
		if self.connected:
			self.collection.insert_one(data)
			return True
		
		print('Warning! DB not connected.')
		return False


	def purge_all(self):
		if self.connected:
			userin = input('Delete all records (y/n)? ')

			if userin == 'y':
				self.collection.delete_many({})
				print('All records deleted.')
				return True
			
			else:
				print('Deletion aborted.')

		return False

if __name__ == "__main__":
	db = DBManager()
	patient_data = db.fetch(100158)
	print(patient_data)

	# db.purge_all()
