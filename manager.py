import streamlit as st
import joblib

import os, pydicom, joblib
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
join = os.path.join

import warnings
warnings.simplefilter('ignore')


CLASSIFIER  = joblib.load(os.path.join("classifiers", "FusionModel LR_97.41.pkl"))
CLINICAL_DATA = pd.read_csv(os.path.join("data", "all_features_combined_new.csv"))


# Load VGG Model

torch.manual_seed(0)
device = torch.device('cpu')

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# def classifier_loader(model_name:str):
# 	models =  {
# 		"Logistic Regression":r'A:\Software Projects\NLST-App\free_text\classifiers\FusionModel LR_97.41.pkl',
# 		"KNN":r'A:\Software Projects\NLST-App\free_text\classifiers\FusionModel KNN_73.28.pkl',
# 		"Naive Bayes":r'A:\Software Projects\NLST-App\free_text\classifiers\FusionModel NB_78.45.pkl',
# 		"Random Forest":r'A:\Software Projects\NLST-App\free_text\classifiers\FusionModel RFC_91.38.pkl',
# 		"XGBosst":r'A:\Software Projects\NLST-App\free_text\classifiers\FusionModel XGB_92.24.pkl'
# 	}

# 	return joblib.load(models[model_name])


class LungCancerVGG16Fusion(nn.Module):
	def __init__(self):
		super(LungCancerVGG16Fusion, self).__init__()
		vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
		for param in vgg16.features[:20].parameters():
			param.requires_grad = False
			
		self.features = vgg16.features
		self.avgpool = vgg16.avgpool
		
		self.classifier1 = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(4096, 1024),
			nn.ReLU(True),
			nn.Dropout(0.5)
		)
		
		# New fusion layer ----------------------------------------------------------------------
		self.fusion_layer = nn.Linear(1024, 8)
		self.classifier2 = nn.Linear(8, 2)
		
	def forward(self, x, return_features=False):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier1(x)
		
		# Get fusion features
		fusion_features = self.fusion_layer(x)
		
		# Get final prediction
		output = self.classifier2(fusion_features)
		
		if return_features:
			return output, fusion_features
		return output

	
def file_to_tensor(path:str):
	device = torch.device('cpu')

	if path.lower().endswith('.dcm'):
		dicom_data = pydicom.dcmread(path)
	
		pixel_array = dicom_data.pixel_array
		pixel_array_normalized = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
		pixel_array_normalized = pixel_array_normalized.astype(np.uint8)

		pil_image = Image.fromarray(pixel_array_normalized).convert('RGB')
		
		image_tensor = transform(pil_image)
		image_tensor = image_tensor.unsqueeze(0)
		
		return image_tensor.to(device)
	
	elif path.lower().endswith('.jpg') or \
		path.lower().endswith('.jpeg') or \
		path.lower().endswith('.png'):

		image = Image.open(path)
		image_tensor = transform(image)
		image_tensor = image_tensor.unsqueeze(0)
	
		return image_tensor.to(device)
	
	else:
		raise NotImplementedError(f'Unable to process file type {path}')

	
def reduce_to_n(filelist:list, n:int) -> list:
	if len(filelist) <= n:
		return filelist

	# Base percentages for left and right
	base_left_percentage = 0.15  # 10%
	base_right_percentage = 0.3  # 20%
	
	if len(filelist) > 250:
		base_left_percentage = 0.35  # 10%
		base_right_percentage = 0.3

	# Increment of 5% for both
	increment = 0.05
	left_percentage = base_left_percentage + increment  # 15%
	right_percentage = base_right_percentage + increment  # 25%

	# Calculate total deletions needed
	total_to_delete = len(filelist) - n

	# Calculate deletions based on adjusted percentages
	delete_from_left = int(total_to_delete * left_percentage / (left_percentage + right_percentage))
	delete_from_right = total_to_delete - delete_from_left

	# Slice the list
	reduced_list = filelist[delete_from_left:len(filelist) - delete_from_right]
	
	return reduced_list


def get_fusion_features_simple(model, input_tensor):
	model.eval()  # Set to evaluation mode
	with torch.no_grad():
		_, features = model(input_tensor, return_features=True)
	
	return features


class DataManager():
	def __init__(self, vgg_model, imagedir='', csvfile=os.path.join("data", "all_features_combined_new.csv")):
		self.vgg = vgg_model
		self.df = pd.read_csv(csvfile)
		self.imagedir = imagedir
		self.images = []
		self.row = pd.DataFrame()
		self.subject = None


	def load_images(self, subject, count=32):
		temp = []

		for root, dirs, files in os.walk(self.imagedir):
			for file in files:
				if file.lower().endswith('.jpg') and str(subject) in file:
					temp.append(os.path.join(root, file))

		self.images = reduce_to_n(temp, count)
		del temp


	def load_row(self, subject):
		self.row = self.df[self.df['Subject'] == subject]
		return self.row


	def set_subject(self, subject):
		self.subject = subject
		self.load_images(subject)
		self.load_row(subject)


	def extract_features(self, imagelist:str) -> list:
		# if len(self.images) < 1 or self.row.empty:
		# 	raise NotImplementedError(f'Image and/or csv row has not been loaded.')
		
		if imagelist and isinstance(imagelist[0], Image.Image):
		# if imagelist:
			self.vgg.eval()
			all_outcomes = []

			for image in imagelist:
				image_tensor = transform(image)
				image_tensor = image_tensor.unsqueeze(0)
				_, features = self.vgg.forward(image_tensor, return_features=True)
				all_outcomes.append(features.tolist()[0])

			df = pd.DataFrame(all_outcomes, columns=[str(i) for i in range(len(all_outcomes[0]))])
			averages = [sum(df[column])/len(df[column]) for column in df.columns]

			return averages
		
		else:
			raise NotImplementedError('Given image list is not Image object')
	

	def classify(self, X:list,  model) -> int:
		y_pred = model.predict([X])

		return int(y_pred)
	

	def predict(self, classifier):
		if self.row.empty:
			raise NotImplementedError(f'Csv row has not been loaded.')

		slice = self.df[['Subject',
			'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
			'age', 'ethnic', 'gender', 'height', 'race', 'weight',
			'age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday',
			'smokelive', 'smokework', 'smokeyr', 'llm_sentiment'
		]]

		row = slice[slice['Subject'] == self.subject]
		row.drop(columns=['Subject'], inplace=True)

		try:
			probabilities = classifier.predict_proba(row)
			positive_probability = probabilities[0][1]  # Access the probability for the positive class (index 1) of the first instance (index 0).
			return f"Positive ({positive_probability * 100:.2f}%)"  # Format as percentage

		except AttributeError:  # Handle cases where predict_proba is not available
			try:
				output = classifier.predict(row)
				return 'Positive' if output == 1 else 'Negative'
			
			except Exception as e:
				return f"Error during prediction: {e}"

		except Exception as e: # Catch any other potential errors during prediction
			return f"Error during prediction: {e}"
		


if __name__ == "__main__":
	pass
	# VGG_16 = LungCancerVGG16Fusion().to(device)
	# print('Init Model')
	# VGG_16.load_state_dict(torch.load(r'A:\Software Projects\NLST-App\checkpoints\best_vgg16.pth', weights_only=True))
	# VGG_16.eval()
	# print('Load eval state')

	# imgpath = r"A:\Software Projects\NLST-Dataset\images\1A\100158_025.jpg"
	# img_tensor = file_to_tensor(imgpath)

	# output, features = VGG_16.forward(x=img_tensor, return_features=True)

	# print(output, features)

	# manager = DataManager(imagedir=r'A:\Software Projects\NLST-Dataset\images\1A', vgg_model=VGG_16)
	# classifier = classifier_loader('KNN')

	# manager.set_subject(100158)
	# print(manager.predict(classifier))

	# gemini = LLM(api_key=API)
	# print(gemini.ask('Hello'))
	# print(gemini.ask('Tell me more about yourself'))