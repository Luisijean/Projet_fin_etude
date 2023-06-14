from django.shortcuts import render
import spacy
import numpy as np
import requests
from .models import *
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import csv
import unittest
import snscrape.modules.twitter as sntwitter
import numpy as np


columns = ['index','mobile_number', 'churn', 'Predictions','Propension de Churn(%)','Classement']

df= pd.read_csv('cnn/static/store/csv/Churn_CR_Django.csv', sep= ',',names=columns)

print(df)

def projet(request):    
    
	context = locals()
	template = 'Projet.html'
	return render(request,template,context)

def modelisation(request):

	num=[]
	for ob in df['mobile_number'].values:
		num.append(str(ob))
	response = None


	churn=[]
	for ob in df['churn'].values:
		churn.append(str(ob))
	response = None

	pred=[]
	for ob in df['Predictions'].values:
		pred.append(str(ob))
	response = None

	prop=[]
	for ob in df['Propension de Churn(%)'].values:
		prop.append(str(ob))
	response = None

	clas=[]
	for ob in df['Classement'].values:
		clas.append(str(ob))
	response = None

	list_churn0=[num,churn,pred,prop,clas]
	list_churn1=np.array(list_churn0)
	list_churn=np.transpose(list_churn1)

	if request.POST.get('nlp_search'):
		numero=request.POST.get('search_title')
		response = int(request.POST.get('search_title'))
		df2=df[df['mobile_number'] == response]
		response=str(df2['Propension de Churn(%)'].values).strip('[]')
		classement2=str(df2['Classement'].values).strip('[]')

	context = locals()
	template = 'Modelisation.html'
	return render(request,template,context)

def simulation(request):    

	context = locals()
	template = 'Simulation.html'
	return render(request,template,context)

def about(request):    
    

	context = locals()
	template = 'About.html'
	return render(request,template,context)

def analyse(request):    
    

	context = locals()
	template = 'Analyse.html'
	return render(request,template,context)

def index_movie_prediction(request):

	context = locals()
	template = 'Index.html'
	return render(request,template,context)



def result(request):

	context = locals()
	template = 'Result.html'
	return render(request,template,context)




