from django.conf.urls import url
from . import views

urlpatterns = [
	    url(r'^$', views.index_movie_prediction, name='index_movie_prediction'),
	    url(r'^modelisation$', views.modelisation, name='modelisation'),
	    url(r'^projet$', views.projet, name='projet'),
	    url(r'^simulation$', views.simulation, name='simulation'),
	    url(r'^about$', views.about, name='about'),
	    url(r'^analyse$', views.analyse, name='analyse'),
	    url(r'^result$', views.result, name='result'),
	    
	]