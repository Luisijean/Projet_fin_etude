from django.db import models

# Create your models here.
class Article(models.Model):
   num = models.CharField(max_length=100)
   amt = models.CharField(max_length=100)
   churn = models.CharField(max_length=50)