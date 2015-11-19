from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=50, null=False, blank=False, unique=True)
    password = models.CharField(max_length=50, null=False, blank=False)
    gender = models.CharField(max_length=5, null=False, blank=False)
    nationality = models.CharField(max_length=50, null=False, blank=False)
    occupation = models.CharField(max_length=50, null=False, blank=False)

    def __str__(self):
        return self.username


class UserHistory(models.Model):
    username = models.CharField(max_length=50, null=False, blank=False)
    sentence = models.CharField(max_length=100, null=False, blank=False)
    date = models.DateField()
    vowels = models.BinaryField(null=False, blank=False)

    def __str__(self):
        return self.sentence
