from django.db import models

# Create your models here.
class User(models.Model):
    username = models.CharField(max_length=50, null=False, blank=False, unique=True)
    password = models.CharField(max_length=50, null=False, blank=False) #EncryptedCharField(max_length=255)
    gender = models.CharField(max_length=5, null=False, blank=False)
    nationality = models.CharField(max_length=50, null=False, blank=False)
    occupation = models.CharField(max_length=50, null=False, blank=False)

    def __str__(self):
        return self.username

class UserScore(models.Model):
    user_id = models.ForeignKey(User)
    sentence = models.CharField(max_length=100, null=False, blank=False)
    score = models.IntegerField(null=False, blank=False)
    stress_list = models.TextField(null=False)
    phonemes_list = models.TextField(null=False)

    def __str__(self):
        return self.sentence

class GeneralScore(models.Model):
    user_score_id = models.ForeignKey(User)
    total_score = models.IntegerField(null=False, blank=False)

    def __str__(self):
        return self.user_score_id