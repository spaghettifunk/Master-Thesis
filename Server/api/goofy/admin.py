from django.contrib import admin

# Register your models here.
from .models import User, UserHistory

admin.site.register(User)
admin.site.register(UserHistory)