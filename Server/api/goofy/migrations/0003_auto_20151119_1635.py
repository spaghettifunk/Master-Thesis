# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0002_auto_20151119_1617'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userhistory',
            name='user_id',
        ),
        migrations.AddField(
            model_name='userhistory',
            name='username',
            field=models.CharField(default=datetime.datetime(2015, 11, 19, 16, 35, 5, 632496, tzinfo=utc), max_length=50),
            preserve_default=False,
        ),
    ]
