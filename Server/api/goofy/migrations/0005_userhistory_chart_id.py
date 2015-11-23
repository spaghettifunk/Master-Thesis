# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0004_userhistory_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='userhistory',
            name='chart_id',
            field=models.CharField(default=datetime.datetime(2015, 11, 22, 23, 25, 58, 310355, tzinfo=utc), max_length=50),
            preserve_default=False,
        ),
    ]
