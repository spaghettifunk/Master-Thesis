# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0003_auto_20151119_1635'),
    ]

    operations = [
        migrations.AddField(
            model_name='userhistory',
            name='date',
            field=models.DateField(default=datetime.datetime(2015, 11, 19, 21, 31, 8, 18157, tzinfo=utc)),
            preserve_default=False,
        ),
    ]
