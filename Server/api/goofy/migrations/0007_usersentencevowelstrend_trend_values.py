# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0006_usersentencevowelstrend'),
    ]

    operations = [
        migrations.AddField(
            model_name='usersentencevowelstrend',
            name='trend_values',
            field=models.TextField(default=datetime.datetime(2015, 11, 23, 15, 34, 17, 169975, tzinfo=utc)),
            preserve_default=False,
        ),
    ]
