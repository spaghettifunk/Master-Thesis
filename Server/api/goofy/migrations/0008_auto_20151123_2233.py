# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0007_usersentencevowelstrend_trend_values'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usersentencevowelstrend',
            name='trend',
            field=models.BinaryField(null=True, blank=True),
        ),
    ]
