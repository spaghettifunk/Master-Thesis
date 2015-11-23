# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0008_auto_20151123_2233'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='usersentencevowelstrend',
            name='last_edit',
        ),
    ]
