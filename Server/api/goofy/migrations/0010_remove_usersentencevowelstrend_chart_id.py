# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0009_remove_usersentencevowelstrend_last_edit'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='usersentencevowelstrend',
            name='chart_id',
        ),
    ]
