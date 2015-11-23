# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0005_userhistory_chart_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserSentenceVowelsTrend',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('username', models.CharField(max_length=50)),
                ('sentence', models.CharField(max_length=100)),
                ('vowel', models.CharField(max_length=5)),
                ('chart_id', models.CharField(max_length=50)),
                ('last_edit', models.DateField()),
                ('trend', models.BinaryField()),
            ],
        ),
    ]
