# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goofy', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserHistory',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('sentence', models.CharField(max_length=100)),
                ('vowels', models.BinaryField()),
                ('user_id', models.ForeignKey(to='goofy.User')),
            ],
        ),
        migrations.RemoveField(
            model_name='generalscore',
            name='user_score_id',
        ),
        migrations.RemoveField(
            model_name='userscore',
            name='user_id',
        ),
        migrations.DeleteModel(
            name='GeneralScore',
        ),
        migrations.DeleteModel(
            name='UserScore',
        ),
    ]
