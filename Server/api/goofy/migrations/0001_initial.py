# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GeneralScore',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('total_score', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('username', models.CharField(unique=True, max_length=50)),
                ('password', models.CharField(max_length=50)),
                ('gender', models.CharField(max_length=5)),
                ('nationality', models.CharField(max_length=50)),
                ('occupation', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='UserScore',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('sentence', models.CharField(max_length=100)),
                ('score', models.IntegerField()),
                ('stress_list', models.TextField()),
                ('phonemes_list', models.TextField()),
                ('user_id', models.ForeignKey(to='goofy.User')),
            ],
        ),
        migrations.AddField(
            model_name='generalscore',
            name='user_score_id',
            field=models.ForeignKey(to='goofy.User'),
        ),
    ]
