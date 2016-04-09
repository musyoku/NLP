# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import model
import vocab

data_dir = "text"
dataset = vocab.load(data_dir)
beluga = model.build()