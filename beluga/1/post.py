# -*- coding: utf-8 -*-
import os, sys, time, codecs, urllib, urllib2, cookielib
import numpy as np
import model
import vocab
from config import config

def post(text):
	u, p = 'usernmae', '********'

	opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookielib.CookieJar()))
	opener.addheaders.append(("Cookie", "404_not_found=97e9e4c173d1692af91f362714294d0f8f8f8fa5b569e68be82462c51e08d2fd; |-_-|=9303; _ga=GA1.2.68549008.1429454566"))

	post = {
		"text": text.encode("utf-8", "xmlcharrefreplace"),
		"authenticity_token": "b55ca781c3fc46bba9ec07ce24b30219c482414c176a6d86eb98139bd00555dc",
		"hashtag_title":"事務室"
	}
	data = urllib.urlencode(post)
	conn = opener.open("http://beluga.fm/i/statuses/update.json", data)
	print conn.read().decode("utf-8")

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

data_dir = "text"
model_dir = "model"
dataset, n_vocab, n_dataset = vocab.load(data_dir)
lm = model.build(n_vocab)
lm.load(model_dir)

for phrase in xrange(10):
	lm.reset_state()
	str = ""
	char = dataset[np.random.randint(0, n_dataset)][0]
	for n in xrange(1000):
		str += vocab.id_to_word(char)
		dist = lm.distribution(char, test=True)[0]
		id = np.random.choice(np.arange(n_vocab, dtype=np.uint8), 1, p=dist)[0]
		if id == 0:
			break
		char = id
	print str
	post(str)
	time.sleep(60)

