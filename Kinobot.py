import pickle
import telebot
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import os
import configparser


good = pd.read_csv('good_.tsv')#, delimiter ='\t')

curr_path = os.path.dirname(os.path.abspath(__file__))  
config = configparser.ConfigParser()
config.read(curr_path +  r'\bot.ini')

for acc in config.sections(): 
    print("Bot '", acc, "' ready to start")
    TOKEN = config[acc]['TOKEN']

class  Rnd_Sel():
    def __init__(self, model, samp_size):
        self.model = model
        self.samp_size = samp_size
    def fit(self):
        pass
    
    def predict(self, qw):
        dist, indx = self.model.kneighbors(qw, n_neighbors=self.samp_size)
        answ = good.reply.loc[np.random.choice(indx[0])]
        return answ

with open('pipe_kn_11.mod','rb') as fm:
    pipe_kn_11 = pickle.load(fm)

#print(pipe_kn_11.predict(['Откуда дровишки?']))
#print(pipe_kn_11.predict([TEST]))

bot = telebot.TeleBot(TOKEN)
#bot.delete_webhook()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Hi there! How are you!')
    
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, pipe_kn_11.predict([message.text]))
        
bot.polling()
