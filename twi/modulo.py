import pandas as pd
import numpy as np
import os
import csv
import datetime
import matplotlib.pyplot as plt

def get_users_with_tweets(folder_tweets):
        archivo=open(folder_tweets, 'rb' )
        reader = csv.reader(archivo,delimiter='\t')
        tabla=[]
        for line in reader:
            tabla.append(line)
        sujetos={}
        for item in reversed(tabla[1:]):
            if item[0] not in sujetos:
                sujetos[item[0]]={"tweets":[],"lugar":os.path.splitext(folder_tweets)[0].split("_")[-1]}
            tweet=str.split(item[2])
            sujetos[item[0]]["tweets"].append([item[1],tweet])
        return sujetos
def number_users(s):
        return len(s)

def tweets_for_user(s,user_id):
        if user_id in s:
            return len(s[user_id]["tweets"])
        else:
            return "User with Id="+user_id+" not found"

def mean_std_tweets_for_users(s):
        keys=s.keys()
        values=[]
        for k in keys:
            values.append(len(s[k]["tweets"]))
        mean=np.mean(values)
        std=np.std(values)
        return mean,std

def fecha_min_max_by_city(s):
    fecha_min=datetime.datetime.today()    
    fecha_max=datetime.datetime.strptime("1950-01-01",'%Y-%m-%d')
    for user in s:
        for tweet in s[user]["tweets"]:
            fecha_tweet=datetime.datetime.strptime(tweet[0],'%Y-%m-%d')
            if fecha_tweet < fecha_min:
                fecha_min =fecha_tweet
            if fecha_tweet > fecha_max:
                fecha_max =fecha_tweet
    return fecha_min,fecha_max

def time_serie_user(user,palabras,fmin,fmax):
    delta=(fmax-fmin).days
    timeserie=np.zeros(delta+1)
    for tweet in user["tweets"]:
        fecha_tweet=datetime.datetime.strptime(tweet[0],'%Y-%m-%d')
        comun=len(set(palabras).intersection(tweet[1]))
        valor= 0
        if comun >=1:
            valor=1
        index=(fecha_tweet-fmin).days
        if timeserie[index] == 0:
            timeserie[index]=valor
    return timeserie