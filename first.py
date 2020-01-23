#!/usr/bin/env python3.7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)  # Give us all the columns without truncation

##############################  Games   #########################################
# steamid, appid, playtime_2weeks, playtime_forever, dateretrieved

gamesFilename1 = "data/Games_1.csv"
games1 = pd.read_csv(gamesFilename1, header=0, low_memory=False)
gamesFilename2 = "data/Games_2.csv"
games2 = pd.read_csv(gamesFilename2, header=0, low_memory=False)

games = pd.concat([games1, games2])
games.reset_index(drop=True, inplace=True)

games.drop(['dateretrieved'], inplace=True, axis=1)
###### Active Games
#games = games[pd.notnull(games['playtime_forever'])]
###### Make sure no null
games = games[pd.notnull(games['steamid'])]
games = games[pd.notnull(games['appid'])]
#print(games)
##################################################################################

################### Game Info Lookup Data Files###################################
####  Games_Developers
# appid, Developer
####  Games_Genres
# appid, Genre
####  Games_Publishers
# appid, Publisher
####  App_ID_Info:
# appid, title, type (game, mod, dlc, hardware etc), Required_Age, Is_multiplayer, Price, Rating


appInfoFile = "data/App_ID_Info.csv"
appInfo = pd.read_csv(appInfoFile, header=0, low_memory=False)
#drop_list = ['title', 'Required_Age']
#appInfo.drop(drop_list, inplace=True, axis=1)
appInfo = appInfo[pd.notnull(appInfo['appid'])]
appInfo = appInfo[pd.notnull(appInfo['Price'])]

###################################################################################

##################### Player Info Lookup #########################################
#### Player_Summaries
# steamid, personaname, profileurl, avatar, avatarmedium, avatarfull, personastate, communityvisibilitystate,
# profilestate, lastlogoff, commentpermission, realname, primaryclanid, timecreated, gameid, gameserverip,
# gameextrainfo, cityid, loccountrycode, locstatecode, loccityid, dateretrieved
playerFile = "data/Player_Summaries.csv"
playerData = pd.read_csv(playerFile, header=0, low_memory=False)

#drop some cols -> 'avatar' 'avatarmedium' 'avatarfull' 'profileurl' 'primaryclanid'
#Need to remove columns of data that do not refer to games - eg other Steam software

drop_list = ['personaname', 'avatar', 'avatarmedium', 'avatarfull', 'profileurl', 'primaryclanid']
playerData.drop(drop_list, inplace=True, axis=1)
playerData = playerData[pd.notnull(playerData['gameid'])]
playerDataCountry = playerData[pd.notnull(playerData['loccountrycode'])]
playerDataCountry.reset_index(drop=True, inplace=True)
#print(data.head())
#print(data.dtypes)
#print(list(data.columns.values))
#print(dataCut["gameid"])
###################################################################################

################ Social Network ##############################
####  Friends
# steamid_a, steamid_b, friend_since
##############################################################


###################### Not too sure on use ################################
####  Groups
# steamid, groupid, dateretrieved
####  Achievement_Percentages   #### steamid vs achievement would be useful...
# appid, Name, Percentage
############################################################################

############## Player Demographics #########################################
###### Countries

playerCountries = playerDataCountry.loccountrycode.unique()
playerCountriesUniqueCount = playerCountries.size
colNames = ['Country', 'Percentage of Players']
playerCountriesCount = pd.DataFrame(columns=colNames)
for i in range(playerCountriesUniqueCount-1):
    playerCountriesCount.loc[i] = [playerCountries[i]] + [len(playerDataCountry[(playerDataCountry.loccountrycode == playerCountries[i])])]
playerCountriesCountTotal = playerCountriesCount['Percentage of Players'].sum()
playerCountriesCount['Percentage of Players'] = (playerCountriesCount['Percentage of Players']*100) / playerCountriesCountTotal
playerCountriesCount = playerCountriesCount.sort_values(by=['Percentage of Players'], ascending=False)
playerCountriesCount.reset_index(drop=True, inplace=True)
print(playerCountriesCount)

#Plotting
#labels = playerCountriesCount['Country']
#sizes = playerCountriesCount['Percentage of Players']
#patches, texts = plt.pie(sizes, shadow=True, startangle=90)
#plt.legend(patches, labels, loc="best")
#plt.axis('equal')
#plt.show()

###### Money Spent
## Look at all games with appid and steamid -> morph steamid to country code, morph game to price

# games: df with all owned games: steamid, appid
# gameIDs = games[['steamid', 'appid']].copy()
# playerDataCountry: df with steamid and loccountrycode
# gameLocs = playerDataCountry[['steamid', 'loccountrycode']].copy()
# appInfo: appid, Price, Rating
# gameInfos = appInfo[['appid', 'Price']].copy()

#colNames = ['appid', 'steamid', 'loccountrycode', 'Price', 'Rating']
gameMoney = pd.DataFrame()#columns=colNames)

# ###### Naive approach:
# for i in range(games.size-1):
#    gameMoney.loc[i] = [games.appid[i]] + \
#                       [games.steamid[i]] + \
#                       [playerDataCountry.loccountrycode[playerDataCountry.steamid == games.steamid[i]]] + \
#                       [appInfo.Price[appInfo.appid == games.appid[i]]] + \
#                       [appInfo.Rating[appInfo.appid == games.appid[i]]]

# The above takes a long time. Numpy might be quicker for this dataset but does not scale well. Dictionary mapping or
# merging is probably the way to go

# Therefore, the better approach is to create a DF such that:
# [ games.appid, games.steamid, dictionary1, dictionary2, dictionary3]
# dictionary1 - key steamid and loccountrycode from playerDataCountry
# dictionary2 - key appid and price from appInfo
# dictionary3 - key appid and Rating from appInfo
# Therefore need the following DF:
# [games.appid, games.steamid, games.steamid, games.appid, games.appid ]

gameMoney.insert(0, "appid", games.appid)
gameMoney.insert(1, "steamid", games.steamid)
gameMoney.insert(2, "loccountrycode", np.nan)
gameMoney.insert(3, "Price", np.nan)
gameMoney.insert(4, "Rating", np.nan)

dic1 = playerDataCountry['loccountrycode']
dic1.index = playerDataCountry['steamid']
dictionary1 = dic1.to_dict()

dic2 = appInfo['Price']
dic2.index = appInfo['appid']
dictionary2 = dic2.to_dict()

dic3 = appInfo['Rating']
dic3.index = appInfo['appid']
dictionary3 = dic3.to_dict()

gameMoney['loccountrycode'] = games.steamid.map(dictionary1).fillna(gameMoney['loccountrycode'])
gameMoney['Price'] = games.appid.map(dictionary2).fillna(gameMoney['Price'])
gameMoney['Rating'] = games.appid.map(dictionary3).fillna(gameMoney['Rating'])

gameLocPrice = gameMoney.drop(['appid','steamid','Rating'], axis=1)

gameLocPrice = gameLocPrice[pd.notnull(gameMoney['loccountrycode'])]
gameLocPrice = gameLocPrice[pd.notnull(gameMoney['Price'])]
gameLocPrice.reset_index(drop=True, inplace=True)
#


gameLocPriceCountries = gameLocPrice.loccountrycode.unique()
gameLocPriceCountriesUniqueCount = gameLocPriceCountries.size
colNames = ['Country', 'Total_Revenue', 'Revenue_Per_Player']
gameLocPriceCount = pd.DataFrame(columns=colNames)
for i in range(gameLocPriceCountriesUniqueCount-1):
    sumRev = (gameLocPrice['Price'][(gameLocPrice.loccountrycode == gameLocPriceCountries[i])]).sum()
    countRev = len(gameLocPrice['Price'][(gameLocPrice.loccountrycode == gameLocPriceCountries[i])])
    gameLocPriceCount.loc[i] = [gameLocPriceCountries[i]] + [sumRev] + [sumRev/countRev]

gameLocPriceCount = gameLocPriceCount.sort_values(by=['Total_Revenue'], ascending=False)
gameLocPriceCount.reset_index(drop=True, inplace=True)
print(gameLocPriceCount)

gameLocPriceCount = gameLocPriceCount.sort_values(by=['Revenue_Per_Player'], ascending=False)
gameLocPriceCount.reset_index(drop=True, inplace=True)
print(gameLocPriceCount)
