"""
15-110 Hw6 - Social Media Analytics Project
Name: Joanne Tsai
AndrewID: chihant
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### WEEK 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

def makeDataFrame(filename):
    filename_df = pd.read_csv(filename)
    return filename_df

def parseName(fromString):
    startIndex = fromString.find(": ") + len(": ")
    endIndex = fromString.find(" (")
    name = fromString[startIndex:endIndex]
    return name

def parsePosition(fromString):
    startIndex = fromString.find(" (") + len(" (")
    endIndex = fromString.find(" from")
    position = fromString[startIndex:endIndex]
    return position

def parseState(fromString):
    startIndex = fromString.find("from ") + len("from ")
    endIndex = fromString.find(")")
    position = fromString[startIndex:endIndex]
    return position

endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
def findHashtags(message):
    hashtagList = []
    for charIndex in range(len(message)):
        if message[charIndex] == "#":
            hashEndIndex = charIndex + 1
            while hashEndIndex < len(message) and message[hashEndIndex] not in endChars:
                hashEndIndex += 1
            hashtagList.append(message[charIndex:hashEndIndex])
    return hashtagList

def getRegionFromState(stateDf, state):
    row = stateDf.loc[stateDf['state'] == state, 'region']
    value = row.values[0]
    return value

def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []

    for index, row in data.iterrows():
        names.append(parseName(row["label"]))
        positions.append(parsePosition(row["label"]))
        states.append(parseState(row["label"]))
        regions.append(getRegionFromState(stateDf, parseState(row["label"])))
        hashtags.append(findHashtags(row["text"]))
    data['name'] = names
    data['position'] = positions
    data['state'] = states
    data['region'] = regions
    data['hashtags'] = hashtags

    return None


### WEEK 2 ###

def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"

def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentimentsList = []
    for index, row in data.iterrows():
        message = row["text"]
        sentimentsList.append(findSentiment(classifier, message))
    data['sentiment'] = sentimentsList

    return None

def getDataCountByState(data, colName, dataToCount):
    d = {}
    for index, row in data.iterrows():
        state = row['state']
        if colName == "" and dataToCount == "":
            if state not in d:
                d[state] = 1
            else:
                 d[state] += 1
        else:
            col = row[colName]
            if col == dataToCount:
                if state not in d:
                    d[state] = 1
                else:
                    d[state] += 1
    return d


def getDataForRegion(data, colName):
    d = {}
    for index, row in data.iterrows():
        region = row['region']

        if region not in d:
            d[region] = {}
        col = row[colName]

        if col not in d[region]:
            d[region][col] = 1
        else:
            d[region][col] += 1

    return d


def getHashtagRates(data):
    d = {}
    for index, row in data.iterrows():
        for hashtag in row['hashtags']:
            if hashtag not in d:
                d[hashtag] = 1
            elif hashtag in d:
                d[hashtag] += 1
    return d


def  mostCommonHashtags(hashtags, count):
    d = {}
    i = 1
    while i <= count:
        highestCountHashtag = None
        currentBiggestCount = 0
        for hashtag in hashtags:
            if hashtags[hashtag] > currentBiggestCount and hashtag not in d:
                currentBiggestCount = hashtags[hashtag]
                highestCountHashtag = hashtag

        d[highestCountHashtag] = currentBiggestCount
        i += 1
    return d


def getHashtagSentiment(data, hashtag):
    messageCount = 0
    score = 0
    for index, row in data.iterrows():
        if hashtag in findHashtags(row['text']):
            messageCount += 1
            if row['sentiment'] == "positive":
                score += 1
            elif row['sentiment'] == "negative":
                score += -1
            elif row['sentiment'] == "neutral":
                score += 0
    return score/messageCount



### WEEK 3 ###

import matplotlib.pyplot as plt

def graphStateCounts(stateCounts, title):
    statesLst = []
    statesValueLst = []
    for state in stateCounts:
        statesLst.append(state)
        statesValueLst.append(stateCounts[state])
    fig, ax = plt.subplots()
    ind = range(len(statesValueLst))
    rects1 = ax.bar(ind, statesValueLst)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(statesLst)
    plt.xticks(rotation = 'vertical')
    plt.show()
    return None

def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    d = {}
    i = 1
    while i <= n:
        currentHighestRate = 0
        currentHighestState = None
        for state in stateCounts:
            for state in stateFeatureCounts:
                if stateFeatureCounts[state]/stateCounts[state] > currentHighestRate and state not in d:
                    currentHighestRate = stateFeatureCounts[state]/stateCounts[state]
                    cuurentHighestState = state
        d[cuurentHighestState] = currentHighestRate
        i += 1
    graphStateCounts(d, title)
    return None


def graphRegionComparison(regionDicts, title):
    featureLst = []
    regionLst = []
    regionFeatureLst = []
    for region in regionDicts:
        for feature in regionDicts[region]:
            if feature not in featureLst:
                featureLst.append(feature)
    for region in regionDicts:
        regionLst.append(region)
    for region in regionDicts:
        tempLst = []
        for i in featureLst:
            if i in regionDicts[region]:
                tempLst.append(regionDicts[region][i])
            else:
                tempLst.append(0)
        regionFeatureLst.append(tempLst)
    sideBySideBarPlots(featureLst, regionLst, regionFeatureLst, title)
    return None


def graphHashtagSentimentByFrequency(df):
    dict = getHashtagRates(df)
    mostCommonHashtagsDict = mostCommonHashtags(dict, 50)
    hashtagsLst = []
    frequenciesLst = []
    sentimentScoresLst = []
    for hashtag in mostCommonHashtagsDict:
        hashtagsLst.append(hashtag)
        frequenciesLst.append(mostCommonHashtagsDict[hashtag])
        sentimentScoresLst.append(getHashtagSentiment(df, hashtag))

    scatterPlot(frequenciesLst, sentimentScoresLst, hashtagsLst, "Hashtag Sentiments to Frequencies")
    return None


#### WEEK 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    x = np.arange(len(xLabels)) # gets the indexes of the bars
    width = 0.8 / len(labelList)  # the width of the bars
    fig, ax = plt.subplots()
    for index in range(len(valueLists)):
        ax.bar(x - 0.4 + width*(index+0.5), valueLists[index], width, label=labelList[index])
    ax.set_xticks(x)
    ax.set_xticklabels(xLabels)
    plt.xticks(rotation="vertical")
    ax.legend()
    plt.title(title)
    fig.tight_layout()
    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)
    plt.title(title)
    plt.ylim(-1, 1)
    plt.show()


### RUN CODE ###

#This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
