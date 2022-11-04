import numpy as np
import operator
import math
#from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

stopwordsList1 = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 
'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 
'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 
'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 
'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 
'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 
'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 
'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 
'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 
'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 
'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 
'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 
'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 
'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 
'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 
'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 
'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 
'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 
'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 
'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 
'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer',
'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 
'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 
'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer',
'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 
'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on',
'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 
'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 
'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point',
'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 
'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 
'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 
'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 
'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 
'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 
'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 
'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 
'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 
'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 
'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 
'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 
'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 
'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 
'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 
'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 
'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 
'yours', 'z']



def cleanData(doc):
    #To get all sentances from a paragraph.
    tokenizer = RegexpTokenizer(r'[^.?!]+')  
    sentancesList = tokenizer.tokenize(doc)
    wordsDict = {}
    #To get all words from sentances.
    tokenizer = RegexpTokenizer(r'\w+\'*-*\w+')
    for i in range(len(sentancesList)):
        sentanceWordsList = tokenizer.tokenize(sentancesList[i])
        for j in range(len(sentanceWordsList)):
            word = sentanceWordsList[j]
            word = word.lower()
            word = lemma.lemmatize(word)
            if word in stopwordsList1:
                continue
            elif word in stopwordsList2:
                continue
            elif word in wordsDict:
                wordsDict[word].append(i)
            else:
                wordsDict[word] = [i]
    wordsList = list(wordsDict.keys())
    n = len(wordsList)
    m = len(sentancesList)
    termDocMatrix = np.zeros((n,m))
    for i in range(n):
        word = wordsList[i]
        dictList = wordsDict[word]
        for j in range(len(dictList)):
            termDocMatrix[i][dictList[j]] += 1
    #print(sentancesList)
    #print(wordsList)
    #print(termDocMatrix)
    return sentancesList, wordsList, termDocMatrix 

def applySVD(termDocMatrix):
    U, S, Vt = svd(termDocMatrix, full_matrices=False)
    return U, S, Vt

            
def summarizer(doc=None, k=4):
    sentancesList, wordsList, termDocMatrix = cleanData(doc)
    #print(sentancesList)
    #print(wordsList)
    #print(termDocMatrix)
    U, S, Vt = applySVD(termDocMatrix)
    #print(U)
    #print(S)
    #print(Vt)
    l = S.size
    n, m = Vt.shape
    #l is equal to n which is number of dimensions in reduced space.
    #m is number of sentances.
    scoreDict = {}
    for i in range(m):
        score = 0
        for j in range(l):
            score += S[j]*S[j]*Vt[j,i]*Vt[j,i]      
        score = math.sqrt(score)
        #score contains the square of the magnitude of the sentance vector.
        scoreDict[i] = score
    summarySentancesList = []
    ctr = k
    sortedDictList = sorted(scoreDict.items(), key=operator.itemgetter(1), reverse=True)
    for key, value in sortedDictList:
        summarySentancesList.append(key)
        ctr -= 1
        if ctr==0:
            break
    summarySentancesList.sort()
    summary = ''
    for i in range(len(summarySentancesList)):
        summary += sentancesList[summarySentancesList[i]]
        summary += '.'
        summary += '\n'
    return summary


stopwordsList2 = list(stopwords.words('english'))
lemma = WordNetLemmatizer()

def supprimer_nomber(texte):
    import re
    for i in re.findall('\d{1,}', texte):
        texte = texte.replace(i, '')
    return texte

def preproces(docs, entitie, separateur):
    docs = [supprimer_nomber(art) for art in docs ]
    return [art.replace(entitie, separateur) for art in docs ]

mot_inutils = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=mot_inutils, use_idf=True,ngram_range=(1, 5))


