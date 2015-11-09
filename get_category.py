

from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding("UTF-8")
import logging, nltk, re
import gensim
from gensim import corpora, models, similarities, matutils

import string
import nltk

import codecs
from collections import Counter
from collections import defaultdict

from gensim.models import LogEntropyModel

# cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;DATABASE=INN')
# cursor = cnxn.cursor()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# nltk.internals.config_java("C:/Program Files/Java/jdk1.7.0_45/bin/java.exe")
# path_to_model = "C:/stanford-postagger-2014-01-04/models/english-bidirectional-distsim.tagger"
# path_to_jar = "C:/stanford-postagger-2014-01-04/stanford-postagger.jar"
# tagger = nltk.tag.stanford.POSTagger(path_to_model, path_to_jar)

class SemanticSpace(object):
    '''
    classdocs
    write something here ...
    '''
    rebuild = False
    dims = 300

    '''
    simple settings here
    '''
    working_directory = './kb_dataset/'
    stoplist = nltk.corpus.stopwords.words('english')

    def __init__(self,file):
        self.filename = file

    def model_folder(self, filename):
        '''
        Generate model folder
        '''
        return '%s%s' % (self.working_directory, filename)

    def read_file(self):
        '''open read line'''
        self.texts=[]
        f = open(self.filename)
        f= [l for l in (line.strip() for line in f) if l]
        for line in f:
            stopwords = nltk.corpus.stopwords.words('english')
            line = re.sub('\#.*?\#','', line)
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            line = regex.sub(' ', line)
            word=line.lower().split()
            filtered_words = [w for w in word if not w in stopwords]
            self.texts.append(filtered_words)

        return self.texts

    """read item file"""
    def read_orig_file(self):
        '''open read line'''
        #self.orifilename='training5.txt'
        self.oritexts=[]
        f = codecs.open('item_ann.txt', encoding='utf-8')
        for line in f:
            self.oritexts.append(line.lower().split())
        return self.oritexts

    def build_dictionary(self,texts):
        dictionarykb=corpora.Dictionary(texts) #build dictionary
        dictionarykb.save_as_text('dict.txt')
        dictionarykb.save(self.model_folder('dictionary'))
        return dictionarykb


    def build_corpus(self,dictionarykb,texts):
        corpuskb = [dictionarykb.doc2bow(text) for text in texts] #convert all texts into vector
        corpora.MmCorpus.serialize(self.model_folder('corpus'), corpuskb)
        return corpuskb

    def build_tfidf(self,corpuskb,dictionary):
        #tfidfkb = models.TfidfModel(corpuskb) #build tfidf model
        tfidfkb = LogEntropyModel(corpuskb,id2word=dictionary)

        corpus_tfidf = tfidfkb[corpuskb] #convert all texts into tfidf model

        tfidfkb.save(self.model_folder('tfidf_model'))
        return tfidfkb,corpus_tfidf

    def build_lsi_lsa(self,corpus_tfidf,dictionary):
        lsi = gensim.models.lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
        lsi.save(self.model_folder('lsi_model'))
        lsi_sim= similarities.MatrixSimilarity(lsi[corpus_tfidf])
        lsi_sim.save(self.model_folder('lsi_sim'))

        return lsi

    def build_model(self):
        self._text= self.read_file()
        self._dictionary=self.build_dictionary(self._text)

        self._corpus=self.build_corpus(self._dictionary, self._text)
        self._tfidf,self._corpus_tfidf=self.build_tfidf(self._corpus,self._dictionary)

        self.lsi=self.build_lsi_lsa(self._corpus_tfidf,self._dictionary)

    def load_model(self):
        print 'now load model'
        global dictionary,corpus,tfidf_space,lsi,lsi_sim

        self.dictionary = corpora.Dictionary.load(self.model_folder('dictionary'))
        self.corpus = corpora.MmCorpus(self.model_folder('corpus'))
        #self.corpus_train = corpora.MmCorpus(self.model_folder('corpus'))
        self.tfidf_space = models.TfidfModel.load(self.model_folder('tfidf_model'))
        self.lsi = models.LsiModel.load(self.model_folder('lsi_model'))

        self.lsi_sim = similarities.MatrixSimilarity.load(self.model_folder("lsi_sim_ann"))

    def get_construct(self,idd):
        #print idd
        if idd=="277":
            an="Perceived Ease of Use"
            return an
        elif idd=="278":
            an="Perceived Usefulness"
            return an
        elif idd=="291":
            an="Affects towards Technology"
            return an
        elif idd=="296":
            an="Social Influence"
            return an
        elif idd=="343":
            an="Trust"
            return an


    def get_construct_name(self,text):

        # m = re.search(r"\#*(\d+)\#",text)
        # variable_id= m.group(1)
        m = re.search(r"\#(.*)\#",text)
        vid= m.group(1)

        con_name=self.get_construct(vid)
        item=self.get_item(text)
        return con_name,item

    """ filter the result to get best category id """

    def get_best_ux(self,result):
        scorelist=[]
        idlist=[]
        uxdict={}


        for items in result:
            scorelist.append(items[2])#get the similarity score
            idlist.append(items[0])#get the category id

        variance=scorelist[0]-scorelist[1] #check the variance for the first two item

        #print 'variance',variance

        """if the variance>0.2,direct get the first item as the best category"""
        if variance>0.2:
            #print 'large variance'
            cid=idlist[0]
            with open('similar_item.txt', 'a') as f:
                f.write('large variance'+'\n')


        else:
            """if variance less than threshold,take the one with highest frequency"""
            #print 'calculate frequency'
            c=Counter(elem for elem in idlist)
            with open('similar_item.txt', 'a') as f:
                f.write(str(c)+'\n')
            #print c
            m = max(v for _, v in c.iteritems())
            r = [k for k, v in c.iteritems() if v == m]

            """if have similar frequency,take the one with higher similarity score"""

            if len(r)>1:
                with open('similar_item.txt', 'a') as f:
                    f.write('share same frequency,get the construct with highest max score'+'\n')
                #print 'share same frequency,get the construct with highest max score'
                uxdict= defaultdict( list )

                """group the category id in dictionary"""

                for v,w,x in result:
                    if v in r:
                        uxdict[v].append(x)
                #print uxdict
                """take the one with maximum score"""

                for key,value in uxdict.items():
                    uxdict[key]=max(value)
                cid= max(uxdict, key=uxdict.get)

            else:

                with open('similar_item.txt', 'a') as f:
                    f.write('get the construct with highest frequency'+'\n')
                #print 'get the construct with highest frequency'
                for item in r:
                    cid=item
        #print 'cid',cid
        return cid

    def get_item(self,result):

        items=re.sub('\#.*?\#','', result)
        return items

    def get_definition(self,id):

        f = codecs.open('C:/behavior/pb/definition.txt', encoding='utf-8')
        for line in f:
            cid=self.get_construct_name(line)
            if cid==id:
                define=line=re.sub('\#.*?\#','', line)

        return define

    def get_category(self,oriline):

        num_topic=6

        self.texts=[]
        self.result_final=[]
        self.text=[]

        #print 'testing item',oriline

        with open('similar_item.txt', 'a') as f:
            f.write(oriline+'\n')

        m = re.search(r"\#*(\d+)\#",oriline)#extract variable id

        """preprocessing"""

        stopwords = nltk.corpus.stopwords.words('english')
        # line=re.sub('\*.*?\*','', oriline)
        line=re.sub('\#.*?\#','', oriline)
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        line = regex.sub(' ', line)

        word=line.lower().split()
        self.text = [w for w in word if not w in stopwords]
        line=' '.join(self.text)


        self.result=self.doc_similarity(line,num_topic)#lsi

        with open('similar_item.txt', 'a') as f:
            f.write(str(self.result)+'\n')

        #self.best_ux=self.get_best_ux(self.result)  # filter the lsi result,return the best category id
        #self.item=self.get_item(self.result)
        #self.define=self.get_definition(self.best_ux)
       


        # with open('similar_item.txt', 'a') as f:
        #     f.write('Final id:::'+str(self.best_ux)+'\n'+'\n')
        #self.result_final.append((self.vname,self.define))
        # with open("predict_behavior.txt", 'a') as f:
        #         f.write('@'+self.best_ux+'@'+oriline+'\n')

        return self.result

    def doc_similarity(self,query,num_topic):


        x=int(num_topic)
        result=[]
        result2=[]


        self._oritext= self.read_orig_file()#read the file projected to lsi as comparison to query

        """converting query"""

        vec_bow = self.dictionary.doc2bow(query.lower().split())
        doc_bow = self.tfidf_space[vec_bow]
        vec_lsi = self.lsi[doc_bow]

        """print topics generated from query"""
        #for doc in vec_lsi:
            #print doc

        """print topics generated from training data"""

        #self.corpus_train = corpora.MmCorpus(self.model_folder('corpustfidf'))
        #vec_lsi2 = self.lsi[self.corpus_train]
        #for doc in vec_lsi2:
            #print doc

        """find the similarity"""

        sims = self.lsi_sim[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        for a in list(enumerate(sims[:x])):
            #print a
            self.similarity_score=a[1][1]
            self.doc=' '.join(self._oritext[a[1][0]])
            #print self.doc
            #s=self.doc.decode('cp1252')
            #self.doc=s.encode('utf8')

            """get construct id,construct name,category name,category id"""
            self.con_name,self.item=self.get_construct_name(self.doc)

            #s=str(self.formula).decode('cp1252')
            #self.formula=s.encode('utf8')
            #result.append((self.similarity_score,self.doc,self.con_id,self.con_name,self.cat_name))  # save result in tuple
            
            result.append((self.con_name,self.item,self.similarity_score))

            result2.append((self.con_name))
            result3=list(set(result2))

        return result

