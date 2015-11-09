from get_category import SemanticSpace
from training_vector import SubSpace
import re

"""get semantic training data and initiate class"""
c=SemanticSpace('semantic_training_data.txt')

"""BUILD SEMANTIC SPACE"""
#c.build_model()

"""BUILD ITEM SUBSPACE"""

"""get item file and initiate class"""
# d=SubSpace("item_ann.txt")
# d.load_model()
# d.build_tfidf_corpus()

"""LOAD MODEL"""
c.load_model()

print "############################################################################"

"""get query in review sentence"""

doc='this phone is useful'
print "Review sentence:::",doc

"""get item list (construct, item, similarity score)"""
#print '##item similarity##'
result=c.get_category(doc)
for item in result:
    print item

"""predict best ux"""

best_ux= c.get_best_ux(result)
print best_ux

print "####################################################################################"

"""get query in review text"""

doc2="This phone is useful. I love it."
print "Review Text:::",doc2

"""Split the text into review sentence"""

m = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', doc2)
for sent in m:
    print "Review sentence:::",sent

    """get item list (construct, item, similarity score)"""

    result=c.get_category(sent)
    for item in result:
        print "Similar measurement items",item

    """predict best ux"""

    best_ux= c.get_best_ux(result)
    print "Best predicted user experience:::",best_ux
###add on text



"""find similar item"""
"""
print '##item similarity##'
result=c.doc_similarity(doc)
for item in result:
    print item[1],'\t',item[0],'\t',item[2]
"""
