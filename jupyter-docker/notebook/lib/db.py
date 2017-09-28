import pandas as pd
import MySQLdb as my
import re
import os
from sets import Set
from sqlalchemy import create_engine
pd.set_option('display.width', 1000)

#------ GEM METHODS START
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import nltk
from nltk.corpus import wordnet
from nltk import metrics, stem, tokenize
from nltk.metrics import *
from nltk.corpus import wordnet
from difflib import SequenceMatcher
import pandas as pd

import sframe

stemmer = stem.PorterStemmer()

def normalize(s):
    if s and (s.lower != 'empty') and (s.lower() != 'none') and (s is not None):
        try:
            words = tokenize.wordpunct_tokenize(s.lower().strip())
            return ' '.join([stemmer.stem(w) for w in words])
        except:
            return ''
    else:
        return None
    
def fuzzy_match(s1, s2, max_dist=3):
    dist = edit_distance(s1, s2)
    return  dist

def strict_match(s1, s2):
    if s1==s2:
        return 0
    else:
        return 1000000


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def nlp_wordnet(s1, s2):
    wordFromList1 = wordnet.synsets(s1)
    wordFromList2 = wordnet.synsets(s2)
    if wordFromList1 and wordFromList2: #Thanks to @alexis' note
        return wordFromList1[0].wup_similarity(wordFromList2[0])
    else:
        return None

def fuzzy_match_list(s, ss, method=strict_match, verbose=False):
    res = None
    pmin = 1000000
    for s1 in ss:
        cmin = method(s, s1)
        if cmin < pmin:
            res = s1
            pmin = cmin
    if verbose:
        print 'matching: ' + s + ' Found:  ' + str(pmin)
    return res

def split_name(x):
    y=str(x).split(']-', 1)
    if ( y and len(y) > 1):
        return (y[1] or None)
    else:
        return None

def split_cols(df):
    for col in df.column_names():
        df[col] = df[col].apply(split_name)
        
def cleanse(df, cols=[]):
    cols = (cols or df.column_names())
    for col in cols:
#         df['l1'] = df['l1'].apply(lambda x: None if (not x or x == 'EMPTY' or x == 'none') else x.replace('...', ''))
        df['c_'+col] = df[col].apply(normalize)
    return df[cols + ['c_'+c for c in cols]]

def fuzz_naive(s1, s2):
    return fuzz.ratio(s1, s2)

def fuzz_partial(s1, s2):
    if s1 and s2:
        return fuzz.partial_ratio(s1, s2)
    else:
        return None

def fuzz_ratio(s1, s2):
    return max([
       fuzz.ratio(s1, s2),
       fuzz.partial_ratio(s1, s2)
    ])


def get_one(s_str, s_list, verbose=False):
    r_str, ratio =  process.extractOne(s_str, s_list)
    res = (r_str, ratio) if ratio > 90 else None
    if verbose:
        print(str(s_str) + " -> " + str(res))
    return res

def nltk_dist(s1, s2):
    return nltk.edit_distance(s1, s2)

def get_max(s_str, s_list, method=fuzz_naive, verbose=False):
#     if(s_str and (s_str.lower() != 'none') and (s_str.lower() != 'empty')  and (s_str.lower() != 'empti')):
    if s_str:
        res =  max([(s,method(s_str, s)) for s in s_list], key=lambda item:item[1])
    else:
        res = None
#     if verbose:
#         print(str(s_str) + " -> " + str(res))
    return res


#------ OLD METHODS
def short_bn(bn, n=2):        
    return ":".join(bn.split("|")[-n:])

def cleanse(str):
    return re.sub(" +"," ",re.sub("[^0-9a-z ]"," ",str.lower())).strip()

def last_n(sentence, w=5):
        return " ".join(cleanse(sentence).split(" ")[-w:])

def word_match_list(s1,s2,stem=4):
    #print(s1)
    #print(s2)
    return max([ word_match(s1,x) for x in s2 ])

def word_match(s1,s2,stem=4):
    s1=Set([x[0:stem] for x in cleanse(s1).split(" ") ])
    s2=Set([x[0:stem] for x in cleanse(s2).split(" ") ])
    return 1.0*len(s1 & s2)/len(s1 | s2)

def load_sql(sql):
	conn = my.connect('127.0.0.1','root','root','ml')
	return pd.read_sql(sql, con=conn)

def load_csv(csv):
	return pd.read_csv(csv, sep="\t")

class MyDB:
    conn = None

    def connect(self):
        self.conn = my.connect('127.0.0.1','root','root','ml')

    def query(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            self.conn.commit()
        except (AttributeError, my.OperationalError):
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(sql)
            self.conn.commit()
        return cursor

db = MyDB()
def execute_sql(sql):
    db.query(sql)

def to_sql(df, tbl_name, db='ml'):
	engine = create_engine("mysql+mysqldb://root:"+'root'+"@127.0.0.1/"+db)
	df.to_sql(con=engine, name=tbl_name, if_exists="append", chunksize=1000, index=False)
    

def manual_override(bn_id_and_lca, **kwargs):
    bn_id, lca = bn_id_and_lca.split("$$", 2)
    return add_manual_override(bn_id, lca, **kwargs)

def add_manual_override(bn_id, lca, amazon_root, score=1, reason="override", view=-1):
    insert_sql = """
        INSERT INTO ml.manual_evaluation (bn_id,lca,man_score_override, man_score_override_reason, amazon_root)
        VALUES ('{0}', '{1}', '{2}', '{3}', '{4}')
        ON DUPLICATE KEY UPDATE man_score_override='{2}', man_score_override_reason='{3}';
    """.format(bn_id, lca, score, reason, amazon_root)
    #print(insert_sql)
    execute_sql(insert_sql)
    if view == 0:
        select_sql = "select * from ml.manual_evaluation where bn_id={0}".format(bn_id)
        return load_sql(select_sql) 
    elif view > 0:
        select_sql = "select * from ml.manual_evaluation"
        return load_sql(select_sql); 
    else:
        return True

def escape_str(input_str, escape_char="'", encoding='utf-8'):
    output_str=input_str
    if(isinstance(input_str, basestring)):
        # Might look wierd but works!
        output_str = input_str.replace("\\"+escape_char, escape_char).replace(escape_char,"\\"+escape_char).encode(encoding)
    return output_str

def add_amazon_item(item):
    values = [item["id"],item["bn_id"],item["title"], -2]
    values = [escape_str(value) for value in values]
    insert_sql = """
        INSERT INTO crawls.amazon_products (id, bn_id, title, source_id)
        VALUES ('{0}', '{1}', '{2}', '{3}')
        ON DUPLICATE KEY UPDATE id='{0}';
    """.format(*values)
    #print(insert_sql)
    execute_sql(insert_sql)
    return True
