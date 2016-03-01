from nltk import word_tokenize
from nltk import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


class PorterStemmerTokenizer(object):
    """A tokenizer that also stems tokens using a porter stemmer"""
    def __init__(self):
        self.non_alphanum_regex = re.compile('[^ 0-9a-zA-Z]')
        self.porter = PorterStemmer()
    def __call__(self, doc):
        doc = self.non_alphanum_regex.sub(' ', doc)
        tokens_alpha = word_tokenize(doc.lower())
        return [self.porter.stem(t) for t in tokens_alpha]

def combine_title_feat(df):
    df.loc[df.loc[:, 'VAR_0404']=='-1', 'VAR_0404'] = ' '
    df.loc[df.loc[:, 'VAR_0493']=='-1', 'VAR_0493'] = ' '
    df.loc[df.loc[:, 'VAR_0404']==-1, 'VAR_0404'] = ' '
    df.loc[df.loc[:, 'VAR_0493']==-1, 'VAR_0493'] = ' '

    df.VAR_0404.fillna(' ', inplace=True)
    df.VAR_0493.fillna(' ', inplace=True)
    df['COMBINED_TITLE'] = df['VAR_0404']+' '+df['VAR_0493']
    return df
        
def gen_text_feats_from_title_columns(train, test):
    vec = CountVectorizer(tokenizer=PorterStemmerTokenizer(), 
                          min_df=10, ngram_range=(1, 3), binary=True)

    train = combine_title_feat(train)
    test = combine_title_feat(test)
    
    txttrain = vec.fit_transform(train.COMBINED_TITLE)
    txttest = vec.transform(test.COMBINED_TITLE)
    
    train = train.drop(['VAR_0404', 'VAR_0493', 'COMBINED_TITLE'], axis=1)
    test = test.drop(['VAR_0404', 'VAR_0493', 'COMBINED_TITLE'], axis=1)
    
    return train, test, txttrain, txttest, vec.get_feature_names()
    
    