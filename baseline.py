# -*- coding: utf-8 -*-
#这是SemEval当年的baseline程序，这儿只截取了部分

import entity
import preProcessing
import copy
import xml.etree.ElementTree as ET

#AspectTerm抽取器
class BaselineAspectExtractor():

    #初始化时,获取语料中最频繁的Aspect Term
    def __init__(self, corpus):
        self.candidates = [a.lower() for a in corpus.top_aspect_terms]

    #找到Aspect Term的offsetd
    def find_offsets_quickly(self, term, text):
        start = 0
        while True:
            #在text中查找Aspect Term
            start = text.find(term, start)
            if start == -1: return
            #使用yield生成迭代器元素,用于对某个AspectTerm的多次匹配
            yield start
            start += len(term)

    #找到某个Aspect Term的offset(可能重复出现)
    def find_offsets(self, term, text):
        offsets = [(i, i + len(term)) for i in list(self.find_offsets_quickly(term, text))]
        #返回(start,end)的list
        return offsets

    #找到句子中所有的AspectTerm(采用匹配的方法)
    def tag(self, test_instances):
        #记录Aspect Term+offset元组
        clones = []
        for i in test_instances:
            #拷贝对象（深拷贝）
            i_ = copy.deepcopy(i)
            #记录Aspect Term
            i_.aspect_terms = []
            #遍历所有可能的Aspect Term
            for c in set(self.candidates):
                #如果文本中含有此Aspect Term
                if c in i_.text:
                    offsets = self.find_offsets(' ' + c + ' ', i.text)
                    for start, end in offsets: i_.addAspectTerm(term=c,
                                                                  offset={'from': str(start + 1), 'to': str(end - 1)})
            clones.append(i_)
        #对每个句子转化为Instance的list并返回
        return clones
        
def fd2(counts):
    '''Given a list of 2-uplets (e.g., [(a,pos), (a,pos), (a,neg), ...]), form a dict of frequencies of specific items (e.g., {a:{pos:2, neg:1}, ...}).'''
    d = {}
    for i in counts:
        # If the first element of the 2-uplet is not in the map, add it.
        if i[0] in d:
            if i[1] in d[i[0]]:
                d[i[0]][i[1]] += 1
            else:
                d[i[0]][i[1]] = 1
        else:
            d[i[0]] = {i[1]: 1}
    return d
    
#nltk中的停用词表
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
   
#字符串相似度计算公式
def dice(t1,t2,stopwords=[]):
    #对句子进行分词并统计全部不同的词
    tokenize = lambda t: set([w for w in t.split() if (w not in stopwords)])
    t1,t2=tokenize(t1),tokenize(t2)
    #intersection是指交集
    return 2. * len(t1.intersection(t2)) / (len(t1) + len(t2))
     
#AspectTerm极性评估器
class BaselineAspectPolarityEstimator():
    
    def __init__(self, corpus):
        self.corpus = corpus#一个Corpus对象
        self.fd = fd2([(a.term, a.pol) for i in self.corpus.corpus for a in i.aspect_terms])
        self.major = entity.freq_rank(entity.fd([a.pol for i in self.corpus.corpus for a in i.aspect_terms]))[0]

    # 使用knn进行聚类，其中文本距离用dice来计算
    def k_nn(self, text, aspect, k=5):
        neighbors = dict([(i, dice(text, next.text, stopwords)) for i, next in enumerate(self.corpus.corpus) if
                          aspect in next.get_aspect_terms()])
        ranked = entity.freq_rank(neighbors)
        topk = [self.corpus.corpus[i] for i in ranked[:k]]
        return entity.freq_rank(entity.fd([a.pol for i in topk for a in i.aspect_terms]))

    def majority(self, text, aspect):
        if aspect not in self.fd:
            return self.major
        else:
            polarities = self.k_nn(text, aspect, k=5)
            if polarities:
                return polarities[0]
            else:
                return self.major

    #对测试集进行标注
    def tag(self, test_instances):
        clones = []
        for i in test_instances:
            i_ = copy.deepcopy(i)
            for j in i_.aspect_terms: j.polarity = self.majority(i_.text, j.term)
            clones.append(i_)
        return clones
        
#评估器类
class Evaluate():
    #初始化，先把
    def __init__(self, correct, predicted):
        self.size = len(correct)
        #correct和predict是两个Instance的list
        self.correct = correct
        self.predicted = predicted

    # Aspect Extraction (no offsets considered)
    #评估Aspect Term（offset不参与评估）
    def aspect_extraction(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        #遍历每个句子
        for i in range(self.size):
            #正确的aspect term集合
            cor = [a.offset for a in self.correct[i].aspect_terms]
            #预测的aspect term集合
            pre = [a.offset for a in self.predicted[i].aspect_terms]
            #正确预测的aspect term个数
            common += len([a for a in pre if a in cor])
            #预测的aspect term总个数
            retrieved += len(pre)
            #实际的aspect term个数
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, common, retrieved, relevant
        
    #aspect term的极性评估，使用准确率（aspect term是已知的）
    def aspect_polarity_estimation(self, b=1):
        common, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.pol for a in self.correct[i].aspect_terms]
            pre = [a.pol for a in self.predicted[i].aspect_terms]
            common += sum([1 for j in range(len(pre)) if pre[j] == cor[j]])
            retrieved += len(pre)
        acc = common / retrieved
        return acc, common, retrieved
        
def baselineExamForResturant():
    corpus=preProcessing.loadXML('../../data/ABSA_2014_origin/Restaurants_Train_v2.xml')
    print('开始切割数据集')
    train,seen=corpus.split()
    
    corpus.write_out('tmp/train.xml', train, short=False)
    traincorpus = entity.Corpus(ET.parse('tmp/train.xml').getroot().findall('sentence'))
    
    corpus.write_out('tmp/test.gold.xml', seen, short=False)
    seen = entity.Corpus(ET.parse('tmp/test.gold.xml').getroot().findall('sentence'))
    
    corpus.write_out('tmp/test.xml', seen.corpus)
    unseen = entity.Corpus(ET.parse('tmp/test.xml').getroot().findall('sentence'))
    
    print('开始统计和抽取AspectTerm')
    b1=BaselineAspectExtractor(traincorpus)
    predicted = b1.tag(unseen.corpus)
    corpus.write_out('tmp/test.predicted-aspect.xml', predicted, short=False)
    print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' % Evaluate(seen.corpus,
                                                                                                     predicted).aspect_extraction())
        
if __name__=='__main__':
    print('开始试验')
    baselineExamForResturant()
    