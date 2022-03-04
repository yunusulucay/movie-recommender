import numpy as np
import keras.backend as K
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers import Layer, InputSpec

##################################
# 1- Clustering using Cast People#
##################################

class prepare_text_data:
    
    def __init__(self, df):
        self.df = df

    def prepare_dataset(self):
        """
        df : df_movies_data for this project
        Return:
        cast_listed( list type )
        """
        cast_listed = []
        for director_name, actor_1, actor_2, actor_3 in zip(self.df["director_name"], 
                                                            self.df["actor_1_name"], 
                                                            self.df["actor_2_name"], 
                                                            self.df["actor_3_name"]):

            cast_listed.append("|".join([actor_1, actor_2, actor_3, director_name]))
        self.cast_listed = cast_listed
        
    def token(self, text):
        """
        Return splitted tokens.
        Return:
        list()
        """
        return(text.split("|"))

    def return_lists(self):
        """
        Return lists depending on plot_keywords, genres and director, actor features.
        Return:
        keywords_list, genres_list, cast_list
        """

        count_vectorizer = CountVectorizer(max_features=100, tokenizer=self.token)
        keywords = count_vectorizer.fit_transform(self.df["plot_keywords"])
        keywords_list = ["keyword_"+i for i in count_vectorizer.get_feature_names()]

        count_vectorizer = CountVectorizer(tokenizer=self.token)
        genres = count_vectorizer.fit_transform(self.df["genres"])
        genres_list = ["genres_"+i for i in count_vectorizer.get_feature_names()]

        count_vectorizer = CountVectorizer(max_features=100, tokenizer=self.token)
        casts = count_vectorizer.fit_transform(self.cast_listed)
        cast_list = ["cast_"+i for i in count_vectorizer.get_feature_names()]
        self.kw, self.gn, self.cl = keywords, genres, casts

    def final_predict_data(self):
        cluster_data = np.hstack([self.kw.todense(), self.gn.todense(), self.cl.todense()*2])
        # all_names_list = keywords_list+genres_list+cast_list

        return cluster_data
    
##################################    
# 2- Clustering with AutoEncoder #
##################################

def get_word_data(texts):
    max_words = 10000
    maxlen = 1500 #only use this number of most frequent words
    
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) # transforms strings in list of intergers
    word_index = tokenizer.word_index # calculated word index
    print(f"{len(word_index)} unique tokens found")

    data = pad_sequences(sequences, maxlen=maxlen) #transforms integer lists into 2D tensor
    return data

# Defining autoencoder

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric auto-encoder model.
  
    dims: list of the sizes of layers of encoder like [500, 500, 2000, 10]. 
          dims[0] is input dim, dims[-1] is size of the latent hidden layer.

    act: activation function
    
    return:
        (autoencoder_model, encoder_model): Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    
    input_data = Input(shape=(dims[0],), name='input')
    x = input_data
    
    # internal layers of encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # latent hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)

    x = encoded
    # internal layers of decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # decoder output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    
    decoded = x
    
    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model     = Model(inputs=input_data, outputs=encoded, name='encoder')
    
    return autoencoder_model, encoder_model

# Defining clustering layer 

class ClusteringLayer(Layer):
    '''
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the 
    probability of the sample belonging to each cluster. 
    The probability is calculated with student's t-distribution.
    '''

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        ''' 
        student t-distribution, as used in t-SNE algorithm.
        It measures the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
       
        inputs: the variable containing data, shape=(n_samples, n_features)
        
        Return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        '''
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure all of the values of each sample sum up to 1.
        
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T