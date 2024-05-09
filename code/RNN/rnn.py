import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=192, embed_size=96):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        ## TODO:
        ## - Define an embedding component to embed the word indices into a trainable embedding space.
        ## - Define a recurrent component to reason with the sequence of data. 
        ## - You may also want a dense layer near the end...    
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=False)
        self.dense1 = tf.keras.layers.LeakyReLU(vocab_size)
        self.dense2 = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """
        X_RNN_embedding = self.embedding(inputs)
        # print('>>> debug: X_RNN_embedding.shape:', X_RNN_embedding.shape)

        output_lstm = self.lstm(X_RNN_embedding, initial_state = None) # sequence
        # output_lstm, state_h, state_c = self.lstm(X_RNN_embedding, initial_state = None) # sequence + state / state
        # output_lstm = self.lstm(X_RNN_embedding, initial_state = None) # sequence
        # print('>>> debug: output_lstm.shape:', output_lstm.shape)

        output_leaky = self.dense1(output_lstm)

        logits = self.dense2(output_leaky)
        # print('>>> debug: logits.shape:', logits.shape)

        return logits

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]])
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            logits = np.array(logits[0,0,:])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array([[out_index]])

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyRNN(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    def perplexity(y_true, y_pred):
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss = tf.reduce_mean(cross_entropy)
        return tf.exp(loss)

    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    acc_metric  = perplexity

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 4,
        batch_size = 20,
    )



#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    import preprocess
    data_path = "../data"
    train_id, test_id, vocab = preprocess.get_data(f"{data_path}/nlp_train.txt", f"{data_path}/nlp_test.txt")
    # vocab = None

    window_size = 20
    def process_RNN_data(window_size, data):
        sentence_tokenized_array = np.array(data)
        remainder = (len(sentence_tokenized_array) - 1)%window_size
        sentence_tokenized_array = sentence_tokenized_array[:-remainder]

        X_RNN = sentence_tokenized_array[:-1].reshape(-1, window_size)
        y_RNN = sentence_tokenized_array[1:].reshape(-1, window_size)
        return X_RNN, y_RNN


    X0, Y0 = process_RNN_data(window_size,train_id)
    X1, Y1 = process_RNN_data(window_size,test_id)
    print('>>> check X0.shape:', X0.shape)
    print('>>> check Y0.shape:', Y0.shape)

    ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in 'speak to this brown deep learning student'.split():
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: args.model.generate_sentence(word1, 20, vocab, 10)

if __name__ == '__main__':
    main()
