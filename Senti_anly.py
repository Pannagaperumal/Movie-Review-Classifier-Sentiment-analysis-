import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
print(tf.__version__)

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1",url,untar=True,cache_dir='.',cache_subdir='')
dataset_dir=os.path.join(os.path.dirname(dataset),'aclImdb')

train_dir= os.path.join(dataset_dir,'train')
os.listdir(train_dir)

remove_dir =os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
os.listdir(dataset_dir)

batch_size=32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',batch_size=batch_size,validation_split = 0.2,subset='training',seed=seed
)

'''
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review",text_batch.numpy()[i])
        print("Label",label_batch.numpy()[i])
        '''
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed = seed
)
raw_test_ds=tf.keras.utils.text_dataset_from_directory('aclImdb/test',
batch_size=batch_size)

def custom_standardisation(input_data):
    lowercase=tf.strings.lower(input_data)
    stripped_html= tf.strings.regex_replace(lowercase,'<br />',' ')
    return tf.strings.regex_replace(stripped_html,'[%s]' %re.escape(string.punctuation),
    '')        

max_features = 10000
sequence_length=250
vectorize_layer=layers.TextVectorization(
    standardize=custom_standardisation, max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

train_text= raw_train_ds.map(lambda x,y:x)
vectorize_layer.adapt(train_text)
def vectorize_text(text,label):
    text=tf.expand_dims(text, -1)
    return vectorize_layer(text),label
text_batch, label_batch=next(iter(raw_train_ds))
first_review, first_label=text_batch[0], label_batch[0]
'''
print("Review",first_review)
print("label", raw_train_ds.class_names[first_label])
print("vectorized Review", vectorize_text(first_review, first_label))
print("1287 ----->",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---->",vectorize_layer.get_vocabulary()[313])
print("vocabulary size:{}".format(len(vectorize_layer.get_vocabulary())))
'''
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds=raw_test_ds.map(vectorize_text)

AUTOTUNE= tf.data.AUTOTUNE

train_ds=train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding= 16
model = tf.keras.Sequential([
   #vectorize_layer,
    layers.Embedding(max_features + 1,embedding),
    layers.Dropout(0.2),
    layers.GlobalAvgPool1D(),
    layers.Dropout(0.2),
    layers.Dense(1),
    #layers.Activation('sigmoid')
    ])
model.summary()

model.compile(
        optimizer='adam',
loss=losses.BinaryCrossentropy(from_logits=True),
metrics=['BinaryAccuracy'])
epochs=10
history= model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

loss , accuracy= model.evaluate(test_ds)
print("loss: ",loss)
print("Accuracy: ",accuracy)





final_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
final_model.compile(loss=losses.BinaryCrossentropy(from_logits=False),optimizer='adam',metrics=['accuracy'])
''''optional Learning Graphs
history_dict=history.history
history_dict.keys()
acc = history_dict['binary_accuracy']
val_acc= history_dict['val_binary_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']
epochs=range(1, len(acc)+1)
#bo for blue dot
plt.plot(epochs, loss, 'bo', label='Training loss')
#b for blue solid line
plt.plot(epochs, val_loss,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#bo for blue dot
plt.plot(epochs, acc, 'bo', label='Training loss')
#b for blue solid line
plt.plot(epochs, val_acc,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
'''
####USSER INTERFACE######3###################
import numpy as np
examples = [
    "the movie was great",
    "the movie was somewhat ok",
    "the movie was terrible"
]
q=input("enter a review: \n")
examples.append(q)
p=final_model.predict(examples)
print(p[-1])
if p[-1]<=0.5:
    print("This review is quite negative!!")
else:
    print("this review is quite possitive!!")    
    
