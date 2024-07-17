# Imports
import re
import pickle
import nltk
import sklearn
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, roc_auc_score

# Carrega o dataset
df = pd.read_csv('dataset.csv')

# Ajustamos os labels para representação numérica
df.sentiment.replace('positive', 1, inplace = True)
df.sentiment.replace('negative', 0, inplace = True)

# Função de limpeza geral de dados
def limpa_dados(texto):
    cleaned = re.compile(r'<.*?>') 
    return re.sub(cleaned, '', texto)

# Aplica a função ao nosso dataset
df.review = df.review.apply(limpa_dados)

# Função para limpeza de caracteres especiais
def limpa_caracter_especial(texto):
    rem = ''
    for i in texto:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
            
    return rem

# Aplica a função
df.review = df.review.apply(limpa_caracter_especial)

# Função para converter o texto em minúsculo
def converte_minusculo(texto):
    return texto.lower()

# Aplica a função
df.review = df.review .apply(converte_minusculo)

nltk.download('punkt')
nltk.download('stopwords')

# Função para remover stopwords
def remove_stopwords(texto):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(str(texto))
    return [w for w in words if w not in stop_words]

df.review = df.review.apply(remove_stopwords)

# Função para o stemmer
def stemmer(texto):
    objeto_stemmer = SnowballStemmer('english')
    return " ".join([objeto_stemmer.stem(w) for w in texto])

df.review = df.review.apply(stemmer)

# Extrai o texto da avaliação (entrada)
x = np.array(df.iloc[:,0].values)

# Extrai o sentimento (saída)
y = np.array(df.sentiment.values)

# Divisão dos dados em treino e teste com proporção 80/20
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = 0.2, random_state = 0)

# Cria um vetorizador (vai converter os dados de texto em representação numérica)
vetorizador = CountVectorizer(max_features = 1000)

# Fit e transform do vetorizador com dados de treino
x_treino_final = vetorizador.fit_transform(x_treino).toarray()

# Apenas transorm nos dados de teste
x_teste_final = vetorizador.transform(x_teste).toarray()

# Cria o modelo 1
modelo_v1 = GaussianNB()

# Treina o modelo 1
modelo_v1.fit(x_treino_final, y_treino)

# Cria o modelo 2
modelo_v2 = MultinomialNB(alpha = 1.0, fit_prior = True)

# Treina o modelo 2
modelo_v2.fit(x_treino_final, y_treino)

# Cria o modelo 3
modelo_v3 = BernoulliNB(alpha = 1.0, fit_prior = True)

# Treina o modelo 3
modelo_v3.fit(x_treino_final, y_treino)

# Previsões com dados de teste
ypred_v1 = modelo_v1.predict(x_teste_final)

# Previsões com dados de teste
ypred_v2 = modelo_v2.predict(x_teste_final)

# Previsões com dados de teste
ypred_v3 = modelo_v3.predict(x_teste_final)

print("Acurácia do Modelo GaussianNB = ", accuracy_score(y_teste, ypred_v1) * 100)
print("Acurácia do Modelo MultinomialNB = ", accuracy_score(y_teste, ypred_v2) * 100)
print("Acurácia do Modelo BernoulliNB = ", accuracy_score(y_teste, ypred_v3) * 100)

# Import
from sklearn.metrics import roc_auc_score

# AUC do GaussianNB
y_proba = modelo_v1.predict_proba(x_teste_final)[:, 1]
auc = roc_auc_score(y_teste, y_proba)
print("AUC do Modelo GaussianNB =", auc)

# AUC do MultinomialNB
y_proba = modelo_v2.predict_proba(x_teste_final)[:, 1]
auc = roc_auc_score(y_teste, y_proba)
print("AUC do Modelo MultinomialNB =", auc)

# AUC do BernoulliNB
y_proba = modelo_v3.predict_proba(x_teste_final)[:, 1]
auc = roc_auc_score(y_teste, y_proba)
print("AUC do Modelo BernoulliNB =", auc)

# Texto de uma avaliação de usuário (esse texto apresenta sentimento positivo)
texto_aval = """This is probably the fastest-paced and most action-packed of the German Edgar Wallace ""krimi"" 
series, a cross between the Dr. Mabuse films of yore and 60's pop thrillers like Batman and the Man 
from UNCLE. It reintroduces the outrageous villain from an earlier film who dons a stylish monk's habit and 
breaks the necks of victims with the curl of a deadly whip. Set at a posh girls' school filled with lecherous 
middle-aged professors, and with the cops fondling their hot-to-trot secretaries at every opportunity, it 
certainly is a throwback to those wonderfully politically-incorrect times. There's a definite link to a later 
Wallace-based film, the excellent giallo ""Whatever Happened to Solange?"", which also concerns female students 
being corrupted by (and corrupting?) their elders. Quite appropriate to the monk theme, the master-mind villain 
uses booby-trapped bibles here to deal some of the death blows, and also maintains a reptile-replete dungeon 
to amuse his captive audiences. <br /><br />Alfred Vohrer was always the most playful and visually flamboyant 
of the series directors, and here the lurid colour cinematography is the real star of the show. The Monk appears 
in a raving scarlet cowl and robe, tastefully setting off the lustrous white whip, while appearing against 
purplish-night backgrounds. There's also a voyeur-friendly turquoise swimming pool which looks great both 
as a glowing milieu for the nubile students and as a shadowy backdrop for one of the murder scenes. 
The trademark ""kicker"" of hiding the ""Ende"" card somewhere in the set of the last scene is also quite 
memorable here. And there's a fine brassy and twangy score for retro-music fans.<br /><br />Fans of the series 
will definitely miss the flippant Eddie Arent character in these later films. Instead, the chief inspector 
Sir John takes on the role of buffoon, convinced that he has mastered criminal psychology after taking a few 
night courses. Unfortunately, Klaus Kinski had also gone on to bigger and better things. The krimis had 
lost some of their offbeat subversive charm by this point, and now worked on a much more blatant pop-culture 
level, which will make this one quite accessible to uninitiated viewers."""

# Fluxo de transformação dos dados
tarefa1 = limpa_dados(texto_aval)
tarefa2 = limpa_caracter_especial(tarefa1)
tarefa3 = converte_minusculo(tarefa2)
tarefa4 = remove_stopwords(tarefa3)
tarefa5 = stemmer(tarefa4)

# Convertendo a string para um array Numpy (pois foi assim que o modelo foi treinado)
tarefa5_array = np.array(tarefa5)

# Aplicamos o vetorizador com mais uma conversão para array NumPy a fim de ajustar o shape de 0-d para 1-d
aval_final = vetorizador.transform(np.array([tarefa5_array])).toarray()

# Previsão com o modelo
previsao = modelo_v3.predict(aval_final.reshape(1,1000))

# Estrutura condicional para verificar o valor de previsao
if previsao == 1:
    print("O Texto Indica Sentimento Positivo!")
else:
    print("O Texto Indica Sentimento Negativo!")