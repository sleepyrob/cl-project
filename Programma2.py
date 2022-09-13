import sys
import nltk
from nltk import bigrams
from nltk import trigrams
import math

#Programma 2 (Roberto Cannarella, matr. 616400).
# Compiti del programma: estrarre le seguenti informazioni per ciascuno dei due corpora:

#1 - 1# Estrarre e ordinare in ordine di frequenza decrescente, indicando anche la relativa frequenza:
# - - le 10 PoS più frequenti
# - - i 20 S e i 20 V più frequenti
# - - i 20 bigrammi Sost, V più frequenti
# - - i 20 bigrammi Agg, Sost più frequenti

# - 2# Estrarre e ordinare i 20 bigrammi di token (ogni token freq. > 3):
# - - con P congiunta massima, indicando anche la relativa probabilità
# - - con P condizionata massima, indicando anche la relativa P
# - - con forza associativa (LMI) massima, indicando anche la relativa forza assoiativa

# - 3# Per ogni lunghezza di frase da 8 a 15 token, estrarre la frase con P più alta, dove la P deve essere calcolata attraverso un modello di Markov di ordine 1 ocn Add-one smoothing. Il modello deve usare le statistiche estratte dla corpus che contiene le frasi.

# - 4# Dopo avere individuato e classificato le NE, estrarre:
# - - i 15 nomi propri di perosna più frequenti (tipi), ordinati per frequenza
# - - i 15 nomi propri di luogo più frequenti (tipi), ordinati per frequenza

def CalcoliPoS(frasi):
    tokensTesto = []
    listaTokenPOS = []
    #Definizione e poi riempimento delle liste PoS per ogni frase
    listaPOS = []
    listaV = []
    listaS = []
    #Queste liste tornano utili più volte per controllare la PoS di un token taggato (usando il costrutto if ... in lista)
    PoSSostantivi = ["NN", "NNP", "NNS", "NNPS"]
    PoSVerbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    PoSAggettivi = ["JJ", "JJR", "JJS"]
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensTesto = tokensTesto + tokens
        for tok in tokensPOS:
            listaTokenPOS.append(tok)
            listaPOS.append(tok[1])
            if tok[1] in PoSSostantivi:
                listaS.append(tok)
            if tok[1] in PoSVerbi:
                listaV.append(tok)

    #Rimuovo segni di punteggiatura che NLTK categorizza come V
    for elem in listaV:
        if elem[0][0] == "’" or elem[0][0] == "’" or elem[0][0] == "’":
            listaV.remove(elem)
    #Rimuovo segni di punteggiatura e rumore vario che NLTK categorizza come N
    for elem in listaS:
        if elem[0][0] == "’" or elem[0][0] == "’" or elem[0][0] == "’" or elem[0][0] == "m" or elem[0][0] == "t":
            listaS.remove(elem)
    for elem in listaTokenPOS:
        if elem[0][0] == "’" or elem[0][0] == "’" or elem[0][0] == "’" or elem[0][0] == "m" or elem[0][0] == "t":
            listaTokenPOS.remove(elem)

    #Ciascuna "porzione" di codice si occupa di individuare con nltk.FreqDist i relativi elementi più frequenti
    DistrFreq1 = nltk.FreqDist(listaPOS)
    lista10POS = DistrFreq1.most_common(10)
    print("[1.1] Le 10 Part of Speech più frequenti sono (fra parentesi, la relativa frequenza):")
    for elem in lista10POS:
        print(elem[0], "\t\t("+str(elem[1])+")")
    print()
    
    DistrFreq2 = nltk.FreqDist(listaS)
    lista20S = DistrFreq2.most_common(20)
    print("[1.2.1] I 20 sostantivi più frequenti sono (fra parentesi, la relativa frequenza):")
    for elem in lista20S:
        print(elem[0][0], "\t\t("+str(elem[1])+")")
    print()
    
    
    DistrFreq3 = nltk.FreqDist(listaV)
    lista20V = DistrFreq3.most_common(20)
    print("[1.2.2] I 20 verbi  più frequenti sono (fra parentesi, la relativa frequenza):")
    for elem in lista20V:
        print(elem[0][0], "\t\t("+str(elem[1])+")")
    print()

    print("[1.3] I 20 bigrammi Sostantivo, Verbo  più frequenti sono (fra parentesi, la relativa frequenza):")
    bigrammiPOS = list(bigrams(listaTokenPOS))
    listaSostV = []
    for bigramma in bigrammiPOS:
        #individuazione dei bigrammi Sostantivo-Verbo
        if bigramma[0][1] in PoSSostantivi and bigramma[1][1] in PoSVerbi:
            listaSostV.append(bigramma)
    #Pulizia per evitare output indesiderati
    for elem in listaSostV:
        if elem[0][0][0] == "’" or elem[0][0][0] == "’" or elem[0][0][0] == "’" or elem[0][0][0] == "m" or elem[0][0][0] == "t" or elem[0][0][0] == "ll":
            listaSostV.remove(elem)
        
    DistrFreq4 = nltk.FreqDist(listaSostV)
    lista20SV = DistrFreq4.most_common(20)
    for elem in lista20SV:
        print(elem[0][0][0], elem[0][1][0], "\t\t("+str(elem[1])+")")
    print()

    print("[1.4] I 20 bigrammi Aggettivo, Sostantivo più frequenti sono (fra parentesi, la relativa frequenza):")
    bigrammiPOS = list(bigrams(listaTokenPOS))
    listaAggS = []
    for bigramma in bigrammiPOS:
        #individuazione dei bigrammi Agg-Sost
        if bigramma[0][1] in PoSAggettivi  and bigramma[1][1] in PoSSostantivi:
            listaAggS.append(bigramma)
    for elem in listaAggS:
        if elem[0][0][0] == "’" or elem[0][0][0] == "’" or elem[0][0][0] == "’" or elem[0][0][0] == "m" or elem[0][0][0] == "t" or elem[0][0][0] == "ll" or elem[0][0][0] == "re":
            listaAggS.remove(elem)
            
    DistrFreq5 = nltk.FreqDist(listaAggS)
    lista20AS = DistrFreq5.most_common(20)
    for elem in lista20AS:
        print(elem[0][0][0], elem[0][1][0], "\t\t("+str(elem[1])+")")
    print()


def CalcoliProbabilita(frasi):
    punteggiatura = [",", ".", ":", "’", "(", ")", "[", "]"]
    tokensTOT = []
    tokensTOTCorpus = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensTOT = tokensTOT + tokens
    for tok in tokensTOT:
        tokensTOTCorpus.append(tok)
    for tok in tokensTOT:
        freqTok = tokensTOT.count(tok)
        if freqTok < 3:
            tokensTOT.remove(tok)
    #TokensTOT contiene adesso solo i tokens con frequenza di almeno 3, tokensTOTCorpus invece tutti quelli del testo

    bigrammi = list(bigrams(tokensTOT))

    #Probabilità congiunta, calcolata come rapporto fra la frequenza del bigramma e la lunghezza del corpus
    print("[2.1] I 20 bigrammi con Probabilità congiunta superiore sono (probabilità fra parentesi):")
    DizionarioPC = {}
    for bigramma in bigrammi:
        if bigramma[0] in punteggiatura or bigramma[1] in punteggiatura:
            bigrammi.remove(bigramma)
        else:
            freqBig = bigrammi.count(bigramma)
            PCongiunta = freqBig / len(tokensTOTCorpus)
            DizionarioPC[bigramma] = PCongiunta
    DizionarioOrdinato = sorted(DizionarioPC.items(), key=lambda x: x[1], reverse = True)
    for elem in DizionarioOrdinato[:20]:
        print("Bigramma:", elem[0][0], elem[0][1], "("+str(elem[1])+")")
    print()

    #Probabilità condizionata, calcolata come rapporto fra la probabilità congiunta e la frequenza del primo token
    #Local Mutual Information
    print("[2.2] I 20 bigrammi con Probabilità condizionata superiore sono (probabilità fra parentesi):")
    DizionarioPCond = {}
    DizionarioLMI = {}
    for bigramma in bigrammi:
        if tokensTOTCorpus.count(bigramma[0]) < 3 or tokensTOTCorpus.count(bigramma[1]) < 3:
            bigrammi.remove(bigramma)
        else:
            freqPrimoTok = tokensTOTCorpus.count(bigramma[0])
            freqBig = bigrammi.count(bigramma)
            #freqBig = Frequenza osservata del bigramma (O)
            PCongiunta = freqBig / len(tokensTOTCorpus)
            PCondizionata = PCongiunta / freqPrimoTok
            DizionarioPCond[bigramma] = PCondizionata

            #Frequenza attesa (E) = probabilità del bigramma se le due parole fossero statisticamente indipendenti
            freqAttesa = (tokensTOTCorpus.count(bigramma[0]) * tokensTOTCorpus.count(bigramma[1])) / len(tokensTOTCorpus)
            LMI = math.log(freqBig * (freqBig/freqAttesa))
            DizionarioLMI[bigramma] = LMI
        
    DizionarioOrdinatoPCond = sorted(DizionarioPCond.items(), key=lambda x: x[1], reverse = True)
    #Versione ordinata  del dizionario contenente i bigrammi con Probabilità condizionata più alta
    for elem in DizionarioOrdinatoPCond[:20]:
        print("Bigramma:", elem[0][0], elem[0][1], "("+str(elem[1])+")")
    print()

    print("[2.3] I 20 bigrammi con Local Mutual Information superiore sono (forza associativa fra parentesi):")
    DizionarioOrdinatoLMI = sorted(DizionarioLMI.items(), key=lambda x: x[1], reverse = True)
    #Versione ordinata  del dizionario contenente i bigrammi con LMI più alta
    for elem in DizionarioOrdinatoLMI[:20]:
        print("Bigramma:", elem[0][0], elem[0][1], "("+str(elem[1])+")")
    print()


def CalcoliMarkov(frasi):
    
    print("[3.1]")

    tokensTOT = []
    frasi8 = []
    frasi9 = []
    frasi10 = []
    frasi11 = []
    frasi12 = []
    frasi13 = []
    frasi14 = []
    frasi15 = []
    #Divisione delle frasi in liste in base alla lunghezza
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensTOT = tokensTOT + tokens
        if len(tokens) == 8:
            frasi8.append(frase)
        elif len(tokens) == 9:
            frasi9.append(frase)
        elif len(tokens) == 10:
            frasi10.append(frase)
        elif len(tokens) == 11:
            frasi11.append(frase)
        elif len(tokens) == 12:
            frasi12.append(frase)
        elif len(tokens) == 13:
            frasi13.append(frase)
        elif len(tokens) == 14:
            frasi14.append(frase)
        elif len(tokens) == 15:
            frasi15.append(frase)

    vocabolario = set(tokensTOT)
    bigrammiTOT = list(bigrams(tokensTOT))
    print("La frase di lunghezza 8 (punteggiatura inclusa) con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):")
    #Per l'individuazione della frase si rimanda a una funzione apposita, CalcolaPMarkov (segue)
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi8)
    print()
    
    print("La frase di lunghezza 9 (punteggiatura inclusa)  con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi9)
    print()
    
    print("La frase di lunghezza 10 (punteggiatura inclusa) con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi10)
    print()
    
    print("La frase di lunghezza 11 (punteggiatura inclusa) con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi11)
    print()

    print("La frase di lunghezza 12 (punteggiatura inclusa) con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi12)
    print()
    
    print("La frase di lunghezza 13 (punteggiatura inclusa) con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi13)
    print()

    print("La frase di lunghezza 14 (punteggiatura inclusa) con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi14)
    print()
    
    print("La frase di lunghezza 15 (punteggiatura inclusa)  con probabilità maggiore è la seguente (fra parentesi la probabilità, calcolata attraverso un modello markoviano di ordine 1):") 
    CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi15)
    print()

def CalcolaPMarkov(vocabolario, bigrammiTOT, tokensTOT, frasi): 
    PMarkovMax = 0
    fraseMax = ""
    #valori place-holder
    
    for frase in frasi:
        tokensFrase = nltk.word_tokenize(frase)
        bigrammi = list(bigrams(tokensFrase))
        freqPrimaParola = tokensTOT.count(bigrammi[0][0])
        #Probabilità con catena di Markov di ordine 1: P = P(A) * P(B|A) * (C|B), ecc...
        #Quindi il valore può essere inizializzatoto come P(A), cioè con la frequenza della prima parola divisa per la cardinalità
        #del corpus + lo smoothing, che fa aumentare di 1 la frequenza e aggiunge la cardinalità del vocabolario al divisore
        PMarkov = freqPrimaParola + 1 / (len(tokensTOT) + len(vocabolario))
        for bigramma in bigrammi:
            ProbBigramma = bigrammiTOT.count(bigramma) / len(tokensTOT)
            ProbToken1 = tokensTOT.count(bigramma[0]) / len(tokensTOT)
            #Di ogni bigramma, ad es. P(B|A), si calcola la P condizionata, anche in questo caso includendo le variazioni dello smoothing
            PCondizionata = (ProbBigramma + 1) / (ProbToken1 + len(vocabolario))
            PMarkov = PMarkov * PCondizionata
        if PMarkov > PMarkovMax or PMarkovMax == 0:
            #se la P della frase è superiore rispetto a quella salvata
            #oppure se è a 0 (ovvero: nel caso della primissima frase); aggiornare PMarkovMax
            PMarkovMax = PMarkov
            fraseMax = frase
        
    print(fraseMax, "("+str(PMarkovMax)+")")
    print()
   

def NER(frasi):
    tokensTOT = []
    tokensPOStot = []
    NamedEntityList = []
    NomiPersona = []
    NomiLuogo = []
    for frase in frasi:
        tokensFrase = nltk.word_tokenize(frase)
        tokensTOT = tokensTOT + tokensFrase
        tokensPOS = nltk.pos_tag(tokensFrase)
        tokensPOStot = tokensPOStot + tokensPOS
        analisi = nltk.ne_chunk(tokensPOS)
        #Analisi corrisponde a un albero rappresentato con una notazione a parentesi
        #L'albero è diviso in sottoalberi e nodi: nel for che segue scorro i nodi
        for nodo in analisi:
            NE = ""
            if hasattr(nodo, "label"):
                if nodo.label() in ["PERSON"]:
                    for partNE in nodo.leaves():
                        NomiPersona.append(partNE[0])
                        #Controllo se la label del nodo è PERSON e in quel caso aggiungo l'entità (il nome proprio) alla lista NomiPersona
            if hasattr(nodo, "label"):
                if nodo.label() in ["GPE"]:
                    for partNE in nodo.leaves():
                        NomiLuogo.append(partNE[0])
                        ##Controllo se la label del nodo è GPE  e in quel caso aggiungo l'entità (il nome del luogo) alla lista NomiLuogo

    distribuzionePersona = nltk.FreqDist(NomiPersona)
    freqNomi = distribuzionePersona.most_common(15)
    #Uso due funzioni di NLTK prima per riordinare i nomi in ordine di frequenza, poi per individuare i 15 più comuni. Uguale dopo coi GPE
    print("[4.1] I 15 nomi propri di persona più frequenti sono (fra parentesi la relativa frequenza):")
    for elem in freqNomi:
        print(elem[0], "\t\t("+str(elem[1])+")")
    print()
    
    distribuzioneLuoghi = nltk.FreqDist(NomiLuogo)
    freqLuoghi = distribuzioneLuoghi.most_common(15)
    print("[4.2] I 15 nomi propri di luogo più frequenti sono (fra parentesi la relativa frequenza):")
    for elem in freqLuoghi:
        print(elem[0], "\t\t("+str(elem[1])+")")
        
    print()

def main(file1, file2):
    #Presentazione task programma
    print("Questo programma estrae per ciascuno dei due corpora le seguenti informazioni:")
    print("\t[1.1] Le 10 Part of Speech più frequenti")
    print("\t[1.2.1] I 20 sostantivi più frequenti")
    print("\t[1.2.2] I 20 verbi  più frequenti")
    print("\t[1.3] I 20 bigrammi Sostantivo, Verbo più frequenti")
    print("\t[1.4] I 20 bigrammi Aggettivo, Sostantivo più frequenti")
    print("\t[2.1] I 20 bigrammi con Probabilità congiunta superiore")
    print("\t[2.2] I 20 bigrammi con Probabilità condizionata  superiore")
    print("\t[2.3] I 20 bigrammi con Local Mutual Information superiore")
    print("\t[3.1] Per ogni lunghezza di frase da 8 a 15 token, la frase con Probabilità  più alta")
    print("\t[4.1] I 15 nomi propri di persona più frequenti")
    print("\t[4.2] I 15 nomi propri di luogo più frequenti")
    print()
    print("# ------------ #")
    print()
    #Apertura in input dei due file txt
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()

    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    frasiTesto1 = sentence_tokenizer.tokenize(raw1)
    frasiTesto2 = sentence_tokenizer.tokenize(raw2)
    #Non essendoci confronto, in questo caso le operazioni sono svolte interamente nelle funzioni stesse
    #Ogni funzione è invocata una volta sulle frasi del primo testo e una volta sulle frasi del secondo testo
    print("[1] Prima serie di calcoli sul primo testo,", file1)
    CalcoliPoS(frasiTesto1)
    print()
    print("[1] Prima serie di calcoli sul secondo testo,", file2)
    CalcoliPoS(frasiTesto2)
    print()
    print("# ------------ #")
    print()

    print("[2] Seconda serie di calcoli sul primo testo,", file1)
    CalcoliProbabilita(frasiTesto1)
    print()
    print("[2] Seconda serie di calcoli sul secondo testo,", file2)
    CalcoliProbabilita(frasiTesto2)
    print()
    print("# ------------ #")
    print()

    print("[3] Terza serie di calcoli sul primo testo,", file1)
    CalcoliMarkov(frasiTesto1)
    print("[3] Terza serie di calcoli sul secondo testo,", file2)
    CalcoliMarkov(frasiTesto2)
    print()
    print("# ------------ #")
    print()

    print("[4] Named Entity Recognition sul primo testo,", file1)
    NER(frasiTesto1)
    print("[4] Named Entity Recognition sul secondo testo,", file2)
    NER(frasiTesto2)
    print()


main(sys.argv[1], sys.argv[2])
