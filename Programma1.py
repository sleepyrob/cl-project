import sys
import nltk

#Programma 1 (Roberto Cannarella, matr. 616400).
# Compiti del programma: confrontare sulla base di:

#1 - il N di frasi; il N di token
#2 - la L media delle frasi in token; delle parole in termini di char
#3 - la grandezza di V; la ricchezza di V, come TTR, nei primi 5000 token
#4 - distribuzione classi di frequenza |V1|, |V5|, |V10| all'aumentare del corpus, ogni 500 token
#5 - media di Sostantivi e Verbi per frase
#6 - densità lessicale: rapporto fra N totale di occorrenze nel testo di Sost, Verb, Agg, Avv e il numero totale di parole nel testo (escludendo POS = "." ",")

def CalcolaTokensEFrasi(raw1, raw2):
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    #Dal sentence_tokenizer vengono generate due FrasiTesto che sono liste (delle frasi)
    frasiTesto1 = sentence_tokenizer.tokenize(raw1)
    frasiTesto2 = sentence_tokenizer.tokenize(raw2)
    
    tokensTOTTesto1  = []
    for frase in frasiTesto1:
        tokens = nltk.word_tokenize(frase)
        tokensTOTTesto1 = tokensTOTTesto1 + tokens

    tokensTOTTesto2 = []
    for frase in frasiTesto2:
        tokens = nltk.word_tokenize(frase)
        tokensTOTTesto2 = tokensTOTTesto2 + tokens

    return frasiTesto1, frasiTesto2, tokensTOTTesto1, tokensTOTTesto2

def CalcolaLunghezzeMedie(frasiTesto1, frasiTesto2):
    # Calcolo il numero di frasi attraverso un contatore e il totale della loro lunghezza, poi faccio una media:
    
    LunghezzaTOTTesto1 = 0
    ContatoreTesto1 = 0
    LunghezzaTokenTOTTesto1 = 0
    ContatoreToken1 = 0
    
    for frase in frasiTesto1:
        tokens = nltk.word_tokenize(frase)
        LunghezzaTOTTesto1 = LunghezzaTOTTesto1 + len(tokens)
        ContatoreTesto1 = ContatoreTesto1 + 1
        for tok in tokens:
            LunghezzaTokenTOTTesto1 = LunghezzaTokenTOTTesto1 + len(tok)
            ContatoreToken1 = ContatoreToken1 + 1
    LunghezzaMediaTesto1 = LunghezzaTOTTesto1 / ContatoreTesto1
    LunghezzaMediaTokenTesto1 = LunghezzaTokenTOTTesto1 / ContatoreToken1

    # Uguale per le frasi del 2° testo:
    LunghezzaTOTTesto2 = 0
    ContatoreTesto2 = 0
    LunghezzaTokenTOTTesto2 = 0
    ContatoreToken2 = 0
    
    for frase in frasiTesto2:
        tokens = nltk.word_tokenize(frase)
        LunghezzaTOTTesto2 = LunghezzaTOTTesto2 + len(tokens)
        ContatoreTesto2 = ContatoreTesto2 + 1
        for tok in tokens:
            LunghezzaTokenTOTTesto2 = LunghezzaTokenTOTTesto2 + len(tok)
            ContatoreToken2 = ContatoreToken2 + 1
    LunghezzaMediaTesto2 = LunghezzaTOTTesto2 / ContatoreTesto2
    LunghezzaMediaTokenTesto2 = LunghezzaTokenTOTTesto2 / ContatoreToken2

    return LunghezzaMediaTesto1, LunghezzaMediaTesto2, LunghezzaMediaTokenTesto1, LunghezzaMediaTokenTesto2


def CreaVocabolario(tokensTesto):
    NuovaListaTok = []
    for tok in tokensTesto:
        if len(NuovaListaTok) < 5000:
            NuovaListaTok.append(tok)
    vocabolario = list(sorted(set(NuovaListaTok)))
    TTR = len(vocabolario) / len(NuovaListaTok)
    return vocabolario, TTR


def CalcolaDistribuzioniClassi(tokensTesto1, tokensTesto2):
    #Definizione degli intervalli su cui ciclare per il calcolo
    listaIntervalli = []
    i = 500
    #Viene definito un numero di intervalli di 500 sulla base della lunghezza di entrambi i testi
    while (i < len(tokensTesto1) and i < len(tokensTesto2)):
           listaIntervalli.append(i)
           i = i + 500

    #Nel codice che segue, ad ogni intervallo si individua la frequenza della parola ("for elem in vocabolario" in poi)
    #e, in base ad essa, si aggiunge (o meno) la parola alla classe di frequenza appropriata
    for intervallo in listaIntervalli:
        #Liste delle classi (ri)definite per ciascun intervallo in lista
        classeV1Testo1 = []
        classeV5Testo1 = []
        classeV10Testo1  = []
        
        listaTokenInt1 = tokensTesto1[:intervallo]
        vocabolario1 = set(listaTokenInt1)

        for elem in vocabolario1:
            freq = listaTokenInt1.count(elem)
            if freq == 1:
                classeV1Testo1.append(elem)
            elif freq == 5:
                classeV5Testo1.append(elem)
            elif freq == 10:
                classeV10Testo1.append(elem)

        classeV1Testo2 = []
        classeV5Testo2 = []
        classeV10Testo2  = []
        
        listaTokenInt2 = tokensTesto2[:intervallo]
        vocabolario2 = set(listaTokenInt2)

        for elem in vocabolario2:
            freq = listaTokenInt2.count(elem)
            if freq == 1:
                classeV1Testo2.append(elem)
            elif freq == 5:
                classeV5Testo2.append(elem)
            elif freq == 10:
                classeV10Testo2.append(elem)
       

        #In questo caso il confronto avviene direttamente nella funzione, perché avviene ciclicamente per ogni intervallo (ciclo for definito su)
        print("Intervallo:", intervallo)
        print("[Testo 1] Classe di frequenza |V1|:", len(classeV1Testo1), "- |V5|:", len(classeV5Testo1), "- |V10|:", len(classeV10Testo1))
        print("[Testo 2] CLasse di frequenza |V1|:", len(classeV1Testo2), "- |V5|:", len(classeV5Testo2), "- |V10|:", len(classeV10Testo2))
        print("\tIn questo intervallo:")
        if len(classeV1Testo1) > len(classeV1Testo2):
            print("\tClasse di frequenza V1: Testo 1 > Testo 2")
        elif len(classeV1Testo2) > len(classeV1Testo1):
            print("\tClasse di frequenza V1: Testo 2 > Testo 1")
        elif len(classeV1Testo1) == len(classeV1Testo2):
            print("\tClasse di frequenza V1: Testo 1 = Testo 2")
			
        if len(classeV5Testo1) > len(classeV5Testo2):
            print("\tClasse di frequenza V5: Testo 1 > Testo 2")
        elif len(classeV5Testo2) > len(classeV5Testo1):
            print("\tClasse di frequenza V5: Testo 2 > Testo 1")
        elif len(classeV5Testo1) == len(classeV5Testo2):
            print("\tClasse di frequenza V5: Testo 1 = Testo 2")
			
        if len(classeV10Testo1) > len(classeV10Testo2):
            print("\tClasse di frequenza V10: Testo 1 > Testo 2")
        elif len(classeV10Testo2) > len(classeV10Testo1):
            print("\tClasse di frequenza V10: Testo 2 > Testo 1")
        elif len(classeV10Testo1) == len(classeV10Testo2):
            print("\tClasse di frequenza V10: Testo 1 = Testo 2")
        print()


def CalcolaMediaSostantiviVerbi(frasi):
    mediaSTotale = 0
    mediaVTotale = 0
    mediaSVTotale = 0
    
    for frase in frasi:
        sostantivi = []
        verbi = []
        tokensFrase = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokensFrase)
        for tokPOS in tokensPOS:
            #Per evitare il tagger consideri erroneamente anche la punteggiatura (come faceva).
            #Trattandosi di nomi e verbi escludere i token di lunghezza 1 non dovrebbe creare problemi
            if len(tokPOS[0]) > 1:
                if tokPOS[1] == "NN" or tokPOS[1] == "NNS" or tokPOS[1] == "NNP" or tokPOS[1] == "NNPS":
                    sostantivi.append(tokPOS)
                elif tokPOS[1] == "VB" or tokPOS[1] == "VBD" or tokPOS[1] == "VBG" or tokPOS[1] == "VBN" or tokPOS[1] == "VBP" or tokPOS[1] == "VBZ":
                    verbi.append(tokPOS)
        numeroS = len(sostantivi)
        numeroV = len(verbi)
        numeroSV = numeroS + numeroV

        #Per ogni frase si calcolano le varie medie e si aggiorna il valore della media totale
        mediaSFrase = numeroS/len(tokensFrase)
        mediaSTotale = mediaSTotale + mediaSFrase
        
        mediaVFrase = numeroV/len(tokensFrase)
        mediaVTotale = mediaVTotale + mediaVFrase
        
        mediaSVFrase = numeroSV/len(tokensFrase)
        mediaSVTotale = mediaSVTotale + mediaSVFrase
    #Le medie finali relative all'intero testo
    mediaSTesto = mediaSTotale/len(frasi)
    mediaVTesto = mediaVTotale/len(frasi)
    mediaSVTesto = mediaSVTotale/len(frasi)
    return mediaSTesto, mediaVTesto, mediaSVTesto


def CalcolaDensitaLessicale(tokens):
    tokensPOS = nltk.pos_tag(tokens)
    parolePiene = []
    #Creazione di una lista in cui vengono poi inserite le parole "piene", ovvero appartenenti alle PoS indicate nella  consegna
    categoriePiene = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"]
    for tokPOS in tokensPOS:
        if tokPOS[1] in categoriePiene:
            parolePiene.append(tokPOS)
    
    for tok in tokensPOS:
        #Esclusione dei segni di punteggiatura per il calcolo della densità lessicale
        if tok[1] == "." or tok[1] == ",": 
            tokensPOS.remove(tok)
    numeroParolePiene = len(parolePiene)
    numeroParoleVuote = len(tokensPOS)
    DensitaLessicale = numeroParolePiene / numeroParoleVuote
    return DensitaLessicale



def main(file1, file2):
    #Presentazione task programma
    print("Questo programma svolge i seguenti compiti:")
    print("\t[1.1] Confronto dei due testi in base al numero di frasi")
    print("\t[1.2] Confronto dei due testi in base al numero di token")
    print("\t[2.1] Confronto dei due testi in base alla lunghezza media delle frasi (espressa in numero di token)")
    print("\t[2.2] Confronto dei due testi in base alla lunghezza media dei token (espressa in numero di caratteri)")
    print("\t[3.1] Confronto dei due testi in base alla grandezza  del vocabolario (N di parole tipo)")
    print("\t[3.2] Confronto dei due testi in base alla ricchezza lessicale (calcolata come Type-Token Ratio)")
    print("\t[4.1] Confronto della distribuzione delle classi di frequenza calcolate all'aumentare del testo di 500 token")
    print("\t[5.1] Confronto della media dei sostantivi per frase")
    print("\t[5.2] Confronto della media dei verbi per frase")
    print("\t[5.3] Confronto della media dei sostantivi E dei verbi per frase")
    print("\t[6.1] Confronto della densità lessicale")
    print()
    print("# ------------ #")
    print()
    
    #Apertura in input dei due file txt
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")
    
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()

    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    # Un'unica funzione per i primi calcoli - i valori (array) sono restituiti come
    # variabili del main, così da potere essere riutilizzati in seguito
    frasiTesto1, frasiTesto2, tokensTesto1, tokensTesto2 = CalcolaTokensEFrasi(raw1, raw2)

    #I primi confronti (frasi e token) invece avvengono  nel main:    
    #1.1 Confronto del numero di frasi:

    print("[1.1] Confronto dei due testi in base al numero di frasi:")
    if len(frasiTesto1) > len(frasiTesto2):
        print("\tIl numero di frasi del primo testo, cioè", file1+", è superiore a quello del secondo testo, cioè", file2+".")
        print("\t"+file1, "ha infatti", len(frasiTesto1),"frasi, mentre", file2, "ne ha", str(len(frasiTesto2))+".")
    elif len(frasiTesto2) > len(frasiTesto1):
        print("\tIl numero di frasi del secondo testo, cioè", file2+", è superiore a quello del primo testo, cioè", file1+".")
        print("\t"+file2, "ha infatti", len(frasiTesto2),"frasi, mentre", file1, "ne ha", str(len(frasiTesto1))+".")
    print()
    
    #1.2 Confronto del numero di token:

    print("[1.2] Confronto dei due testi in base al numero di token:")
    if len(tokensTesto1) > len(tokensTesto2):
        print("\tIl numero di token del primo testo, cioè", file1+", è superiore a quello del secondo testo, cioè", file2+".")
        print("\t"+file1, "ha infatti", len(tokensTesto1),"token, mentre", file2, "ne ha", str(len(tokensTesto2))+".")
    elif len(tokensTesto2) > len(tokensTesto1):
        print("\tIl numero di token  del secondo testo, cioè", file2+", è superiore a quello del primo testo, cioè", file1+".")
        print("\t"+file2, "ha infatti", len(tokensTesto2),"token, mentre", file1, "ne ha", str(len(tokensTesto1))+".")
    print()
    print("# ------------ #")
    print()


    #2.1 Lunghezza media delle frasi in token:
    print("[2.1] Confronto dei due testi in base alla lunghezza media delle frasi (espressa in numero di token):")

    #Valori calcolati nella funzione; confronti direttamente nel main
    LunghezzaMediaFrasiTesto1, LunghezzaMediaFrasiTesto2, LunghezzaMediaTokenTesto1, LunghezzaMediaTokenTesto2 = CalcolaLunghezzeMedie(frasiTesto1, frasiTesto2)
  
    if LunghezzaMediaFrasiTesto1 > LunghezzaMediaFrasiTesto2:
        print("\tLa lunghezza media delle frasi del primo testo, cioè ", file1+", è di", str(LunghezzaMediaFrasiTesto1)+ ", mentre quella del file", file2, "è di", str(LunghezzaMediaFrasiTesto2)+". Il primo testo ha quindi una lunghezza media delle frasi superiore.")
    elif LunghezzaMediaFrasiTesto2 > LunghezzaMediaFrasiTesto1:
        print("\tLa lunghezza media delle frasi del secondo  testo, cioè ", file2+", è di", str(LunghezzaMediaFrasiTesto2)+", mentre quella del file", file1, "è di", str(LunghezzaMediaFrasiTesto1)+". Il secondo testo ha quindi una lunghezza media delle frasi superiore.")
    print()

    #2.2 Lunghezza media dei token in caratteri:
    print("[2.2] Confronto dei due testi in base alla lunghezza media dei token (espressa in numero di caratteri):")

    if LunghezzaMediaTokenTesto1 > LunghezzaMediaTokenTesto2:
        print("\tLa lunghezza media dei token del primo testo è di", str(LunghezzaMediaTokenTesto1)+ ", mentre quella del secondo è di", str(LunghezzaMediaTokenTesto2)+". Il primo testo ha quindi una lunghezza media dei token superiore.")
    elif LunghezzaMediaTokenTesto2 > LunghezzaMediaTokenTesto1:
        print("\tLa lunghezza media dei token del secondo testo è di", str(LunghezzaMediaTokenTesto2)+ ", mentre quella del primo è di", str(LunghezzaMediaTokenTesto1)+". Il secondo testo ha quindi una lunghezza media dei token superiore.")
    print()
    print("# ------------ #")
    print()


    #3.1 Grandezza del Vocabolario
    print("[3.1] Confronto dei due testi in base alla grandezza  del vocabolario (N di parole tipo):")
    vocabolarioTesto1, TTRTesto1 = CreaVocabolario(tokensTesto1)
    vocabolarioTesto2, TTRTesto2 = CreaVocabolario(tokensTesto2)
    GrandezzaVocabolarioTesto1 = len(vocabolarioTesto1)
    GrandezzaVocabolarioTesto2 = len(vocabolarioTesto2)

    #Anche in questo caso: funzione apposita per la definizione delle variabili, che sono poi confrontate nel main    
    if GrandezzaVocabolarioTesto1 > GrandezzaVocabolarioTesto2:
        print("\tIl vocabolario del primo testo,", file1+", è più grande di quello del secondo, cioè", file2+". Il vocabolario del primo testo è grande infatti", GrandezzaVocabolarioTesto1, "parole tipo, quello del secondo è grande", GrandezzaVocabolarioTesto2, "parole tipo.")
    if GrandezzaVocabolarioTesto2 > GrandezzaVocabolarioTesto1:
        print("\tIl vocabolario del secondo testo,", file2+", è più grande di quello del primo, cioè", file1+". Il vocabolario del secondo testo è grande infatti", GrandezzaVocabolarioTesto2, "parole tipo, quello del primo  è grande", GrandezzaVocabolarioTesto1, "parole tipo.")
    print()

    #3.2 Ricchezza del vocabolario calcolata come Type-Token Ratio
    print("[3.2] Confronto dei due testi in base alla ricchezza lessicale (calcolata come Type-Token Ratio):")
    if TTRTesto1  > TTRTesto2:
        print("\tIl primo testo è lessicalmente più ricco del secondo. La sua TTR è di", TTRTesto1, "mentre quella del secondo testo è di", str(TTRTesto2)+".")
    elif TTRTesto2  > TTRTesto1:
        print("\tIl secondo  testo è lessicalmente più ricco del primo. La sua TTR è di", TTRTesto2, "mentre quella del secondo testo è di", str(TTRTesto1)+".")
    print()
    print("# ------------ #")
    print()

    #4.1 Distribuzione delle classi di frequenza |V1|, |V5| e |V10| all'aumentare del testo, 500 parole per volta
    print("[4.1] Confronto della distribuzione delle classi di frequenza calcolate all'aumentare del testo di 500 token:")
    print()
    #Interamente gestito dalla funzione
    CalcolaDistribuzioniClassi(tokensTesto1, tokensTesto2)
    print()
    print("# ------------ #")
    print()


    #5.1 Media di sostantivi e verbi per frase
    #Per questo esercizio non ero sicuro se la consegna volesse dire "media dei sostantivi e media dei verbi" oppure "media di sostantivi+verbi": nel dubbio ho svolto tutti e due (/tre) i calcoli.
    mediaSostantivi1, mediaVerbi1, mediaSostantiviVerbi1 = CalcolaMediaSostantiviVerbi(frasiTesto1)
    mediaSostantivi2, mediaVerbi2, mediaSostantiviVerbi2 = CalcolaMediaSostantiviVerbi(frasiTesto2)

    #Le medie sono calcolate nella funzione e confrontate nel main
    print("[5.1] Confronto della media dei sostantivi per frase:")
    if mediaSostantivi1 > mediaSostantivi2:
        print("\tLa media dei sostantivi per frase nel primo testo,", str(file1)+", è:", str(mediaSostantivi1)+", ed è superiore a quella del secondo testo, cioè", str(file2)+", che è:", str(mediaSostantivi2)+".")
    elif mediaSostantivi2 > mediaSostantivi1:
        print("\tLa media dei sostantivi per frase nel secondo testo,", str(file2)+", è:", str(mediaSostantivi2)+", ed è superiore a quella del primo testo, cioè", str(file1)+", che è:", str(mediaSostantivi1)+".")
    print()
    
    print("[5.2] Confronto della media dei verbi per frase:")
    if mediaVerbi1 > mediaVerbi2:
        print("\tLa media dei verbi per frase nel primo testo,", str(file1)+", è:", str(mediaVerbi1)+", ed è superiore a quella del secondo testo, cioè", str(file2)+", che è:", str(mediaVerbi2)+".")
    elif mediaVerbi2 > mediaVerbi1:
        print("\tLa media dei verbi per frase nel secondo testo,", str(file2)+", è:", str(mediaVerbi2)+", ed è superiore a quella del primo testo, cioè", str(file1)+", che è:", str(mediaVerbi1)+".")
    print()
    
    print("[5.3] Confronto della media dei sostantivi E dei verbi per frase:")
    if mediaSostantiviVerbi1 > mediaSostantiviVerbi2:
        print("\tLa media dei verbi e dei sostantivi per frase nel primo testo,", str(file1)+", è:", str(mediaSostantiviVerbi1)+", ed è superiore a quella del secondo testo, cioè", str(file2)+", che è:", str(mediaSostantiviVerbi2)+".")
    elif mediaSostantiviVerbi2 > mediaSostantiviVerbi1:
        print("\tLa media dei verbi e dei sostantivi per frase nel secondo testo,", str(file2)+", è:", str(mediaSostantiviVerbi2)+", ed è superiore a quella del primo testo, cioè", str(file1)+", che è:", str(mediaSostantiviVerbi1)+".")
    print()
    print("# ------------ #")
    print()

    
    #6.1 Densità lessicale
    print("[6.1] Confronto della densità lessicale:")
    DensitaLessicaleTesto1 = CalcolaDensitaLessicale(tokensTesto1)
    DensitaLessicaleTesto2 = CalcolaDensitaLessicale(tokensTesto2)

    if DensitaLessicaleTesto1 > DensitaLessicaleTesto2:
        print("\tIl primo testo è lessicalmente più denso del secondo. Esso ha infatti una densità lessicale di", str(DensitaLessicaleTesto1)+", mentre il secondo testo la ha di", str(DensitaLessicaleTesto2)+".")
    elif DensitaLessicaleTesto2 > DensitaLessicaleTesto1:
        print("\tIl secondo testo è lessicalmente più denso del primo. Esso ha infatti una densità lessicale di", str(DensitaLessicaleTesto2)+", mentre il primo testo la ha di", str(DensitaLessicaleTesto1)+".")

    
main(sys.argv[1], sys.argv[2])
