import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

if __name__ == "__main__":
    activationFilesAalborg = []
    for i in range(584):
        activationFilesAalborg.append(open("Activations/Aalborg/"+str(i),"r"))
    activationAalborg = []
    for file in activationFilesAalborg:
        activationAalborg.append([float(i) for i in file.read().split("\n") if i!=""])  
    for file in activationFilesAalborg:
        file.close()

    activationETrack2 = []
    for i in range(8001):
        file = open("Activations/ETrack2/"+str(i),"r")
        activationETrack2.append([float(i) for i in file.read().split("\n") if i!=""])
        file.close()
    

    montee = range(0,35)+range(327,360)+range(553,583)
    ld = range(36,67)+range(73,105)+range(112,128)+range(137,148)+range(376,413) + [i+583 for i in range(0,55)+range(76,92)+range(106,110)+range(130,140)+range(150,160)+range(170,177)+range(183,192)+range(202,224)+range(330,356)+range(373,391)]
    vSD = range(67,73)+range(128,137)+range(148,153)+range(207,215)+range(317,327)+range(413,431)+range(499,513) + [i+583 for i in range(110,130)+range(140,150)+range(192,202)+range(278,288)+range(365,373)]
    vSG = range(105,112)+range(152,170)+range(360,376)+range(431,440) + [i+583 for i in range(55,76)+range(160,170)+range(224,234)+range(288,330)+range(356,365)]
    vLG = range(170,207)+range(484,499)+range(513,553) + [i+583 for i in range(92,106)+range(234,278)]
    vLD = range(215,317)+range(440,484) + [i+583 for i in range(177,183)]

    ld = range(0,55)+range(76,92)+range(106,110)+range(130,140)+range(150,160)+range(170,177)+range(183,192)+range(202,224)+range(330,356)+range(373,400)
    vSD = range(67,73)+range(128,137)+range(148,153)+range(207,215)+range(317,327)+range(413,431)+range(499,513) + [i+583 for i in range(110,130)+range(140,150)+range(192,202)+range(278,288)+range(365,373)]
    vSG = range(55,76)+range(160,170)+range(224,234)+range(288,330)+range(356,365)
    vLG = range(92,106)+range(234,278)
    vLD = range(177,183)

    # t-SNE

    tsne = TSNE(n_components=2, perplexity = 20,n_iter = 5000, random_state=0)
    activations = activationAalborg+activationETrack2
    activations = activationETrack2
    activations2d = tsne.fit_transform(activations)
    fig,ax = plt.subplots()
    ax.scatter(activations2d[:, 0],activations2d[:, 1])

    a = True
    for i in range(len(activations)):
        if i in montee:
            if i == montee[0]:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['black'],label="montee")
                plt.pause(0.5)
            else:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['black'])
                plt.pause(0.5)
        if i%400 in ld:
            if i == ld[0]:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['red'],label="ligne droite")
                plt.pause(0.5)
            else:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['red'])
                plt.pause(0.5)
        elif i%400 in vSD:
            if i == vSD[0]:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['black'],label="virage serre a droite")
                plt.pause(0.5)
            else:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['sandybrown'])
                plt.pause(0.5)
        elif i%400 in vSG:
            if i == vSG[0]:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['gold'],label="virage serre a gauche")
                plt.pause(0.5)
            else:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['gold'])
                plt.pause(0.5)
        elif i%400 in vLD:
            if i == vLD[0]:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['green'],label="leger virage a droite")
                plt.pause(0.5)
            else:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['green'])
                plt.pause(0.5)
        elif i%400 in vLG:
            if i == vLG[0]:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['violet'],label="leger virage a gauche")
                plt.pause(0.5)
            else:
                ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = ['violet'])
                plt.pause(0.5)
        ax.scatter(activations2d[:,0][i],activations2d[:,1][i],color = 'blue')
        ax.annotate(str(i),(activations2d[:, 0][i],activations2d[:, 1][i]))
        plt.pause(0.5)

    ax.scatter([activations2d[:, 0][i] for i in montee]+[activations2d[:, 0][i] for i in ld]+[activations2d[:, 0][i] for i in vSD]+[activations2d[:, 0][i] for i in vSG]+[activations2d[:, 0][i] for i in vLD]+[activations2d[:, 0][i] for i in vLG], [activations2d[:, 1][i] for i in montee]+[activations2d[:, 1][i] for i in ld]+[activations2d[:, 1][i] for i in vSD]+[activations2d[:, 1][i] for i in vSG]+[activations2d[:, 1][i] for i in vLD]+[activations2d[:, 1][i] for i in vLG],color = ['black']*len(montee)+['red']*len(ld)+['sandybrown']*len(vSD)+['gold']*len(vSG)+['green']*len(vLD)+['violet']*len(vLG))

    # k-means

    kmeans = KMeans(n_clusters=7)
    kmeans.fit(activations2d)
    y_kmeans = kmeans.predict(activations2d)
    #fig,ax = plt.subplots()
    cvals  = range(4)
    colors = ['black','blue','red','violet']

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    ax.scatter(activations2d[:, 0], activations2d[:, 1], c=y_kmeans, s=50, cmap=cmap)

    for i in range(len(activationAalborg)):
       ax.annotate("A"+str(i),(activations2d[:, 0][i],activations2d[:, 1][i]))
    for i in range(len(activationAalborg),len(activationETrack2)):
       ax.annotate("E"+str(i-len(activationAalborg)),(activations2d[:, 0][i],activations2d[:, 1][i]))

    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.legend(loc='best')
    plt.savefig('k-means_Aalborg_ETrack2.png')
    plt.show()