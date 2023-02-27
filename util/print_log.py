import csv
import matplotlib.pyplot as plt
def print_log(path, metric):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        metrics = []
        results = []
        for i, row in enumerate(reader):
            row = row[0].split(',')
            if(i==0):
                metrics = [j for j in row]
            else:
                scores = [j for j in row]
                results.append(scores)
        idx = metrics.index(metric)
        idx_epoch = metrics.index('epoch')
        #val = [float(i[idx]) for i in results]
        val = []
        for j in results:
            try:
                val.append(float(j[idx]))
            except:
                continue
        #epochs = [int(i[idx_epoch]) for i in results]
        plt.plot(val)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.show()