from functions import kmeans, findNumberOfClasses

def main():
    fileName = 'vectors.json'
    dim = 2
    numberOfClasses = 2
    drawPlot = True

    kmeans(fileName, dim, numberOfClasses, drawPlot)

    #nClasses = [3, 4, 5]
    #findNumberOfClasses(fileName, dim, nClasses)


if __name__ == "__main__":
    main()