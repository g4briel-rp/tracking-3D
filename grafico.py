from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

if __name__  == '__main__':
    X = []
    Y = []
    Z = []
    arquivo = open("resultados.txt")
    linhas = arquivo.readlines()
    corte = int(len(linhas) / 4)
    contador = 1

    for linha in linhas:
        if contador % corte == 0 or contador == 1:
            valores = linha.split()
            X.append(float(valores[0]))
            Y.append(-float(valores[1]) + 212)
            Z.append(float(valores[2]))
        contador += 1
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot(X, Z, Y, 'purple') 
    # ax.set_xlim(0, 1920)
    # ax.set_zlim(0, 1080)
    ax.set_title('Ambiente de Plotagem 3D', fontsize=18)

    ax.scatter(float(X[0]), float(Z[0]), float(Y[0]))
    ax.scatter(float(X[len(X) - 1]), float(Z[len(Z) - 1]), float(Y[len(Y) - 1]))

    # ax.axes.set_aspect('equal')
    plt.show()