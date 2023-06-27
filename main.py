import numpy as np
import cv2

# AJUSTES INICIAIS PARA O FUNCIONAMENTO DO CODIGO.
# -> AJUSTAR A DISTANCIA INICIAL.
# -> AJUSTAR A LARGURA DO OBJETO.

# função para calculo da distancia focal
def calcula_distanciaFocal(largura_em_pixels, distancia_conhecida, largura_conhecida):
    distanciaFocal = (largura_em_pixels * distancia_conhecida) / largura_conhecida
    return distanciaFocal

if __name__ == '__main__':
    conta_frames = -1

    # informações inicias
    larguraConhecida = 0.1
    distancia_conhecida = 0.35

    # variaveis para calculo da trajetoria
    pontos_X = []
    pontos_Y = []
    distancia = 0
    distanciaFocal = 0
    pontos_Dist = []
    guarda_X = []
    guarda_Y = []
    guarda_Dist = []
    diferenca_X = 0.0
    diferenca_Y = 0.0
    string_X = ''
    string_Y = ''
    mov_XY = ''
    string_Dist = ''

    # intervalo de cor
    lower_red = np.array([170, 130, 60])
    upper_red = np.array([180, 255, 125])

    kernel = np.ones((7,7), np.uint8)

    # captura de arquivo de video, necessario redimensionar o video
    video = cv2.VideoCapture('./videos/video3.mp4')

    # começa a obtenção dos frames do video e sua exibição
    while(True):
        _, frame = video.read()

        # caso use webcam do celular, necessario rotacionar ela em 90 graus
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame is None:
            break

        # frame = cv2.resize(frame, (480, 640), interpolation=cv2.INTER_AREA)

        # criação da segunda janela para mostrar a direção do movimento em x e y
        second_frame = np.zeros((int(frame.shape[0]), int(frame.shape[1]), 3), dtype=np.uint8)

        # criação da terceira janela para mostrar a previsão do chute na linha do gol [dist = 0.0 m]
        third_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # converte a imagem para HSV, encontrar as cores dentro do intervalo definido e encontra os contornos a partir da mascara obtida
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blur_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
        mask = cv2.inRange(blur_frame, lower_red, upper_red)
        mask = cv2.dilate(mask, kernel, iterations=4)
        final_frame = cv2.bitwise_and(frame, frame, mask=mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # caso encontre contornos, calcula o ID do contorno de maior area
        if cnts:
            i = 0
            maxArea = cv2.contourArea(cnts[0])
            idContourMaxArea = 0

            for c in cnts:
                if maxArea < cv2.contourArea(c):
                    maxArea = cv2.contourArea(c)
                    idContourMaxArea = i
                i += 1

            # obtem as informações do objeto como posição X, Y, altura e largura
            x, y, w, h = cv2.boundingRect(cnts[idContourMaxArea])

            # calculo da distancia focal usada para determinar a distancia da bola ate a camera
            if conta_frames == 0:
                distanciaFocal = calcula_distanciaFocal(w, distancia_conhecida, larguraConhecida)
                distancia = (larguraConhecida * distanciaFocal) / w
            
            if distanciaFocal:
                distancia = (larguraConhecida * distanciaFocal) / w

            # a cada 5 frames é armazenado junto dos valores da posição x e y para determinar a direção do movimento
            if conta_frames != 0 and conta_frames % 5 == 0:
                pontos_X.append(x)
                pontos_Y.append(y)
                pontos_Dist.append(distancia)

            # movimento de aproximação/afastamento da camera
            if len(pontos_Dist) > 2:
                guarda_Dist.append(pontos_Dist[0])
                pontos_Dist.pop(0)

                if pontos_Dist[1] > pontos_Dist[0]:
                    string_Dist = 'Frente'
                elif pontos_Dist[0] > pontos_Dist[1]:
                    string_Dist = 'Atras'
                else:
                    string_Dist = 'Parado'

            # movimento horizontal
            if len(pontos_X) > 2:
                guarda_X.append(pontos_X[0])
                pontos_X.pop(0)
                diferenca_X = pontos_X[1] - pontos_X[0]

                if diferenca_X > 20:
                    string_X = 'Direita'
                elif diferenca_X < -20:
                    string_X = 'Esquerda'
                else:
                    string_X = 'Parado'

            # movimento vertical
            if len(pontos_Y) > 2:
                guarda_Y.append(pontos_Y[0])
                pontos_Y.pop(0)
                diferenca_Y = pontos_Y[1] - pontos_Y[0]

                if diferenca_Y > 20:
                    string_Y = 'Baixo'
                elif diferenca_Y < -20:
                    string_Y = 'Cima'
                else:
                    string_Y = 'Parado'

            # definição do direção no plano X_Y
            if   string_X == 'Direita'  and string_Y == 'Parado': mov_XY = 'Leste'
            elif string_X == 'Direita'  and string_Y == 'Cima':   mov_XY = 'Nordeste'
            elif string_X == 'Parado'   and string_Y == 'Cima':   mov_XY = 'Norte'
            elif string_X == 'Esquerda' and string_Y == 'Cima':   mov_XY = 'Noroeste'
            elif string_X == 'Esquerda' and string_Y == 'Parado': mov_XY = 'Oeste'
            elif string_X == 'Esquerda' and string_Y == 'Baixo':  mov_XY = 'Sudoeste'
            elif string_X == 'Parado'   and string_Y == 'Baixo':  mov_XY = 'Sul'
            elif string_X == 'Direita'  and string_Y == 'Baixo':  mov_XY = 'Sudeste'
            else: mov_XY = 'Parado'

            # construção dos pontos para monitoramento da posição no third_frame
            if len(pontos_X) >= 2 and len(pontos_Y) >= 2: 
                cv2.circle(third_frame, (pontos_X[1] + int(w / 2), pontos_Y[1] + int(h / 2)), 3, (0, 0, 255), -1)
                cv2.circle(third_frame, (pontos_X[0] + int(w / 2), pontos_Y[0] + int(h / 2)), 3, (0, 255, 0), -1)

            # construção dos elementos adicionais de texto do frame
            cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 0, 0), 2)
            cv2.putText(frame, ("%d" % w), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(frame, ("%d" % h), (x + w + 5, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

        # construção dos elementos adicionais de texto do second_frame
        cv2.putText(second_frame, ("frame: %d" % conta_frames), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
        if distancia:
            cv2.putText(frame, ("distancia para a camera: "), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
            cv2.putText(frame, ("%.3f" % distancia + " m"), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
        cv2.putText(second_frame, ("movimento no plano X e Y: " + mov_XY), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
        cv2.putText(second_frame, ("movimento no plano X e Distancia: " + string_Dist), (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

        # construção dos elementos adicionais de texto do third_frame
        cv2.putText(third_frame, ("verde = ponto anterior"), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
        cv2.putText(third_frame, ("vermelho = ponto atual"), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

        cv2.imshow("original", frame)
        # cv2.imshow("informacoes", second_frame)
        # cv2.imshow("monitoramento", third_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        conta_frames += 1

    video.release()
    cv2.destroyAllWindows
    
    with open('resultados.txt', 'w') as arquivo:
        for valor,valor2,valor3 in zip(guarda_X, guarda_Y, guarda_Dist):
            arquivo.write(str(valor) + '\t' + str(valor2) + '\t' + str(valor3) + '\n')