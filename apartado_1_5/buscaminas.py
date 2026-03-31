import pygame
import random
import sys

pygame.init()

# Configuración
MARGEN=30
TAM = 10
TILE = 40
ANCHO = TAM * TILE+MARGEN*2
ALTO = TAM * TILE+MARGEN*2

screen = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Buscaminas PRO")

# Colores
BLANCO = (255, 255, 255)
GRIS = (200, 200, 200)
NEGRO = (50, 50, 50)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)

font = pygame.font.SysFont(None, 30)

# Crear tablero con minas
tablero = [[random.randint(0, 1) for _ in range(TAM)] for _ in range(TAM)]
visible = [[False for _ in range(TAM)] for _ in range(TAM)]
banderas = [[False for _ in range(TAM)] for _ in range(TAM)]

vidas = 10

# 🔢 Contar minas alrededor
def contar_minas(fila, col):
    contador = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            ni = fila + i
            nj = col + j

            if 0 <= ni < TAM and 0 <= nj < TAM:
                if tablero[ni][nj] == 1:
                    contador += 1

    return 8-contador

# 🎨 Dibujar tablero
def dibujar():
    screen.fill(BLANCO)

    for i in range(TAM):
        for j in range(TAM):
            rect = pygame.Rect(MARGEN+j*TILE, MARGEN+i*TILE, TILE, TILE)

            if visible[i][j]:
                pygame.draw.rect(screen, GRIS, rect)

                if tablero[i][j] == 1:
                    pygame.draw.circle(screen, ROJO, rect.center, 10)
                else:
                    minas = contar_minas(i, j)
                    if minas > 0:
                        texto = font.render(str(minas), True, AZUL)
                        screen.blit(texto, (MARGEN+j * TILE + 12, MARGEN+i * TILE + 8))

            else:
                pygame.draw.rect(screen, NEGRO, rect)

                # 👉 Dibujar X si hay bandera
                if banderas[i][j]:
                    texto = font.render("X", True, ROJO)
                    screen.blit(texto, (MARGEN+j * TILE + 12, MARGEN+i * TILE + 8))
            pygame.draw.rect(screen, BLANCO, rect, 1)

    texto = font.render(f"Vidas: {vidas}", True, NEGRO)
    screen.blit(texto, (10, 10))

    pygame.display.flip()

# 🎮 Bucle principal
running = True
while running:
    dibujar()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col = (x - MARGEN) // TILE
            fila = (y - MARGEN) // TILE

                # 🖱️ CLICK IZQUIERDO
            if event.button == 1:
                if not visible[fila][col] and not banderas[fila][col]:
                    visible[fila][col] = True

                    if tablero[fila][col] == 1:
                        vidas -= 1
                        if vidas == 0:
                            print("GAME OVER")
                            pygame.quit()
                            sys.exit()

            # 🖱️ CLICK DERECHO
            elif event.button == 3:
                if not visible[fila][col]:
                    banderas[fila][col] = not banderas[fila][col]