import pygame
import pygame.camera
import tkinter
from tkinter import filedialog
import numpy
import cv2

WHITE = (255, 255, 255)
BLACK = (0,0,0)

def startGui(screen, clock):
    x1=50;y1=100;x2=50;y2=220
    w1=400;h1=150;w2=400;h2=150
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (x1, y1, 400, 200), 5)
    font = pygame.font.SysFont('Helvetica', 25, True, False)
    useCamera=font.render("Take a Picture", True, BLACK)

    takePictureWidth = useCamera.get_width()
    takePictureHeight = useCamera.get_height()

    screen.blit(useCamera, [x1+(400-takePictureWidth)/2, y1+(200-takePictureHeight)/2])
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return 2
        clock.tick(30)

def getWebcam(screen,clock,WIDTH,HEIGHT):
    camera = cv2.VideoCapture(0)
    done = False
    while not done:
        ret, frame = camera.read()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return 0;
            elif event.type == pygame.KEYDOWN:
                img_name = "user.jpg".format(0)
                cv2.imwrite(img_name, frame)
                return "user.jpg";
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = numpy.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (WIDTH,HEIGHT))
        screen.blit(frame, (0, 0))
        pygame.draw.rect(screen, WHITE, (0, 0, 120, 20), 0)
        font = pygame.font.SysFont('Calibri', 20, True, False)
        tips = font.render("Press any Key", True, BLACK)
        screen.blit(tips, [3,3])
        pygame.display.update()
        clock.tick(30)

def countDown(screen, clock):
    for i in range(0,90):
        screen.fill(WHITE)
        font = pygame.font.SysFont('Calibri', 100, True, False)
        time = font.render(str(3-i//30), True, BLACK)
        screen.blit(time, [200,120])
        word=""
        if i<45: word="We think"
        else: word="you look like"
        font = pygame.font.SysFont('Calibri', 40, True, False)
        word = font.render(word, True, BLACK)
        screen.blit(word, [160,220])
        pygame.display.update()
        clock.tick(30)

def resultDisplay(screen, loc1, loc2, person):
    font = pygame.font.SysFont('Calibri', 25, True, False)
    pygame.display.set_caption("GUI")

    pic1 = pygame.image.load(loc1)
    pic1 = pygame.transform.scale(pic1, (200, 300))
    pic2 = pygame.image.load(loc2)
    pic2 = pygame.transform.scale(pic2, (200, 300))
    person = font.render(person, True, BLACK)
    you = font.render("YOU", True, BLACK)

    screen.fill(WHITE)
    screen.blit(you, [20, 320])
    screen.blit(person, [260, 320])
    screen.blit(pic1, [20, 20])
    screen.blit(pic2, [260, 20])

    x3=390;y3=370;
    w3=100;h3=30;
    pygame.draw.rect(screen, BLACK, (x3, y3, w3, h3), 0)
    font = pygame.font.SysFont('Calibri', 20, True, False)
    again = font.render("play again", True, WHITE)
    screen.blit(again, [395, 375])
    pygame.display.flip()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pos()[0] >= x3 and pygame.mouse.get_pos()[1] >= y3:
                    if pygame.mouse.get_pos()[0] <= x3+w3 and pygame.mouse.get_pos()[1] <= y3+h3:
                        return 1