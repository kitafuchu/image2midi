import cv2
import numpy as np
from pathlib import Path
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

# 拡大縮小(アスペクト比保持)
def scale(img, target_h):

    aspect = img.shape[1] / img.shape[0]
    target_w = int(target_h * aspect)
    
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

# xDoGで枠線抽出
def xdog(gray, sigma=0.8, k=1.4, epsilon=1.5, phi=10.0):

    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)

    dog = cv2.absdiff(g1, g2)

    xdog = 1.0 + np.tanh(phi * (dog - epsilon))
    xdog = np.clip(xdog * 255.0 / 2.0, 0, 255).astype(np.uint8)
    
    return xdog

# 2値化
def thresh(gray, threshold=127, rad=5):

    g = cv2.GaussianBlur(gray, (rad, rad), 0)

    _, bin = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)

    return bin

# MIDIに変換・書き出し
def saveAsMidi(img, file_name="tmp"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))

    h, w = img.shape[:2]

    l = 120
    notes = []

    for i in range(h):
        for j in range(w):
            if(img[i, j]):
                notes.append((j*l, Message("note_on", note=127-i)))
                notes.append(((j+1)*l, Message("note_off", note=127-i)))

    notes.sort(key=lambda x: x[0])

    prev = 0
    for t, msg in notes:
        msg.time = t - prev
        track.append(msg)
        prev = t
    
    mid.save(file_name+".mid")

    return 0

path = "sample/kotoha.png"

contour = thresh(xdog(cv2.imread(path, cv2.IMREAD_GRAYSCALE)), 95)

comp = scale(contour, 128)

#cv2.imshow("result", contour)
#cv2.waitKey(0)

saveAsMidi(comp, Path(path).stem)