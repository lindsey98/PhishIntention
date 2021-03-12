import easyocr
import os
import cv2
import time
import numpy as np
import torch
import re
from phishpedia.inference import pred_siamese
from phishpedia.utils import resolution_alignment
from PIL import Image


def ocr_detector(buttons, img_path, ocr_model):
    '''
    Detect OCR for each button 
    params buttons: Nx4 array/tensor
    params img_path: str/np.ndarray
    params ocr_model: OCR model
    returns alltext: list of text for each button
    '''
    
    buttons = buttons.numpy() if isinstance(buttons, torch.Tensor) else buttons
    img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
    
    alltext = []
    allsim = []
    for b in buttons:
        # Call OCR
        crop = img[int(b[1]):int(b[3]) , int(b[0]):int(b[2]), :].copy() # crop button from original image, height first, channel last
        result = ocr_model.readtext(crop) # call OCR
        txtresult = ' '.join([x[1] for x in result]) # get all text
        alltext.append(txtresult)
        
    assert len(alltext) == len(buttons)
    return alltext


def ocr_detector_advanced(buttons, img_path, ocr_model, 
                          icon_model, icon_feat, icon_files):
    '''
    Detect OCR for each button and run icon comparison
    params buttons: Nx4 array/tensor
    params img_path: str/np.ndarray
    params ocr_model: OCR model
    returns alltext: list of text for each button
    '''
    
    buttons = buttons.numpy() if isinstance(buttons, torch.Tensor) else buttons
    img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
    
    alltext = []
    allsim = []
    for b in buttons:
        # Call OCR
        crop = img[int(b[1]):int(b[3]) , int(b[0]):int(b[2]), :].copy() # crop button from original image, height first, channel last
        result = ocr_model.readtext(crop) # call OCR
        txtresult = ' '.join([x[1] for x in result]) # get all text
        alltext.append(txtresult)
        
        # Feed into icon comparison model
        buttons_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # BGR to RGB
        button_pil = Image.fromarray(buttons_rgb) # convert to PIL.Image object
        button_feat = pred_siamese(button_pil, icon_model, grayscale=True) # feed into siamese
        sim_mat = icon_feat @ button_feat.T # compute similarity

        candidate_logo = Image.open(np.asarray(icon_files)[sim_mat == max(sim_mat)][0])
        button_pil, candidate_logo = resolution_alignment(button_pil, candidate_logo)
        img_feat = pred_siamese(button_pil, icon_model, grayscale=True)
        logo_feat = pred_siamese(candidate_logo, icon_model, grayscale=True)
        final_sim = logo_feat.dot(img_feat)
        allsim.append(final_sim)
        
    assert (len(alltext) == len(buttons)) & (len(allsim) == len(buttons))
    return alltext, allsim

def button_rank(buttons, alltext, keyword_check=True):
    '''
    Produce button rankings according to login likeliness
    params buttons: Nx4 np.ndarray/tensor
    params alltext: list of text for each button
    returns buttons: sorted buttons
    '''
    
    # rank by location first, top-right button get higher rank
    buttons = buttons.numpy() if isinstance(buttons, torch.Tensor) else buttons
    buttons = buttons.tolist()
#     idx = [i[0] for i in sorted(enumerate(buttons), key=lambda x: ((x[1][1]+x[1][3])//2, 2000-x[1][0]))]
#     buttons = [i[1] for i in sorted(enumerate(buttons), key=lambda x: ((x[1][1]+x[1][3])//2, 2000-x[1][0]))]
#     alltext = np.asarray(alltext)[idx]
#     alltext = alltext.tolist()
    
    if keyword_check == True:
        # regex on keywords, if the login/signin keywords are found, shift this button to the top
        j = len(buttons) - 1 # check starting from the end to the front
        stopindex = 0
        while j >= stopindex:
            keyword_finder = re.findall('(log)|(sign)|(submit)|(register)|(confirm)|(next)|(continue)|(validate)|(verify)|(access)|(create.*account)|(join now)|(登入)|(登录)|(登錄)|(注册)|(Anmeldung)|(iniciar sesión)|(s\'identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(ВОЙТИ)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar)|(giriş yap)|(เข้าสู่ระบบ)|(สมัครสมาชิก)', alltext[j].lower()) 
            
            if len(keyword_finder) > 0:
                buttons.insert(0, buttons.pop(j)) # shift this button to the front
                alltext.insert(0, alltext.pop(j)) # shift ocr text to the front
                stopindex += 1 # no need to check again
                
            else:
                j -= 1 # go to next button
            
    buttons = np.asarray(buttons)
    return buttons, alltext                
        

    
def button_rank_advanced(buttons, alltext, allsim, keyword_check=True):
    '''
    Produce button rankings according to login likeliness
    params buttons: Nx4 np.ndarray/tensor
    params alltext: list of text for each button
    returns buttons: sorted buttons
    '''
    
    # rank by location first, top-right button get higher rank
    buttons = buttons.numpy() if isinstance(buttons, torch.Tensor) else buttons
    idx = [i[0] for i in sorted(enumerate(buttons), key=lambda x: ((x[1][1]+x[1][3])//2, 2000-x[1][0]))]
    buttons = [i[1] for i in sorted(enumerate(buttons), key=lambda x: ((x[1][1]+x[1][3])//2, 2000-x[1][0]))]
    alltext = np.asarray(alltext)[idx]
    allsim = np.asarray(allsim)[idx]
    alltext = alltext.tolist()
    allsim = allsim.tolist()
    
    if keyword_check == True:
        # regex on keywords, if the login/signin keywords are found, shift this button to the top
        j = len(buttons) - 1 # check starting from the end to the front
        stopindex = 0
        while j >= stopindex:
            keyword_finder = re.findall('(log)|(sign)|(submit)|(register)|(confirm)|(next)|(continue)|(validate)|(verify)|(access)|(create.*account)|(join now)|(登入)|(登录)|(登錄)|(注册)|(Anmeldung)|(iniciar sesión)|(s\'identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(ВОЙТИ)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar)|(giriş yap)|(เข้าสู่ระบบ)|(สมัครสมาชิก)', alltext[j].lower()) 
            
            if len(keyword_finder) > 0 or allsim[j] >= 0.7:
                buttons.insert(0, buttons.pop(j)) # shift this button to the front
                alltext.insert(0, alltext.pop(j)) # shift ocr text
                allsim.insert(0, allsim.pop(j)) # shift sim 
                stopindex += 1 # no need to check again
                
            else:
                j -= 1 # go to next button
            
    buttons = np.asarray(buttons)
    return buttons, alltext, allsim

if __name__ == '__main__':
    # load this model only ONCE
    reader = easyocr.Reader(['en'])
    
    # |(登入)|(登录)|(登錄)|(注册)|(Anmeldung)|(iniciar sesión)|(s\'identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(ВОЙТИ)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar)|(giriş yap)|(เข้าสู่ระบบ)|(สมัครสมาชิก)
    
    