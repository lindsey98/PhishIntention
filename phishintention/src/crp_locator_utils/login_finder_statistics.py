import pandas as pd
import json
import os
import shutil

################# Write failed cases for 460 legitimate ###############################
# get predicted URL
with open('./datasets/460_legitimate_detectedURL_eager_obfuscate.json', 'rt', encoding='utf-8') as handle:
    urldict = json.load(handle)

# read gt URLs
gt = pd.read_table('./datasets/gt_loginurl_for460.txt')
gt['folder'] = gt['base_url'].apply(lambda x: x.split('//')[1])
total = 0
correct = 0
wrong_cases = []
all_cases = []

for base_url in set(list(gt['folder'])):
    gt_login = list(gt.loc[gt['folder'] == base_url]['url'])
    if not base_url in urldict.keys() and base_url not in os.listdir('./datasets/1003_legitimate_loginbutton_labelled/460_legitimate'):
        continue
    if not base_url in urldict.keys():
        print('Not found in prediction dict', base_url)
        all_cases.append(base_url)
        wrong_cases.append(base_url)
        total += 1
        continue

    reported_urls = [x for x in urldict[base_url]]
    found = False

    for url in reported_urls:
        if url in gt_login:
            correct += 1
            found = True
            break
        # TODO: match only the front before spm=, tmall use spm to record data
        elif url.split('spm=')[0] in [x.split('spm=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('client_id=')[0] in [x.split('client_id=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('state=')[0] in [x.split('state=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif 'login' in url or 'signin' in url or 'signup' in url or 'register' in url or 'sign_in' in url or 'log_in' in url or 'sign_up' in url:
            correct += 1
            found = True
            break
        elif url.split('_ga')[0] in [x.split('_ga')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('?vrid=')[0] in [x.split('?vrid=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('interaction/')[0] in [x.split('interaction/')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('?_gl=')[0] in [x.split('?_gl=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('adobe_mc=')[0] in [x.split('adobe_mc=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('?login_challenge=')[0] in [x.split('?login_challenge=')[0] for x in gt_login]:
            correct += 1
            found = True
            break
        elif url.split('&client_sdrn')[0] in [x.split('&client_sdrn')[0] for x in gt_login]:
            correct += 1
            found = True
            break
    if not found:
        wrong_cases.append(base_url)
    total += 1
    all_cases.append(base_url)

print(correct, total)
#
with open('./datasets/fail_login_finder_eager_460_obfuscate.txt', 'w') as f:
    pass
for case in wrong_cases:
    with open('./datasets/fail_login_finder_eager_460_obfuscate.txt', 'a+') as f:
        f.write(case+'\n')

################ Write failed 600 legitimate #################
# with open('./datasets/600_legitimate_detectedURL_eager_HTML.json', 'rt', encoding='utf-8') as handle:
#     urldict = json.load(handle)
#
# # read gt URLs
# gt = pd.read_table('./datasets/gt_loginurl_for600.txt')
# total = 0
# correct = 0
# wrong_cases = []
# all_cases = []
#
# for base_url in set(list(gt['folder'])):
#     gt_login = list(gt.loc[gt['folder'] == base_url]['url'])
#     if not base_url in urldict.keys() and base_url not in os.listdir('./datasets/600_legitimate'):
#         continue
#     if not base_url in urldict.keys():
#         print('Not found in prediction dict', base_url)
#         all_cases.append(base_url)
#         wrong_cases.append(base_url)
#         total += 1
#         continue
#
#     reported_urls = [x for x in urldict[base_url]]
#     found = False
#
#     for url in reported_urls:
#         if url in gt_login:
#             correct += 1
#             found = True
#             break
#         # TODO: match only the front before spm=, tmall use spm to record data
#         elif url.split('spm=')[0] in [x.split('spm=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('client_id=')[0] in [x.split('client_id=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('state=')[0] in [x.split('state=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif 'login' in url or 'signin' in url or 'signup' in url or 'register' in url or 'sign_in' in url or 'log_in' in url or 'sign_up' in url:
#             correct += 1
#             found = True
#             break
#         elif url.split('_ga')[0] in [x.split('_ga')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('?vrid=')[0] in [x.split('?vrid=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('interaction/')[0] in [x.split('interaction/')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('?_gl=')[0] in [x.split('?_gl=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('adobe_mc=')[0] in [x.split('adobe_mc=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('?login_challenge=')[0] in [x.split('?login_challenge=')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#         elif url.split('&client_sdrn')[0] in [x.split('&client_sdrn')[0] for x in gt_login]:
#             correct += 1
#             found = True
#             break
#     if not found:
#         wrong_cases.append(base_url)
#     total += 1
#     all_cases.append(base_url)
#
# print(correct, total)
#
# with open('./datasets/fail_login_finder_eager_HTML.txt', 'w') as f:
#     pass
# for case in wrong_cases:
#     with open('./datasets/fail_login_finder_eager_HTML.txt', 'a+') as f:
#         f.write(case+'\n')
#

######### Error composition ###########################
# timeout_die = ["https://fitbit.com", "https://userscloud.com", "https://hostgator.com", "https://freeadult.games", "https://bs.to", "https://lanacion.com.ar", "https://wuxiaworld.com",
#                "https://sap.com", "https://mts.ru", "https://klikbca.com", "https://pochta.ru", "https://egypt.gov.eg", "https://sonyliv.com", "https://folha.uol.com.br", "https://smartsheet.com",
#                "https://ksl.com", "https://ticketmonster.co.kr", "https://seek.com.au", "https://overstock.com", "https://eztv.ag", "https://zhiding.cn"]
# interact_fail = ["https://vmware.com", "https://ea3w.com", "https://yinyuetai.com", "https://lazada.com.my", "https://lazada.vn", "https://sportbox.ru",
#                  "https://bedbathandbeyond.com", "https://kommersant.ru", "https://qatarairways.com"]
# nokeyword = ["https://correios.com.br", "https://rockstargames.com", "https://olx.com.eg", "https://umn.edu", "https://css-tricks.com",
#              "https://diariolibre.com", "https://teletica.com", "https://netpnb.com", "https://dailystar.co.uk", "https://neoldu.com", "https://scmp.com", "https://sapo.ao"]
# popupwindow = ["https://jusbrasil.com.br", "https://dkb.de", "https://shopee.com.my", "https://xm.com",
#                "https://arbeitsagentur.de", "https://jd.id", "https://eldiario.es", "https://blocket.se",
#                "https://tvn24.pl", "https://made-in-china.com", "https://1und1.de", "https://virginmedia.com", "https://chefkoch.de",
#                "https://duden.de", "https://nouvelobs.com", "https://blibli.com",
#                "https://netgear.com", "https://cimbclicks.com.my", "https://listindiario.com"]
# needtointeract = ["https://auto.ru", "https://gameforge.com"]
# nologin = ["https://bintang.com"]
# eager_mode_notload = ["https://novinky.cz", "https://noticias.uol.com.br", "https://tinhte.vnlife.ru"]
# know_engine = ["https://thekitchn.com", "https://drupal.org", "https://evite.com", "https://fnac.com",
#                "https://rentalcars.com", "https://rarbgto.org", "https://sweetwater.com", "https://skyscanner.net"]
# persian_language = ["https://filimo.com"]
# # TODO: dropdown login (ignore): https://sme.sk, https://orbitz.com, https://malwarebytes.com, https://lds.org, https://thairath.co.th, https://lidl.de, https://bulbagarden.net, https://cleartax.in, https://hostgator.com, https://magicbricks.com, https://laposte.fr, https://dreamstime.com, https://index.hu, https://www.flightradar24.com
#
#
# error600 = [x.strip() for x in open("./datasets/fail_login_finder_eager.txt").readlines()]
# error460 = [x.strip() for x in open("./datasets/fail_login_finder_eager_460.txt").readlines()]
# print(len(error600))
# print(len(error460))
#
# ct_timeout = 0
# ct_interactfail = 0
# ct_nokeyword = 0
# ct_popup = 0
# ct_needinteract = 0
# ct_nologin = 0
# ct_eager = 0
# ct_engine = 0
# ct_persian = 0
# for domain in error600:
#     url = open(os.path.join('./datasets/600_legitimate', domain, 'info.txt')).read()
#     if url in timeout_die:
#         ct_timeout += 1
#     elif url in interact_fail:
#         ct_interactfail += 1
#     elif url in nokeyword:
#         ct_nokeyword += 1
#     elif url in popupwindow:
#         ct_popup += 1
#     elif url in needtointeract:
#         ct_needinteract += 1
#     elif url in nologin:
#         ct_nologin += 1
#     elif url in eager_mode_notload:
#         ct_eager += 1
#     elif url in know_engine:
#         ct_engine += 1
#     elif url in persian_language:
#         ct_persian += 1
#     else:
#         print(domain, ' not found')
#
#
# for domain in error460:
#     url = open(os.path.join('./datasets/460_legitimate', domain, 'info.txt')).read()
#     if url in timeout_die:
#         ct_timeout += 1
#     elif url in interact_fail:
#         ct_interactfail += 1
#     elif url in nokeyword:
#         ct_nokeyword += 1
#     elif url in popupwindow:
#         ct_popup += 1
#     elif url in needtointeract:
#         ct_needinteract += 1
#     elif url in nologin:
#         ct_nologin += 1
#     elif url in eager_mode_notload:
#         ct_eager += 1
#     elif url in know_engine:
#         ct_engine += 1
#     elif url in persian_language:
#         ct_persian += 1
#     else:
#         print(domain, ' not found')
#
#
# print('Timeout/die : {} \n Cannot interact properly: {} \n No keyword: {} \n Popup window block: {}'.format(ct_timeout, ct_interactfail, ct_nokeyword, ct_popup))
# print('Need to interact before entering the website : {} \n No login, ignore: {} \n Eager mode not load completely : {} \n Website know that I am an engine: {} \n Persian language: {}'.format(ct_needinteract, ct_nologin, ct_eager, ct_engine, ct_persian))
# print('Count total = {}'.format(ct_timeout + ct_interactfail + ct_nokeyword + ct_popup + ct_needinteract + ct_nologin + ct_engine + ct_eager + ct_persian))
#
