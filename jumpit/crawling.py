from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import numpy as np
import os
import time

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# 직업 리스트
job_titles = ['데이터 분석', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']

# 크롬 드라이버 경로 설정
service = Service(executable_path='C:/Users/RMARKET/sesac/project4/recruit-information-chatbot/chromedriver-win64/chromedriver.exe')

# 전체 데이터를 저장할 빈 DataFrame 생성
all_job_data = pd.DataFrame()

# 직업 별로 크롤링
for query in job_titles:
    driver = webdriver.Chrome(service=service)
    driver.get('https://jumpit.saramin.co.kr/')

    popup = driver.window_handles
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="modal"]/div/div/div/header/button'))).click()
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '/html/body/main/div/div[3]/div/div/button'))).click()
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '/html/body/header/div/div/div[2]/button[2]'))).click()
    search_box = driver.find_element(By.CSS_SELECTOR, 'input')
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(3)

    for _ in range(10):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    job_data = []

    for i in range(1, 201):
        try:
            block = driver.find_element(By.CSS_SELECTOR, f'body > main > div > section.sc-c12e57e5-3.gjgpzi > section > div:nth-child({i})')

            company = block.find_element(By.CSS_SELECTOR, 'a > div.sc-15ba67b8-0.kkQQfR > div > div').text
            title = block.find_element(By.CSS_SELECTOR, 'a').get_attribute('title')
            url = block.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')

            locexp = block.find_elements(By.CSS_SELECTOR, 'a > div.sc-15ba67b8-0.kkQQfR > ul.sc-15ba67b8-1.cdeuol > li')
            loc = locexp[0].text
            exp = locexp[1].text

            job_data.append([company, title, url, loc, exp])
        
        except Exception as e:
            print(f"Error at index {i}: {e}")

    df = pd.DataFrame(job_data, columns=['company', 'title', 'url', 'loc', 'exp'])
    df['job_name'] = query

    all_job_data = pd.concat([all_job_data, df], ignore_index=True)

    driver.quit()

all_job_data.to_csv('crawling_job_all.csv', encoding='utf-8-sig', index=False)