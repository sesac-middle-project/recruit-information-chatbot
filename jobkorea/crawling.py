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
job_titles = ['데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']

# 크롬 드라이버 경로 설정
service = Service(executable_path='C:/Users/RMARKET/sesac/project4/recruit-information-chatbot/chromedriver-win64/chromedriver.exe')

# 전체 데이터를 저장할 빈 DataFrame 생성
all_job_data = pd.DataFrame()

# 직업별로 크롤링
for query in job_titles:
    # 크롬 드라이버 실행
    driver = webdriver.Chrome(service=service)
    driver.get('https://www.jobkorea.co.kr/')

    # 검색 상자에 키워드 입력 후 검색
    search_box = driver.find_element(By.CSS_SELECTOR, "input#stext")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    # 2번째 페이지로 이동
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="dev-content-wrap"]/article/section[1]/article[3]/button[2]'))).click()

    # 페이지 번호를 제외한 현재 페이지의 전체 URL 저장
    current = driver.current_url[:-1]

    # 데이터 프레임 초기화
    job_data = []
    num = 1

    while num <= 10:  # 페이지 번호가 10 이하일 경우만 크롤링
        # 다음 페이지로 넘어가기 위해 URL에 페이지 번호 추가하고 이동
        next_url = current + str(num)
        driver.get(next_url)

        # 회사 이름 가져오기
        company_name = driver.find_elements(By.CSS_SELECTOR, "a.corp-name-link.dev-view")
        link_name = [name.get_attribute('title') for name in company_name]

        # 구인 게시물 제목 가져오기
        title = driver.find_elements(By.CSS_SELECTOR, "div.information-title > a")
        link_title = [tit.text for tit in title]

        # URL 가져오기
        url = driver.find_elements(By.CSS_SELECTOR, "div.information-title > a")
        link_url = [u.get_attribute('href') for u in url]

         # 자격 요건, 설명 등 가져오기
        options = driver.find_elements(By.CSS_SELECTOR, "ul.chip-information-group > li")

        op = 0
        for j in range(len(link_name)):
            exp, edu, typ, loc, date = None, None, None, None, None
            try:
                # "chip-benefit-group" 요소가 있는지 확인
                benefit_group = driver.find_elements(By.CSS_SELECTOR, "ul.chip-benefit-group")

                if benefit_group:
                    li_elements = driver.find_elements(By.XPATH, f'//*[@id="dev-content-wrap"]/article/section[1]/article[2]/article[{j+1}]/div[2]/ul[1]/li')
                    if len(li_elements) == 5:
                        # 값 추출
                        exp = options[op].text
                        edu = options[op + 1].text
                        typ = options[op + 2].text
                        loc = options[op + 3].text
                        date = options[op + 4].text

                        job_data.append([link_name[j], link_title[j], link_url[j], exp, edu, typ, loc, date, query])

                    else:
                        print(f'{link_title[j]}에는 5개의 항목이 존재하지 않습니다')
                        exp, edu, typ, loc, date = None, None, None, None, None
                        if len(li_elements) == 4:
                            op += 4
                        elif len(li_elements) == 3:
                            op += 3
                        elif len(li_elements) == 2:
                            op += 2
                        elif len(li_elements) == 1:
                            op += 1
                        else:
                            pass
                        continue

                else:
                    # "chip-benefit-group"이 없으면 다른 XPath를 사용하여 li[5] 존재 여부 확인
                    li_elements = driver.find_elements(By.XPATH, f'//*[@id="dev-content-wrap"]/article/section[1]/article[2]/article[{j+1}]/div[2]/ul/li')
                    if len(li_elements) == 5:
                        # 값 추출
                        exp = options[op].text
                        edu = options[op + 1].text
                        typ = options[op + 2].text
                        loc = options[op + 3].text
                        date = options[op + 4].text

                        job_data.append([link_name[j], link_title[j], link_url[j], exp, edu, typ, loc, date, query])

                    else:
                        print(f'{link_title[j]}에는 5개의 항목이 존재하지 않습니다')
                        exp, edu, typ, loc, date = None, None, None, None, None
                        if len(li_elements) == 4:
                            op += 4
                        elif len(li_elements) == 3:
                            op += 3
                        elif len(li_elements) == 2:
                            op += 2
                        elif len(li_elements) == 1:
                            op += 1
                        else:
                            pass
                        continue

                op += 5  # 각 공고의 항목 개수를 처리

            except NoSuchElementException:
                print(f"공고 {link_title[j]}에 필요한 요소가 없습니다.")
                continue

        num += 1
        time.sleep(3)

    # 수집한 데이터를 DataFrame으로 변환
    df = pd.DataFrame(job_data, columns=['company', 'title', 'url', 'exp', 'edu', 'typ', 'loc', 'date', 'job_name'])

    # 'None' 값이 포함된 행을 제거하고 추가
    df = df.dropna(axis=0, how='any')  # NaN이 있는 행을 아예 삭제
    all_job_data = pd.concat([all_job_data, df], ignore_index=True)

    # 브라우저 종료
    driver.quit()

# 전체 데이터를 한 번에 CSV 파일로 저장
all_job_data.to_csv('crawling_job_all.csv', encoding='utf-8-sig', index=False)