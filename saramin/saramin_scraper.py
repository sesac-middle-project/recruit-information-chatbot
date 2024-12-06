from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

# 크롤링할 직무 리스트
job_titles = [ 
    '데이터 분석가',
    '데이터 엔지니어',
    'AI 개발자',
    '챗봇 개발자',
    '클라우드 엔지니어',
    'API 개발자',
    '머신러닝 엔지니어',
    '데이터 사이언티스트'
]

# URL 설정 및 ChromeDriver 실행
url = 'https://www.saramin.co.kr/zf_user/'

# 데이터 저장용 리스트
all_results = []

# 전체 크롤링 시간 기록 시작
start_time = time.time()

# 각 직무에 대해 반복
for query in job_titles:
    print(f'{query} 크롤링 시작')
    
    # 개별 직무 크롤링 시간 기록 시작
    job_start_time = time.time()

    driver = webdriver.Chrome()
    time.sleep(2)
    driver.maximize_window()  # 창 크기 최대화
    driver.get(url)
    time.sleep(5)
    
    # 검색 버튼 클릭
    search = driver.find_element(By.CLASS_NAME, 'btn_search').click()

    # 검색창에 입력
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="ipt_keyword_recruit"]'))
    )
    search_box.clear()  # 기존 입력값 초기화
    search_box.send_keys(query)
    time.sleep(1)
    # search_box.send_keys(Keys.RETURN)
    driver.find_element(By.XPATH, '//*[@id="btn_search_recruit"]').click()
    time.sleep(3)
    
    # 각 페이지 반복 크롤링 (5페이지까지)
    for page in range(0, 5):
        print(f'{query} 크롤링 {page + 1} 페이지 작업 중')
        try:
            # 채용 공고 추출
            job_listings = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'item_recruit'))
            )
            
            for job in job_listings:
                try:
                    link_element = job.find_element(By.CLASS_NAME, 'job_tit').find_element(By.TAG_NAME, 'a')
                    # 구인공고 제목
                    title = link_element.get_attribute('title')
                    # 구인공고 url
                    href = link_element.get_attribute('href')
                    # 구인공고 업체명
                    corp_name = job.find_element(By.CLASS_NAME, 'corp_name').find_element(By.TAG_NAME, 'a').text
                    # 구인공고 지역
                    area_element = job.find_element(By.XPATH, './/div[@class="job_condition"]/span[1]/a[1]')
                    area = area_element.text
                    # 구인공고 지역(구)
                    try:
                        area_gu_element = job.find_element(By.XPATH, './/div[@class="job_condition"]/span[1]/a[2]')
                        area_gu = area_gu_element.text
                    except Exception:
                        area_gu = None 
                    # 구인공고 경력, 학력
                    experience_element = job.find_element(By.XPATH, './/div[@class="job_condition"]/span[2]')
                    experience = experience_element.text
                    education_element = job.find_element(By.XPATH, './/div[@class="job_condition"]/span[3]')
                    education = education_element.text
                    # 구인공고 마감일
                    date = job.find_element(By.XPATH, './/div[@class="job_date"]/span[@class="date"]').text

                    all_results.append({'회사': corp_name, '제목': title, 'URL': href, '경력' : experience, '학력' : education, '지역':area,'지역(구)':area_gu ,'마감일':date, '직무': query})
                except Exception as e:
                    print(f"오류 발생: {e}")
            if page > 0:
                # 다음 페이지로 이동
                next_button = job.find_element(By.XPATH, f'//*[@id="recruit_info_list"]/div[2]/div/a[{page}]')
                next_button.click()
                time.sleep(5) 
        
        except Exception as e:
            print(f"페이지 이동 중 오류 발생: {e}")
            break

    # 드라이버 종료
    driver.quit()
    time.sleep(2)

    # 개별 직무 크롤링 시간 기록 종료
    job_end_time = time.time()
    print(f'{query} 크롤링 완료. 소요 시간: {round(job_end_time - job_start_time, 2)}초')

# 전체 크롤링 시간 기록 종료
end_time = time.time()
print(f"전체 크롤링 완료. 소요 시간: {round(end_time - start_time, 2)}초")


# 데이터 CSV 저장
output_file = 'saramin_jobs.csv'

# 파일이 이미 존재하면 삭제
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"기존 파일 삭제 완료: {output_file}")

# 새로운 데이터 저장
df = pd.DataFrame(all_results)
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"CSV 저장 완료: {output_file}")
